from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BridgeState:
    m_t: torch.Tensor
    delta_t: torch.Tensor
    sigma_t: torch.Tensor


@dataclass
class ReverseBridgeTargetCoefficients:
    coef_xt: torch.Tensor
    coef_y: torch.Tensor
    coef_pred: torch.Tensor
    post_var: torch.Tensor


class LinearBrownianBridgeScheduler:
    """
    Pseudo-paired strict-style BBDM scheduler with bridge-target-consistent reverse.

    Forward marginal:
      q(x_t | x_A, x_B) = N((1-m_t)*x_A + m_t*x_B, delta_t * I)

    with discrete t in [0..T], m_t=t/T, and:
      delta_t = bridge_sigma^2 * m_t * (1-m_t)

    Bridge target used for training:
      bb_t = m_t * (x_B - x_A) + sqrt(delta_t) * eps

    Note: under this forward process, bb_t = x_t - x_A.
    This implementation can still be pseudo-paired (label_random), thus it is
    not official fully paired BBDM.
    """

    def __init__(self, num_steps: int = 1000, bridge_sigma: float = 1.0, eps: float = 1e-6) -> None:
        self.num_steps = int(num_steps)
        self.bridge_sigma = float(bridge_sigma)
        self.eps = float(eps)

    def _normalize_t(self, t_index: torch.Tensor) -> torch.Tensor:
        # Discrete grid [0..T], where T=num_steps.
        return t_index.float() / max(self.num_steps, 1)

    def get_state(self, t_index: torch.Tensor) -> BridgeState:
        m_t = self._normalize_t(t_index).clamp(0.0, 1.0)
        delta_t = (self.bridge_sigma ** 2) * torch.clamp(m_t * (1.0 - m_t), min=0.0)
        sigma_t = torch.sqrt(torch.clamp(delta_t, min=0.0))
        return BridgeState(m_t=m_t, delta_t=delta_t, sigma_t=sigma_t)

    def sample_xt(self, x_a: torch.Tensor, x_b: torch.Tensor, t_index: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        state = self.get_state(t_index)
        m = state.m_t.view(-1, 1, 1, 1)
        s = state.sigma_t.view(-1, 1, 1, 1)
        return (1.0 - m) * x_a + m * x_b + s * noise

    def make_bridge_target(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        t_index: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        state = self.get_state(t_index)
        m = state.m_t.view(-1, 1, 1, 1)
        s = state.sigma_t.view(-1, 1, 1, 1)
        return m * (x_b - x_a) + s * noise

    def recover_xa_from_bridge_target(self, x_t: torch.Tensor, bridge_target_hat: torch.Tensor) -> torch.Tensor:
        # bb_t = x_t - x_A  =>  x_A = x_t - bb_t
        return x_t - bridge_target_hat

    def _transition_coefficients(self, t_index: torch.Tensor, s_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Discrete bridge transition coefficients for s < t:
          x_t = alpha * x_s + beta * x_B + eta,  eta~N(0, trans_var I)
        """
        st = self.get_state(t_index)
        ss = self.get_state(s_index)

        m_t = st.m_t.view(-1, 1, 1, 1)
        m_s = ss.m_t.view(-1, 1, 1, 1)
        d_t = st.delta_t.view(-1, 1, 1, 1)
        d_s = ss.delta_t.view(-1, 1, 1, 1)

        one_minus_ms = torch.clamp(1.0 - m_s, min=self.eps)
        alpha = torch.clamp((1.0 - m_t) / one_minus_ms, min=0.0)
        beta = m_t - alpha * m_s
        trans_var = torch.clamp(d_t - (alpha ** 2) * d_s, min=0.0)
        return alpha, beta, trans_var

    def reverse_bridge_target_coefficients(
        self,
        t_index: torch.Tensor,
        s_index: torch.Tensor,
    ) -> ReverseBridgeTargetCoefficients:
        """
        Reverse coefficients using bridge-target prediction (s < t):
          x_s mean = c_xt * x_t + c_y * x_B - c_pred * bb_hat_t
          x_s var  = post_var

        bb_hat_t is the model prediction for bridge target at time t.
        """
        if torch.any(s_index >= t_index):
            raise ValueError("reverse_bridge_target_coefficients expects s_index < t_index")

        st = self.get_state(t_index)
        ss = self.get_state(s_index)

        m_s = ss.m_t.view(-1, 1, 1, 1)
        d_s = ss.delta_t.view(-1, 1, 1, 1)

        alpha, beta, trans_var = self._transition_coefficients(t_index=t_index, s_index=s_index)

        inv_d_s = 1.0 / torch.clamp(d_s, min=self.eps)
        inv_trans = 1.0 / torch.clamp(trans_var, min=self.eps)

        # Posterior with latent x_A:
        # mean = k_xt*x_t + k_xa*x_A + k_xb*x_B
        post_var = 1.0 / (inv_d_s + (alpha ** 2) * inv_trans)
        k_xt = post_var * alpha * inv_trans
        k_xa = post_var * (1.0 - m_s) * inv_d_s
        k_xb = post_var * (m_s * inv_d_s - alpha * beta * inv_trans)

        # Replace x_A via bridge target: x_A = x_t - bb_t
        coef_xt = k_xt + k_xa
        coef_y = k_xb
        coef_pred = k_xa

        return ReverseBridgeTargetCoefficients(
            coef_xt=coef_xt,
            coef_y=coef_y,
            coef_pred=coef_pred,
            post_var=torch.clamp(post_var, min=0.0),
        )

    def reverse_mean_variance_from_bridge_target(
        self,
        x_t: torch.Tensor,
        x_b: torch.Tensor,
        bridge_target_hat: torch.Tensor,
        t_index: torch.Tensor,
        s_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coef = self.reverse_bridge_target_coefficients(t_index=t_index, s_index=s_index)
        mean = coef.coef_xt * x_t + coef.coef_y * x_b - coef.coef_pred * bridge_target_hat
        return mean, coef.post_var

    def step_stochastic_from_bridge_target(
        self,
        x_t: torch.Tensor,
        x_b: torch.Tensor,
        bridge_target_hat: torch.Tensor,
        t_index: torch.Tensor,
        s_index: torch.Tensor,
        noise: torch.Tensor,
        eta: float = 1.0,
    ) -> torch.Tensor:
        mean, var = self.reverse_mean_variance_from_bridge_target(
            x_t=x_t,
            x_b=x_b,
            bridge_target_hat=bridge_target_hat,
            t_index=t_index,
            s_index=s_index,
        )
        eta_t = float(max(0.0, eta))
        return mean + eta_t * torch.sqrt(torch.clamp(var, min=0.0)) * noise

    def loss_weight(self, t_index: torch.Tensor, mode: str = "none") -> torch.Tensor:
        """
        Optional timestep weighting for bridge-target regression.
        """
        mode_l = str(mode).lower()
        if mode_l == "none":
            return torch.ones_like(t_index, dtype=torch.float32)
        state = self.get_state(t_index)
        d = state.delta_t
        if mode_l == "inv_delta":
            return 1.0 / torch.clamp(d, min=1e-4)
        raise ValueError(f"Unsupported loss weighting mode: {mode}")
