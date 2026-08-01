import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.unsb.dabrf_modules import (
        DiagnosisAwareBridgeResidualRepair,
        project_residual_radius,
        residual_radius_loss,
        target_style_progress_loss,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DABRFModuleTests(unittest.TestCase):
    def test_identity_mode_reproduces_the_unmodified_candidate(self):
        source = torch.randn(2, 3, 16, 16)
        candidate = torch.randn(2, 3, 16, 16)
        module = DiagnosisAwareBridgeResidualRepair(mode="identity")
        result = module(source, candidate)
        self.assertTrue(torch.equal(result["repaired"], candidate))
        self.assertTrue(torch.equal(result["gate"], torch.ones_like(candidate)))

    def test_nonlearned_controls_allocate_no_trainable_gate(self):
        for mode in ("identity", "fixed_scale", "norm_clip"):
            module = DiagnosisAwareBridgeResidualRepair(mode=mode)
            self.assertIsNone(module.gate_network)
            self.assertEqual(sum(p.numel() for p in module.parameters()), 0)

    def test_fixed_scale_is_an_exact_non_learned_control(self):
        source = torch.zeros(1, 3, 8, 8)
        candidate = torch.ones_like(source)
        module = DiagnosisAwareBridgeResidualRepair(
            mode="fixed_scale",
            fixed_scale=0.8,
        )
        result = module(source, candidate)
        self.assertTrue(
            torch.allclose(result["repaired"], torch.full_like(candidate, 0.8))
        )

    def test_radius_projection_enforces_the_per_case_bound(self):
        residual = torch.ones(2, 3, 8, 8)
        projected, ratio, _ = project_residual_radius(
            residual,
            residual,
            max_ratio=0.5,
        )
        self.assertTrue(torch.allclose(projected, 0.5 * residual, atol=1e-6))
        self.assertTrue(torch.allclose(ratio, torch.full_like(ratio, 0.5)))

    def test_learned_gate_starts_near_identity_and_receives_gradient(self):
        source = torch.zeros(2, 3, 8, 8)
        candidate = torch.ones_like(source, requires_grad=True)
        module = DiagnosisAwareBridgeResidualRepair(
            mode="learned",
            gate_init=0.95,
        )
        result = module(source, candidate)
        self.assertAlmostEqual(float(result["gate_mean"]), 0.95, places=5)
        result["repaired"].mean().backward()
        self.assertGreater(float(candidate.grad.abs().sum()), 0.0)
        parameter_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(parameter_gradient, 0.0)

    def test_progress_loss_rejects_excessive_style_rollback(self):
        source = torch.zeros(2, 3, 16, 16)
        reference = torch.ones_like(source)
        candidate = torch.full_like(source, 0.8)
        identity = target_style_progress_loss(
            source,
            candidate,
            candidate,
            reference,
            minimum_retention=0.8,
        )
        rolled_back = target_style_progress_loss(
            source,
            candidate,
            torch.full_like(source, 0.2),
            reference,
            minimum_retention=0.8,
        )
        self.assertAlmostEqual(float(identity["loss"]), 0.0, places=6)
        self.assertGreater(float(rolled_back["loss"]), 0.0)

    def test_radius_metric_detects_oversized_repairs(self):
        source = torch.zeros(1, 3, 8, 8)
        candidate = torch.ones_like(source)
        repaired = 1.2 * candidate
        metric = residual_radius_loss(
            source,
            candidate,
            repaired,
            max_ratio=1.0,
        )
        self.assertAlmostEqual(float(metric["ratio"]), 1.2, places=5)
        self.assertGreater(float(metric["loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
