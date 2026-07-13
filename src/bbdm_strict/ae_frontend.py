from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from idea2_diffusion_baseline.vqgan_pretrain.models_vqgan import (
    AEUNet,
    AEUNetConfig,
    KLVAE,
    KLVAEConfig,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    device_str = str(device) if device is not None else "cuda"
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_ch_mult(val: Any) -> tuple[int, ...]:
    if val is None:
        return (1, 2, 4, 4)
    if isinstance(val, str):
        return tuple(int(x.strip()) for x in val.split(",") if x.strip())
    return tuple(int(x) for x in val)


def _build_ae_config_from_meta(model_cfg: Optional[Dict[str, Any]], fallback_cfg: Optional[Dict[str, Any]]) -> AEUNetConfig:
    src = dict(model_cfg or {})
    if not src and fallback_cfg is not None:
        src = {
            "in_channels": int(fallback_cfg.get("in_channels", 1)),
            "out_channels": int(fallback_cfg.get("out_channels", 1)),
            "base_channels": int(fallback_cfg.get("base_channels", 128)),
            "ch_mult": _parse_ch_mult(fallback_cfg.get("ch_mult", [1, 2, 4, 4])),
            "num_res_blocks": int(fallback_cfg.get("num_res_blocks", 2)),
            "residual_out": bool(fallback_cfg.get("residual_out", True)),
        }
    if "ch_mult" in src:
        src["ch_mult"] = _parse_ch_mult(src["ch_mult"])

    return AEUNetConfig(
        in_channels=int(src.get("in_channels", 1)),
        out_channels=int(src.get("out_channels", 1)),
        base_channels=int(src.get("base_channels", 128)),
        ch_mult=_parse_ch_mult(src.get("ch_mult", [1, 2, 4, 4])),
        num_res_blocks=int(src.get("num_res_blocks", 2)),
        residual_out=bool(src.get("residual_out", True)),
    )


def _build_kl_config_from_meta(model_cfg: Optional[Dict[str, Any]], fallback_cfg: Optional[Dict[str, Any]]) -> KLVAEConfig:
    src = dict(model_cfg or {})
    if not src and fallback_cfg is not None:
        src = fallback_cfg
    return KLVAEConfig(
        in_channels=int(src.get("in_channels", 1)),
        out_channels=int(src.get("out_channels", 1)),
        base_channels=int(src.get("base_channels", 128)),
        ch_mult=_parse_ch_mult(src.get("ch_mult", [1, 2, 4, 4])),
        num_res_blocks=int(src.get("num_res_blocks", 2)),
        z_channels=int(src.get("z_channels", 4)),
    )


def freeze_ae(model: torch.nn.Module) -> torch.nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _is_kl_checkpoint(ckpt: Dict[str, Any]) -> bool:
    mt = str(ckpt.get("model_type", "")).lower()
    if mt:
        return mt == "kl"
    # Fallback: KLVAE encoder emits 2*z_channels and has no quantizer keys.
    sd = ckpt.get("model_state", {})
    return not any("quantizer" in k for k in sd) and not any(k.startswith("down_blocks") for k in sd)


def load_ae_model(
    ae_ckpt: str | Path,
    ae_config: str | Path | None = None,
    device: torch.device | str | None = None,
    freeze: bool = True,
) -> torch.nn.Module:
    dev = _resolve_device(device)
    ckpt = torch.load(Path(ae_ckpt), map_location="cpu")

    fallback_cfg = _load_json(Path(ae_config)) if ae_config else None
    model_cfg = ckpt.get("model_config", None)
    train_cfg = ckpt.get("train_config", None)
    if fallback_cfg is None and isinstance(train_cfg, dict):
        fallback_cfg = train_cfg

    if _is_kl_checkpoint(ckpt):
        cfg = _build_kl_config_from_meta(model_cfg=model_cfg, fallback_cfg=fallback_cfg)
        model: torch.nn.Module = KLVAE(cfg)
        model.is_kl = True
    else:
        cfg = _build_ae_config_from_meta(model_cfg=model_cfg, fallback_cfg=fallback_cfg)
        model = AEUNet(cfg)
        model.is_kl = False

    model.load_state_dict(ckpt["model_state"], strict=True)

    # Latent normalization factor so the bridge (bridge_sigma~1) sees ~unit-variance latents.
    scale = 1.0
    if fallback_cfg and isinstance(fallback_cfg, dict) and fallback_cfg.get("latent_scale"):
        scale = float(fallback_cfg["latent_scale"])
    model._latent_scale = float(scale)

    if freeze:
        freeze_ae(model)
    model.to(dev)
    model.eval()
    return model


def _is_kl(model: torch.nn.Module) -> bool:
    return bool(getattr(model, "is_kl", False)) or isinstance(model, KLVAE)


def ae_reconstruct(model: torch.nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    out = model(x)
    if "delta" not in out:
        out = dict(out)
        out["delta"] = out["recon"].new_zeros(())
    return out


def ae_encode_features(model: torch.nn.Module, x: torch.Tensor):
    if _is_kl(model):
        enc = model.encode(x, sample=False)
        return {"bottleneck": enc["mean"], "encoder_features": []}
    return model.encode_features(x)


def ae_encode_latent(model: torch.nn.Module, x: torch.Tensor):
    if _is_kl(model):
        enc = model.encode(x, sample=False)
        scale = float(getattr(model, "_latent_scale", 1.0))
        return {"latent": enc["mean"] * scale, "encoder_features": None}
    feat = model.encode_features(x)
    return {"latent": feat["bottleneck"], "encoder_features": feat["encoder_features"]}


def ae_decode_latent(
    model: torch.nn.Module,
    latent: torch.Tensor,
    encoder_features=None,
) -> Dict[str, torch.Tensor]:
    if _is_kl(model):
        scale = float(getattr(model, "_latent_scale", 1.0))
        recon = model.decode(latent / scale)
        return {"recon": recon.clamp(-1.0, 1.0), "delta": recon.new_zeros(())}
    dec = model.decode_from_bottleneck(bottleneck=latent, encoder_features=encoder_features)
    delta = dec["delta"]
    recon = delta.clamp(-1.0, 1.0) if model.residual_out else delta
    return {"recon": recon, "delta": delta}
