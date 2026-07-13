from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ae_frontend import ae_decode_latent, ae_encode_latent, ae_reconstruct, load_ae_model
from bridge_scheduler import LinearBrownianBridgeScheduler
from datasets_strict_bbdm import SingleDomainSliceDataset
from models_strict_bbdm import StrictBridgeUNet


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sample with pseudo-paired strict-style BBDM")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--pair_mode", type=str, default=None, choices=["label_random", "paired_csv"])
    ap.add_argument("--target_test_csv", type=str, default=None)
    ap.add_argument("--target_test_root", type=str, default=None)
    ap.add_argument("--source_csv", type=str, default=None)
    ap.add_argument("--source_root", type=str, default=None)
    ap.add_argument("--sample_root", type=str, default=None)
    ap.add_argument("--image_size", type=int, default=None)
    ap.add_argument("--num_inference_steps", type=int, default=None)
    ap.add_argument("--num_train_timesteps", type=int, default=None)
    ap.add_argument("--bridge_sigma", type=float, default=None)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--base_channels", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--reverse_eta", type=float, default=None)
    ap.add_argument("--clamp_each_step", type=int, default=None)
    ap.add_argument("--diagnostics_json", type=str, default=None)
    ap.add_argument("--diagnostics_csv", type=str, default=None)

    ap.add_argument("--use_ae_frontend", type=int, default=None)
    ap.add_argument("--ae_ckpt", type=str, default=None)
    ap.add_argument("--ae_config", type=str, default=None)
    ap.add_argument("--ae_freeze", type=int, default=None)
    ap.add_argument("--ae_input_mode", type=str, default=None, choices=["raw", "recon", "latent"])
    ap.add_argument("--save_ae_vis", type=int, default=None)
    return ap.parse_args()


def _default_config() -> Dict[str, Any]:
    return {
        "exp_name": "bbdm_kneemri_to_mrnet_strict",
        "checkpoint": "/home/mnnu/Standford2KennMRIData1/experiments/bbdm_strict_runs/bbdm_kneemri_to_mrnet_strict/checkpoints/latest.pt",
        "pair_mode": "label_random",
        "target_test_csv": "/home/mnnu/Standford2KennMRIData1/experiments/idea2_diffusion/cache_kneemri_test/test_0_cached_slices.csv",
        "target_test_root": None,
        "source_csv": "/home/mnnu/Standford2KennMRIData1/experiments/idea2_diffusion/cache_mrnet_train/train_0_cached_slices.csv",
        "source_root": None,
        "sample_root": "/home/mnnu/Standford2KennMRIData1/experiments/bbdm_strict_runs",
        "image_size": 128,
        "num_inference_steps": 50,
        "num_train_timesteps": 1000,
        "bridge_sigma": 1.0,
        "num_samples": 64,
        "batch_size": 4,
        "seed": 42,
        "base_channels": 64,
        "device": "cuda",
        "reverse_eta": 0.35,
        "clamp_each_step": 0,
        "diagnostics_json": None,
        "diagnostics_csv": None,

        "use_ae_frontend": 0,
        "ae_ckpt": "/home/mnnu/Standford2KennMRIData1/experiments/ae_unet_knee_pretrain/checkpoints/best.pt",
        "ae_config": None,
        "ae_freeze": 1,
        "ae_input_mode": "raw",
        "save_ae_vis": 0,
    }


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg[k] = v
    return cfg


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(v)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_tensor_png(image_chw: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = image_chw.detach().cpu().clamp(-1, 1)
    image = ((image + 1.0) * 0.5 * 255.0).round().to(torch.uint8).squeeze(0).numpy()
    Image.fromarray(image, mode="L").save(out_path)


def _tensor_stats(image_chw: torch.Tensor) -> Dict[str, float]:
    x = image_chw.detach().float().cpu()
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "var": float(x.var(unbiased=False).item()),
    }


def _clip_ratio(image_chw: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> Dict[str, float]:
    x = image_chw.detach().float().cpu()
    return {
        "clip_low_ratio": float((x <= lo).float().mean().item()),
        "clip_high_ratio": float((x >= hi).float().mean().item()),
    }


def _edge_strength(image_chw: torch.Tensor) -> float:
    x = image_chw.detach().float().cpu()
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
    gy[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    gmag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return float(gmag.mean().item())


def _edge_map_u8(image_chw: torch.Tensor) -> np.ndarray:
    x = image_chw.detach().float().cpu()
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
    gy[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    gmag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    gmag = gmag / (gmag.max() + 1e-6)
    return (gmag.squeeze(0).numpy() * 255.0).astype(np.uint8)


def _aggregate_stats(items: List[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not items:
        return out
    keys = list(items[0].keys())
    for k in keys:
        vals = np.array([float(it[k]) for it in items], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
        out[f"{k}_p10"] = float(np.percentile(vals, 10))
        out[f"{k}_p50"] = float(np.percentile(vals, 50))
        out[f"{k}_p90"] = float(np.percentile(vals, 90))
    return out


def _build_reverse_indices(num_inference_steps: int, num_train_timesteps: int) -> List[int]:
    if num_train_timesteps < 3:
        raise ValueError("num_train_timesteps must be >= 3 for pseudo-paired BBDM-style reverse sampling")

    raw = torch.linspace(num_train_timesteps, 0, steps=max(num_inference_steps, 2)).round().long().tolist()
    indices: List[int] = []
    for v in raw:
        iv = int(max(0, min(num_train_timesteps, v)))
        if len(indices) == 0 or indices[-1] != iv:
            indices.append(iv)

    if indices[0] != num_train_timesteps:
        indices = [num_train_timesteps] + indices
    if indices[-1] != 0:
        indices.append(0)
    return indices


def _sample_reverse_bridge(
    model: StrictBridgeUNet,
    bridge: LinearBrownianBridgeScheduler,
    x_b: torch.Tensor,
    num_steps: int,
    num_train_timesteps: int,
    reverse_eta: float,
    clamp_each_step: bool,
) -> torch.Tensor:
    bsz = x_b.shape[0]
    t_list = _build_reverse_indices(num_steps, num_train_timesteps)

    x_cur = x_b.clone()

    for i, t in enumerate(t_list[:-1]):
        s = int(t_list[i + 1])

        t_tensor = torch.full((bsz,), int(t), device=x_b.device, dtype=torch.long)
        s_tensor = torch.full((bsz,), int(s), device=x_b.device, dtype=torch.long)

        pred_bb = model(x_t=x_cur, timesteps=t_tensor)

        if s == 0:
            x_cur = bridge.recover_xa_from_bridge_target(x_t=x_cur, bridge_target_hat=pred_bb)
            x_cur = x_cur.clamp(-1.0, 1.0)
            break

        noise = torch.randn_like(x_cur)
        x_cur = bridge.step_stochastic_from_bridge_target(
            x_t=x_cur,
            x_b=x_b,
            bridge_target_hat=pred_bb,
            t_index=t_tensor,
            s_index=s_tensor,
            noise=noise,
            eta=float(reverse_eta),
        )
        if clamp_each_step:
            x_cur = x_cur.clamp(-1.0, 1.0)

    return x_cur


def _build_source_label_index(src_ds: SingleDomainSliceDataset) -> Dict[int, List[int]]:
    label_to_indices: Dict[int, List[int]] = {}
    for i in range(len(src_ds.df)):  # type: ignore[attr-defined]
        row = src_ds.df.iloc[i]  # type: ignore[attr-defined]
        if src_ds.label_col and pd.notna(row[src_ds.label_col]):  # type: ignore[attr-defined]
            y = int(row[src_ds.label_col])  # type: ignore[attr-defined]
            label_to_indices.setdefault(y, []).append(i)
    return label_to_indices


def _save_ae_vis_panel(path: Path, before: torch.Tensor, ae_recon: torch.Tensor, translated: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_u8(x: torch.Tensor) -> np.ndarray:
        y = x.detach().cpu().clamp(-1, 1)
        return ((y + 1.0) * 0.5 * 255.0).round().to(torch.uint8).squeeze(0).numpy()

    b = to_u8(before)
    r = to_u8(ae_recon)
    t = to_u8(translated)
    resid = np.abs(t.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    edge = _edge_map_u8(translated)
    canvas = np.concatenate([b, r, t, resid, edge], axis=1)
    Image.fromarray(canvas, mode="L").save(path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))

    use_ae_frontend = _as_bool(cfg.get("use_ae_frontend", 0))
    ae_input_mode = str(cfg.get("ae_input_mode", "raw")).lower()
    save_ae_vis = _as_bool(cfg.get("save_ae_vis", 0))

    out_dir = Path(cfg["sample_root"]) / str(cfg["exp_name"]) / "samples"
    before_dir = out_dir / "before"
    ae_recon_dir = out_dir / "ae_recon"
    translated_dir = out_dir / "translated"
    source_ref_dir = out_dir / "source_refs"
    ae_vis_dir = out_dir / "ae_vis"
    for d in [before_dir, translated_dir, source_ref_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if use_ae_frontend:
        ae_recon_dir.mkdir(parents=True, exist_ok=True)
    if save_ae_vis:
        ae_vis_dir.mkdir(parents=True, exist_ok=True)

    tgt_ds = SingleDomainSliceDataset(
        csv_path=str(cfg["target_test_csv"]),
        image_size=int(cfg["image_size"]),
        root_dir=cfg.get("target_test_root"),
    )
    src_ds = SingleDomainSliceDataset(
        csv_path=str(cfg["source_csv"]),
        image_size=int(cfg["image_size"]),
        root_dir=cfg.get("source_root"),
    )
    src_label_index = _build_source_label_index(src_ds)

    tgt_loader = DataLoader(tgt_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)

    device = _resolve_device(str(cfg["device"]))

    ae_model = None
    if use_ae_frontend:
        ae_ckpt = cfg.get("ae_ckpt")
        if not ae_ckpt:
            raise ValueError("use_ae_frontend=true requires --ae_ckpt")
        ae_model = load_ae_model(
            ae_ckpt=str(ae_ckpt),
            ae_config=cfg.get("ae_config"),
            device=device,
            freeze=_as_bool(cfg.get("ae_freeze", 1)),
        )

    bridge_in_channels = 1
    bridge_sample_size = int(cfg["image_size"])
    if use_ae_frontend and ae_input_mode == "latent":
        if ae_model is None:
            raise ValueError("ae_input_mode=latent requires use_ae_frontend=true")
        with torch.no_grad():
            dummy = torch.zeros((1, 1, int(cfg["image_size"]), int(cfg["image_size"])), device=device)
            lat_dummy = ae_encode_latent(ae_model, dummy)["latent"]
        bridge_in_channels = int(lat_dummy.shape[1])
        bridge_sample_size = int(lat_dummy.shape[-1])

    model = StrictBridgeUNet(
        image_size=bridge_sample_size,
        base_channels=int(cfg["base_channels"]),
        in_channels=bridge_in_channels,
        out_channels=bridge_in_channels,
    )
    ckpt = torch.load(Path(str(cfg["checkpoint"])), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    bridge = LinearBrownianBridgeScheduler(
        num_steps=int(cfg["num_train_timesteps"]), bridge_sigma=float(cfg["bridge_sigma"])
    )

    produced = 0
    rows = []
    before_stats_list: List[Dict[str, float]] = []
    translated_stats_list: List[Dict[str, float]] = []

    with torch.no_grad():
        for batch in tgt_loader:
            if produced >= int(cfg["num_samples"]):
                break

            x_b_raw = batch["image"].to(device)
            t_paths = batch["path"]
            t_labels = batch["label"].tolist()

            x_b_recon = x_b_raw
            if ae_model is not None:
                x_b_recon = ae_reconstruct(ae_model, x_b_raw)["recon"]

            if ae_model is not None and ae_input_mode == "latent":
                b_pack = ae_encode_latent(ae_model, x_b_raw)
                x_b_bridge = b_pack["latent"]
                translated_bridge = _sample_reverse_bridge(
                    model=model,
                    bridge=bridge,
                    x_b=x_b_bridge,
                    num_steps=int(cfg["num_inference_steps"]),
                    num_train_timesteps=int(cfg["num_train_timesteps"]),
                    reverse_eta=float(cfg.get("reverse_eta", 1.0)),
                    clamp_each_step=bool(int(cfg.get("clamp_each_step", 0))),
                )
                translated = ae_decode_latent(ae_model, translated_bridge, encoder_features=b_pack["encoder_features"])["recon"]
            else:
                x_b_in = x_b_recon if (ae_model is not None and ae_input_mode == "recon") else x_b_raw
                translated = _sample_reverse_bridge(
                    model=model,
                    bridge=bridge,
                    x_b=x_b_in,
                    num_steps=int(cfg["num_inference_steps"]),
                    num_train_timesteps=int(cfg["num_train_timesteps"]),
                    reverse_eta=float(cfg.get("reverse_eta", 1.0)),
                    clamp_each_step=bool(int(cfg.get("clamp_each_step", 0))),
                )

            for i in range(x_b_raw.shape[0]):
                if produced >= int(cfg["num_samples"]):
                    break
                sid = f"{produced:06d}"
                before_png = before_dir / f"{sid}.png"
                after_png = translated_dir / f"{sid}.png"

                _save_tensor_png(x_b_raw[i], before_png)
                _save_tensor_png(translated[i], after_png)

                ae_recon_png = ""
                if ae_model is not None:
                    ae_recon_path = ae_recon_dir / f"{sid}.png"
                    _save_tensor_png(x_b_recon[i], ae_recon_path)
                    ae_recon_png = str(ae_recon_path)

                if save_ae_vis:
                    vis_path = ae_vis_dir / f"{sid}.png"
                    _save_ae_vis_panel(vis_path, x_b_raw[i], x_b_recon[i], translated[i])

                b_stats = _tensor_stats(x_b_raw[i])
                t_stats = _tensor_stats(translated[i])
                b_stats["edge"] = _edge_strength(x_b_raw[i])
                t_stats["edge"] = _edge_strength(translated[i])
                b_stats.update(_clip_ratio(x_b_raw[i]))
                t_stats.update(_clip_ratio(translated[i]))
                before_stats_list.append(b_stats)
                translated_stats_list.append(t_stats)

                target_label = int(t_labels[i])
                if target_label in src_label_index and len(src_label_index[target_label]) > 0:
                    src_idx = random.choice(src_label_index[target_label])
                else:
                    src_idx = random.randrange(len(src_ds))

                src_item = src_ds[src_idx]
                src_png = source_ref_dir / f"{sid}.png"
                _save_tensor_png(src_item["image"], src_png)

                note = (
                    "pseudo-paired/class-consistent data assumption; "
                    "objective=bbdm_bridge_target; "
                    "reverse=bbdm_style_stochastic_with_bridge_target; "
                    "optional ae_frontend raw/recon/latent"
                )

                rows.append(
                    {
                        "sample_id": sid,
                        "target_path": str(t_paths[i]),
                        "target_label": target_label,
                        "before_png": str(before_png),
                        "ae_recon_png": ae_recon_png,
                        "translated_png": str(after_png),
                        "source_ref_path": str(src_item["path"]),
                        "source_ref_label": int(src_item["label"]),
                        "source_ref_png": str(src_png),
                        "checkpoint": str(cfg["checkpoint"]),
                        "pair_mode": str(cfg["pair_mode"]),
                        "use_ae_frontend": int(use_ae_frontend),
                        "ae_input_mode": ae_input_mode,
                        "before_min": b_stats["min"],
                        "before_max": b_stats["max"],
                        "before_mean": b_stats["mean"],
                        "before_std": b_stats["std"],
                        "before_var": b_stats["var"],
                        "before_edge": b_stats["edge"],
                        "before_clip_low_ratio": b_stats["clip_low_ratio"],
                        "before_clip_high_ratio": b_stats["clip_high_ratio"],
                        "translated_min": t_stats["min"],
                        "translated_max": t_stats["max"],
                        "translated_mean": t_stats["mean"],
                        "translated_std": t_stats["std"],
                        "translated_var": t_stats["var"],
                        "translated_edge": t_stats["edge"],
                        "translated_clip_low_ratio": t_stats["clip_low_ratio"],
                        "translated_clip_high_ratio": t_stats["clip_high_ratio"],
                        "note": note,
                    }
                )
                produced += 1

    out_csv = out_dir / "sample_pairs.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    diag = {
        "n_samples": int(len(rows)),
        "reverse_eta": float(cfg.get("reverse_eta", 1.0)),
        "clamp_each_step": bool(int(cfg.get("clamp_each_step", 0))),
        "use_ae_frontend": int(use_ae_frontend),
        "ae_input_mode": ae_input_mode,
        "before": _aggregate_stats(before_stats_list),
        "translated": _aggregate_stats(translated_stats_list),
    }
    if before_stats_list and translated_stats_list:
        b_var = np.array([d["var"] for d in before_stats_list], dtype=np.float64)
        t_var = np.array([d["var"] for d in translated_stats_list], dtype=np.float64)
        b_edge = np.array([d["edge"] for d in before_stats_list], dtype=np.float64)
        t_edge = np.array([d["edge"] for d in translated_stats_list], dtype=np.float64)
        b_mean = np.array([d["mean"] for d in before_stats_list], dtype=np.float64)
        t_mean = np.array([d["mean"] for d in translated_stats_list], dtype=np.float64)
        b_std = np.array([d["std"] for d in before_stats_list], dtype=np.float64)
        t_std = np.array([d["std"] for d in translated_stats_list], dtype=np.float64)
        diag["ratios"] = {
            "translated_over_before_mean_ratio": float((t_mean.mean() + 1e-8) / (b_mean.mean() + 1e-8)),
            "translated_over_before_std_ratio": float((t_std.mean() + 1e-8) / (b_std.mean() + 1e-8)),
            "translated_over_before_edge_ratio": float((t_edge.mean() + 1e-8) / (b_edge.mean() + 1e-8)),
            "var_ratio_translated_over_before": float((t_var.mean() + 1e-8) / (b_var.mean() + 1e-8)),
            "edge_ratio_translated_over_before": float((t_edge.mean() + 1e-8) / (b_edge.mean() + 1e-8)),
        }

    diag_path = Path(str(cfg["diagnostics_json"])) if cfg.get("diagnostics_json") else (out_dir / "diagnostics_summary.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)

    diag_csv_path = Path(str(cfg["diagnostics_csv"])) if cfg.get("diagnostics_csv") else (out_dir / "diagnostics_per_sample.csv")
    if rows:
        cols = [
            "sample_id",
            "before_mean", "translated_mean",
            "before_std", "translated_std",
            "before_edge", "translated_edge",
            "before_min", "translated_min",
            "before_max", "translated_max",
            "before_clip_low_ratio", "before_clip_high_ratio",
            "translated_clip_low_ratio", "translated_clip_high_ratio",
            "use_ae_frontend", "ae_input_mode",
        ]
        pd.DataFrame(rows)[cols].to_csv(diag_csv_path, index=False)

    with open(out_dir / "sample_config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(
        "Pseudo-paired strict-style BBDM sampling finished "
        "(objective and reverse both use bridge target). "
        f"{len(rows)} samples saved to: {out_dir}"
    )
    print(f"Pair csv: {out_csv}")
    print(f"Diagnostics: {diag_path}")
    print(f"Per-sample diagnostics: {diag_csv_path}")


if __name__ == "__main__":
    main()
