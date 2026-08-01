#!/usr/bin/env python3
"""Test whether target-reference code leakage causally changes diagnosis.

For each held-out labeled source case, the script generates translations under
K different unlabeled target-train references while reusing the same bridge
noise at every reference. It compares that style-only variation with a
fixed-reference control that changes only bridge noise.

A linear diagnostic probe is fitted on source-train style features and applied
to target references only to obtain a "malignancy-like" reference score. No
target diagnosis label is read. The decisive statistic is whether references
with a higher probe score systematically increase the translated teacher score
for the same source case.

This is a reporting-only mechanism audit. It does not train or select the
translator, teacher, calibration, threshold, or target-domain classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(iterable, **_kwargs):
        """Fall back to an unwrapped iterable in minimal environments."""

        return iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.unsb.style_swap_metrics import (  # noqa: E402
    make_path_noises,
    summarize_style_swap,
)

ID_COLUMN_CANDIDATES = ("case_id", "key", "id", "image_id")
PATH_COLUMN_CANDIDATES = (
    "path",
    "image_path",
    "filepath",
    "file_path",
    "src_path",
    "raw",
)
TRAIN_LIKE_SPLITS = {"src_train", "train", "src_valid", "validation", "val"}
NONTRAIN_TARGET_TOKENS = ("test", "valid", "validation", "val")


@dataclass
class LoadedModel:
    model: object
    options: object
    transform: object
    device: torch.device


def parse_csv_list(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(sorted({int(item) for item in parse_csv_list(value)}))
    if any(step < 1 for step in steps):
        raise ValueError("steps are one-based and must be positive")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--unsb_root", type=Path, required=True)
    parser.add_argument("--checkpoints_dir", type=Path, required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--epoch", default="latest")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tag", default="trsc")

    parser.add_argument("--probe_manifest", type=Path, required=True)
    parser.add_argument("--probe_splits", default="src_train")
    parser.add_argument("--probe_id_col", default="auto")
    parser.add_argument("--probe_path_col", default="src_path")
    parser.add_argument("--probe_label_col", default="label")

    parser.add_argument("--source_manifest", type=Path, required=True)
    parser.add_argument("--source_split", default="src_test")
    parser.add_argument("--source_id_col", default="auto")
    parser.add_argument("--source_path_col", default="auto")
    parser.add_argument("--source_label_col", default="label")
    parser.add_argument("--source_images_root", type=Path)
    parser.add_argument("--source_images_subdir", default="real")
    parser.add_argument("--allow_nonheldout_source", action="store_true")

    parser.add_argument("--target_manifest", type=Path, required=True)
    parser.add_argument(
        "--target_split",
        default="",
        help="Optional target-train split value; empty means the whole manifest is target train",
    )
    parser.add_argument("--target_id_col", default="auto")
    parser.add_argument("--target_path_col", default="auto")
    parser.add_argument("--allow_nontrain_target_refs", action="store_true")

    parser.add_argument("--references", type=int, default=12)
    parser.add_argument("--reference_batch_size", type=int, default=4)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument("--steps", default="1,5")
    parser.add_argument("--score_threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    parser.add_argument("--permutation_draws", type=int, default=2000)
    parser.add_argument("--probe_c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_sources", type=int, default=0)

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_timesteps", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--netG", default="resnet_9blocks_cond")
    parser.add_argument("--normG", default="instance")
    parser.add_argument("--input_nc", type=int, default=3)
    parser.add_argument("--output_nc", type=int, default=3)
    parser.add_argument("--dosc_style_dim", type=int, default=128)
    parser.add_argument("--dosc_encoder_widths", default="32,64,128,256")
    projection = parser.add_mutually_exclusive_group()
    projection.add_argument(
        "--dosc_enable_projection",
        action="store_true",
        help="Evaluate the legacy projection ablation",
    )
    projection.add_argument(
        "--dosc_disable_projection",
        action="store_true",
        help="Deprecated compatibility flag; projection is already disabled by default",
    )
    parser.add_argument("--no_antialias", action="store_true")
    parser.add_argument("--no_antialias_up", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _choose_column(
    frame: pd.DataFrame,
    requested: str,
    candidates: tuple[str, ...],
    description: str,
) -> str:
    if requested != "auto":
        if requested not in frame.columns:
            raise ValueError(
                f"{description} column {requested!r} is absent; "
                f"available={list(frame.columns)}"
            )
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        f"could not infer {description} column from {list(frame.columns)}; "
        "pass it explicitly"
    )


def _resolve_path(value: object, manifest: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (manifest.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    return str(path.resolve())


def load_labeled_manifest(
    manifest: Path,
    splits: tuple[str, ...],
    id_col: str,
    path_col: str,
    label_col: str,
    images_root: Path | None = None,
    images_subdir: str = "",
) -> pd.DataFrame:
    manifest = manifest.expanduser().resolve()
    frame = pd.read_csv(manifest)
    if splits:
        if "split" not in frame.columns:
            raise ValueError(f"{manifest} has no split column")
        frame = frame[frame["split"].astype(str).isin(splits)].copy()
    if frame.empty:
        raise ValueError(f"{manifest}:{splits} is empty")
    id_column = _choose_column(
        frame,
        id_col,
        ID_COLUMN_CANDIDATES,
        "case identifier",
    )
    if label_col not in frame.columns:
        raise ValueError(f"{manifest} has no label column {label_col!r}")
    frame = frame.rename(columns={id_column: "case_id", label_col: "label"}).copy()
    frame["case_id"] = frame["case_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    labels = set(frame["label"].unique())
    if labels - {0, 1}:
        raise ValueError(f"{manifest} contains non-binary labels {sorted(labels)}")
    if frame["case_id"].duplicated().any():
        examples = frame.loc[frame["case_id"].duplicated(), "case_id"].head(5).tolist()
        raise ValueError(f"{manifest}:{splits} has duplicate cases {examples}")

    inferred_path = None
    if path_col == "auto":
        for candidate in PATH_COLUMN_CANDIDATES:
            if candidate in frame.columns:
                inferred_path = candidate
                break
    elif path_col in frame.columns:
        inferred_path = path_col
    if inferred_path is not None:
        frame["path"] = [
            _resolve_path(value, manifest) for value in frame[inferred_path]
        ]
    elif images_root is not None:
        root = images_root.expanduser().resolve()
        if images_subdir:
            root = root / images_subdir
        paths = [root / f"{case_id}.png" for case_id in frame["case_id"]]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} source images are missing; first={missing[0]}"
            )
        frame["path"] = [str(path.resolve()) for path in paths]
    else:
        raise ValueError(
            f"{manifest} has no usable image path column and no images root was given"
        )
    return frame[["case_id", "label", "path"]].reset_index(drop=True)


def load_target_manifest(
    manifest: Path,
    split: str,
    id_col: str,
    path_col: str,
) -> pd.DataFrame:
    manifest = manifest.expanduser().resolve()
    header = pd.read_csv(manifest, nrows=0)
    id_column = _choose_column(
        header,
        id_col,
        ID_COLUMN_CANDIDATES,
        "target reference identifier",
    )
    path_column = _choose_column(
        header,
        path_col,
        PATH_COLUMN_CANDIDATES,
        "target reference path",
    )
    use_columns = [id_column, path_column]
    if split:
        if "split" not in header.columns:
            raise ValueError(f"{manifest} has no split column")
        use_columns.append("split")
    # Read no target diagnosis values, even if the CSV contains label columns.
    frame = pd.read_csv(manifest, usecols=use_columns)
    if split:
        frame = frame[frame["split"].astype(str) == split].copy()
    if frame.empty:
        raise ValueError(f"{manifest}:{split or 'all'} is empty")
    output = pd.DataFrame(
        {
            "case_id": frame[id_column].astype(str),
            "path": [_resolve_path(value, manifest) for value in frame[path_column]],
        }
    )
    if output["case_id"].duplicated().any():
        examples = (
            output.loc[output["case_id"].duplicated(), "case_id"].head(5).tolist()
        )
        raise ValueError(f"{manifest} has duplicate target references {examples}")
    return output.reset_index(drop=True)


def validate_protocol(
    args: argparse.Namespace,
    probe: pd.DataFrame,
    source: pd.DataFrame,
) -> None:
    overlap = sorted(set(probe["case_id"]) & set(source["case_id"]))
    if overlap:
        raise ValueError(
            f"probe/source evaluation case overlap ({len(overlap)}): {overlap[:5]}"
        )
    if (
        not args.allow_nonheldout_source
        and args.source_split.lower() in TRAIN_LIKE_SPLITS
    ):
        raise ValueError(
            "style-swap evaluation requires a held-out source analysis split; "
            "pass --allow_nonheldout_source only for a non-final smoke run"
        )
    target_split = args.target_split.lower()
    if (
        target_split
        and not args.allow_nontrain_target_refs
        and any(token in target_split for token in NONTRAIN_TARGET_TOKENS)
    ):
        raise ValueError(
            "target references must come from target train; "
            "target validation/test references are excluded"
        )


def load_upstream_model(args: argparse.Namespace) -> LoadedModel:
    unsb_root = args.unsb_root.expanduser().resolve()
    required = (
        unsb_root / "options" / "test_options.py",
        unsb_root / "models" / "dosc_sb_model.py",
        unsb_root / "models" / "dosc_modules.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "install the DOSC overlay into the upstream UNSB checkout first; "
            f"missing={missing}"
        )
    sys.path.insert(0, str(unsb_root))
    from data.base_dataset import get_transform
    from models import create_model
    from options.test_options import TestOptions

    if args.netG != "resnet_9blocks_cond":
        raise ValueError(
            "the DOSC style-swap audit requires the conditioned "
            "resnet_9blocks_cond generator"
        )
    gpu_id = args.gpu_id if torch.cuda.is_available() and args.gpu_id >= 0 else -1
    option_args = [
        "--dataroot",
        str(unsb_root),
        "--name",
        args.experiment_name,
        "--checkpoints_dir",
        str(args.checkpoints_dir.expanduser().resolve()),
        "--model",
        "trsc_sb",
        "--dataset_mode",
        "trsc_unaligned",
        "--direction",
        "AtoB",
        "--phase",
        "test",
        "--epoch",
        args.epoch,
        "--gpu_ids",
        str(gpu_id),
        "--preprocess",
        "resize_and_crop",
        "--load_size",
        str(args.crop_size),
        "--crop_size",
        str(args.crop_size),
        "--num_timesteps",
        str(args.num_timesteps),
        "--tau",
        str(args.tau),
        "--ngf",
        str(args.ngf),
        "--normG",
        args.normG,
        "--input_nc",
        str(args.input_nc),
        "--output_nc",
        str(args.output_nc),
        "--dosc_style_dim",
        str(args.dosc_style_dim),
        "--dosc_encoder_widths",
        args.dosc_encoder_widths,
        "--dosc_noise_ratio",
        "0.0",
        "--no_flip",
        "--serial_batches",
        "--eval",
    ]
    if args.dosc_enable_projection:
        option_args.append("--dosc_enable_projection")
    elif args.dosc_disable_projection:
        option_args.append("--dosc_disable_projection")
    if args.no_antialias:
        option_args.append("--no_antialias")
    if args.no_antialias_up:
        option_args.append("--no_antialias_up")

    original_argv = sys.argv
    try:
        sys.argv = ["eval_dosc_style_swap.py", *option_args]
        options = TestOptions().parse()
    finally:
        sys.argv = original_argv
    model = create_model(options)
    model.setup(options)
    model.eval()
    transform = get_transform(options)
    return LoadedModel(
        model=model,
        options=options,
        transform=transform,
        device=model.device,
    )


def load_image_tensor(path: str, transform: object) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


@torch.no_grad()
def extract_style_features(
    loaded: LoadedModel,
    paths: list[str],
    batch_size: int,
) -> dict[str, np.ndarray]:
    style_module = loaded.model._style_module()
    stages: dict[str, list[np.ndarray]] = {
        "raw": [],
        "projected": [],
        "condition": [],
    }
    for start in range(0, len(paths), int(batch_size)):
        images = torch.stack(
            [
                load_image_tensor(path, loaded.transform)
                for path in paths[start : start + int(batch_size)]
            ]
        ).to(loaded.device)
        raw = style_module.encode_raw(images)
        projected = style_module.projector(raw)
        active = projected if style_module.enable_projection else raw
        condition = style_module.to_generator_style(active)
        for name, tensor in (
            ("raw", raw),
            ("projected", projected),
            ("condition", condition),
        ):
            stages[name].append(tensor.detach().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in stages.items()}


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    seed: int,
) -> object:
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("linear probe fitting requires both binary classes")
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=5000,
            random_state=int(seed),
            solver="liblinear",
        ),
    )
    probe.fit(features, labels)
    return probe


def probe_score(probe: object, features: np.ndarray) -> np.ndarray:
    return np.asarray(probe.decision_function(features), dtype=float)


def probe_report(
    probe_features: dict[str, np.ndarray],
    probe_labels: np.ndarray,
    source_features: dict[str, np.ndarray],
    source_labels: np.ndarray,
    target_features: dict[str, np.ndarray],
    domain_train_indices: np.ndarray,
    domain_eval_indices: np.ndarray,
    c_value: float,
    seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    report: dict[str, object] = {"diagnostic": {}, "domain": {}}
    diagnostic_probes = {}
    for stage in ("raw", "projected", "condition"):
        diagnostic = fit_probe(
            probe_features[stage],
            probe_labels,
            c_value,
            seed,
        )
        diagnostic_probes[stage] = diagnostic
        source_score = probe_score(diagnostic, source_features[stage])
        report["diagnostic"][stage] = {
            "auc": float(roc_auc_score(source_labels, source_score)),
            "random_baseline": 0.5,
        }

        domain_train_x = np.concatenate(
            [
                probe_features[stage],
                target_features[stage][domain_train_indices],
            ],
            axis=0,
        )
        domain_train_y = np.concatenate(
            [
                np.zeros(len(probe_features[stage]), dtype=int),
                np.ones(len(domain_train_indices), dtype=int),
            ]
        )
        domain = fit_probe(domain_train_x, domain_train_y, c_value, seed)
        domain_eval_x = np.concatenate(
            [
                source_features[stage],
                target_features[stage][domain_eval_indices],
            ],
            axis=0,
        )
        domain_eval_y = np.concatenate(
            [
                np.zeros(len(source_features[stage]), dtype=int),
                np.ones(len(domain_eval_indices), dtype=int),
            ]
        )
        domain_score = probe_score(domain, domain_eval_x)
        report["domain"][stage] = {
            "auc": float(roc_auc_score(domain_eval_y, domain_score)),
            "random_baseline": 0.5,
        }
    return report, diagnostic_probes


def teacher_scores(teacher: torch.jit.ScriptModule, images: torch.Tensor) -> np.ndarray:
    logits = teacher(images)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            f"teacher must return binary logits [B,2], got {tuple(logits.shape)}"
        )
    return (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()


def _slice_path_noises(
    noises: tuple[torch.Tensor, ...],
    start: int,
    end: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(noise[start:end] for noise in noises)


@torch.no_grad()
def generate_case(
    loaded: LoadedModel,
    teacher: torch.jit.ScriptModule,
    source_row: object,
    reference_rows: pd.DataFrame,
    reference_probe_scores: np.ndarray,
    reference_tensors: torch.Tensor,
    fixed_reference_index: int,
    selected_steps: tuple[int, ...],
    reference_batch_size: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = load_image_tensor(str(source_row.path), loaded.transform)[None].to(
        loaded.device
    )
    source_score = float(teacher_scores(teacher, source)[0])
    style_module = loaded.model._style_module()
    source_style = style_module.encode_projected(source)
    reference_count = len(reference_rows)
    common_noises = make_path_noises(
        source,
        loaded.options.num_timesteps,
        reference_count,
        seed,
        common_across_batch=True,
    )
    independent_noises = make_path_noises(
        source,
        loaded.options.num_timesteps,
        reference_count,
        seed + 1,
        common_across_batch=False,
    )

    swap_rows: list[dict[str, object]] = []
    noise_rows: list[dict[str, object]] = []
    for start in range(0, reference_count, int(reference_batch_size)):
        end = min(start + int(reference_batch_size), reference_count)
        count = end - start
        source_batch = source.expand(count, -1, -1, -1)
        references = reference_tensors[start:end].to(loaded.device)
        reference_style = style_module.encode_projected(references)
        condition = style_module.to_generator_style(reference_style)
        outputs = loaded.model.translate_with_condition(
            source_batch,
            condition,
            path_noises=common_noises,
        )
        for step in selected_steps:
            output = outputs[step - 1]
            output_score = teacher_scores(teacher, output)
            output_style = style_module.encode_projected(output)
            reference_cosine = F.cosine_similarity(
                output_style,
                reference_style,
                dim=1,
            )
            source_distance = 1.0 - F.cosine_similarity(
                output_style,
                source_style.expand_as(output_style),
                dim=1,
            )
            for offset, index in enumerate(range(start, end)):
                swap_rows.append(
                    {
                        "case_id": str(source_row.case_id),
                        "label": int(source_row.label),
                        "reference_id": str(reference_rows.iloc[index]["case_id"]),
                        "reference_probe_score": float(reference_probe_scores[index]),
                        "source_score": source_score,
                        "output_score": float(output_score[offset]),
                        "output_reference_cosine": float(reference_cosine[offset]),
                        "output_style_distance_from_source": float(
                            source_distance[offset]
                        ),
                        "step": int(step),
                    }
                )

        fixed = reference_tensors[fixed_reference_index][None].to(loaded.device)
        fixed = fixed.expand(count, -1, -1, -1)
        fixed_condition = style_module.encode_condition(fixed)
        fixed_outputs = loaded.model.translate_with_condition(
            source_batch,
            fixed_condition,
            path_noises=_slice_path_noises(independent_noises, start, end),
        )
        for step in selected_steps:
            fixed_score = teacher_scores(teacher, fixed_outputs[step - 1])
            for offset, replicate in enumerate(range(start, end)):
                noise_rows.append(
                    {
                        "case_id": str(source_row.case_id),
                        "label": int(source_row.label),
                        "replicate_id": int(replicate),
                        "source_score": source_score,
                        "output_score": float(fixed_score[offset]),
                        "fixed_reference_id": str(
                            reference_rows.iloc[fixed_reference_index]["case_id"]
                        ),
                        "step": int(step),
                    }
                )
    return swap_rows, noise_rows


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    selected_steps = parse_steps(args.steps)
    if max(selected_steps) > args.num_timesteps:
        raise ValueError(
            f"requested step {max(selected_steps)} exceeds "
            f"num_timesteps={args.num_timesteps}"
        )
    if args.references < 2:
        raise ValueError("references must be at least two")
    if args.reference_batch_size < 1 or args.feature_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.probe_c <= 0.0:
        raise ValueError("probe_c must be positive")
    if args.bootstrap_draws < 0 or args.permutation_draws < 0:
        raise ValueError("bootstrap_draws and permutation_draws must be non-negative")

    probe_splits = parse_csv_list(args.probe_splits)
    probe = load_labeled_manifest(
        args.probe_manifest,
        probe_splits,
        args.probe_id_col,
        args.probe_path_col,
        args.probe_label_col,
    )
    source = load_labeled_manifest(
        args.source_manifest,
        (args.source_split,),
        args.source_id_col,
        args.source_path_col,
        args.source_label_col,
        images_root=args.source_images_root,
        images_subdir=args.source_images_subdir,
    )
    if args.max_sources > 0:
        source = source.iloc[: args.max_sources].copy()
    target = load_target_manifest(
        args.target_manifest,
        args.target_split,
        args.target_id_col,
        args.target_path_col,
    )
    validate_protocol(args, probe, source)
    if len(target) < 2 * args.references:
        raise ValueError(
            f"need at least {2 * args.references} target-train references for "
            f"disjoint domain-probe fitting and style swap, got {len(target)}"
        )

    loaded = load_upstream_model(args)
    teacher_path = args.teacher.expanduser().resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(f"teacher not found: {teacher_path}")
    teacher = torch.jit.load(str(teacher_path), map_location=loaded.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    print("Extracting source-train, held-out source, and target style features...")
    probe_features = extract_style_features(
        loaded,
        probe["path"].tolist(),
        args.feature_batch_size,
    )
    source_features = extract_style_features(
        loaded,
        source["path"].tolist(),
        args.feature_batch_size,
    )
    target_features = extract_style_features(
        loaded,
        target["path"].tolist(),
        args.feature_batch_size,
    )

    rng = np.random.default_rng(args.seed)
    target_order = rng.permutation(len(target))
    domain_split = len(target_order) // 2
    domain_train_indices = target_order[:domain_split]
    domain_eval_indices = target_order[domain_split:]
    report, diagnostic_probes = probe_report(
        probe_features,
        probe["label"].to_numpy(dtype=int),
        source_features,
        source["label"].to_numpy(dtype=int),
        target_features,
        domain_train_indices,
        domain_eval_indices,
        args.probe_c,
        args.seed,
    )

    selected_indices = np.sort(
        rng.choice(
            domain_eval_indices,
            size=args.references,
            replace=False,
        )
    )
    references = target.iloc[selected_indices].reset_index(drop=True)
    selected_target_features = {
        stage: values[selected_indices] for stage, values in target_features.items()
    }
    reference_probe_scores = probe_score(
        diagnostic_probes["condition"],
        selected_target_features["condition"],
    )
    fixed_reference_index = int(
        np.argmin(np.abs(reference_probe_scores - np.median(reference_probe_scores)))
    )
    reference_tensors = torch.stack(
        [
            load_image_tensor(path, loaded.transform)
            for path in references["path"].tolist()
        ]
    )

    swap_rows: list[dict[str, object]] = []
    noise_rows: list[dict[str, object]] = []
    for case_index, source_row in enumerate(
        tqdm(
            source.itertuples(index=False),
            total=len(source),
            desc="Style swap",
        )
    ):
        case_swap, case_noise = generate_case(
            loaded,
            teacher,
            source_row,
            references,
            reference_probe_scores,
            reference_tensors,
            fixed_reference_index,
            selected_steps,
            args.reference_batch_size,
            args.seed + 1009 * case_index,
        )
        swap_rows.extend(case_swap)
        noise_rows.extend(case_noise)

    swap_frame = pd.DataFrame(swap_rows)
    noise_frame = pd.DataFrame(noise_rows)
    summaries = {}
    for step in selected_steps:
        summaries[f"U{step}"] = summarize_style_swap(
            swap_frame[swap_frame["step"] == step].reset_index(drop=True),
            noise_frame[noise_frame["step"] == step].reset_index(drop=True),
            threshold=args.score_threshold,
            bootstrap_draws=args.bootstrap_draws,
            permutation_draws=args.permutation_draws,
            seed=args.seed + step,
        )

    checkpoint_root = args.checkpoints_dir.expanduser().resolve() / args.experiment_name
    generator_checkpoint = checkpoint_root / f"{args.epoch}_net_G.pth"
    style_checkpoint = checkpoint_root / f"{args.epoch}_net_S.pth"
    metadata = {
        "tag": args.tag,
        "protocol": {
            "source_cases": int(len(source)),
            "source_split": args.source_split,
            "probe_cases": int(len(probe)),
            "probe_splits": list(probe_splits),
            "target_reference_pool": int(len(target)),
            "selected_references": int(len(references)),
            "target_split": args.target_split or "manifest_is_target_train",
            "target_labels_used": False,
            "common_bridge_noise_across_references": True,
            "fixed_reference_independent_noise_control": True,
            "condition_noise_ratio": 0.0,
            "mechanism_audit_only": True,
        },
        "probe_report": report,
        "summary": summaries,
        "hashes": {
            "probe_manifest": sha256_file(args.probe_manifest.expanduser().resolve()),
            "source_manifest": sha256_file(args.source_manifest.expanduser().resolve()),
            "target_manifest": sha256_file(args.target_manifest.expanduser().resolve()),
            "teacher": sha256_file(teacher_path),
            "generator_checkpoint": sha256_file(generator_checkpoint),
            "style_checkpoint": sha256_file(style_checkpoint),
        },
        "model": {
            "experiment_name": args.experiment_name,
            "epoch": args.epoch,
            "projection_enabled": bool(loaded.model._style_module().enable_projection),
            "num_timesteps": args.num_timesteps,
            "tau": args.tau,
            "seed": args.seed,
        },
    }

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    swap_frame.to_csv(out_dir / "style_swap_per_pair.csv", index=False)
    noise_frame.to_csv(out_dir / "fixed_reference_noise_per_pair.csv", index=False)
    selected = references.copy()
    selected["reference_probe_score"] = reference_probe_scores
    selected["fixed_reference_control"] = False
    selected.loc[fixed_reference_index, "fixed_reference_control"] = True
    selected.to_csv(out_dir / "selected_target_references.csv", index=False)
    (out_dir / "style_swap_summary.json").write_text(
        json.dumps(json_ready(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDiagnostic leakage AUC (random baseline 0.5)")
    for stage, values in report["diagnostic"].items():
        print(f"  {stage:10s}: {values['auc']:.4f}")
    print("Domain AUC (random baseline 0.5)")
    for stage, values in report["domain"].items():
        print(f"  {stage:10s}: {values['auc']:.4f}")
    for name, summary in summaries.items():
        ci = summary["case_slope_ci95"]
        print(
            f"{name}: style std={summary['style_score_std_mean']:.4f}, "
            f"path-noise std={summary['path_noise_score_std_mean']:.4f}, "
            f"slope={summary['case_slope_mean']:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}], "
            f"reference permutation p={summary['reference_permutation_pvalue']:.4g}"
        )
    print(f"Saved audit to {out_dir}")


if __name__ == "__main__":
    main()
