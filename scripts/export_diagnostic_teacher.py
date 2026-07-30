#!/usr/bin/env python3
"""Export a source diagnostic classifier as a UNSB-compatible TorchScript teacher.

The exported module is consumed by ``dosc_modules.diagnostic_non_degradation_loss``, which
scores the TRANSLATED image. A teacher trained on raw source PNGs only is out of
distribution on that rendering (measured here: acc 0.984 raw vs 0.578 on the 256px
generator-input rendering), and then the margin term penalises appearance change instead of
diagnostic damage -- a leash toward the identity mapping. Export the checkpoint produced by
``scripts/train_render_robust_teacher.py``, whose summary reports per-rendering AUC; the
per-rendering block is copied into the metadata below so a mis-scoped teacher is visible at
a glance instead of silently distorting training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval_existing_classifier_on_csv import build_model


class UNSBDiagnosticTeacher(nn.Module):
    """Adapt UNSB [-1, 1] tensors to the source classifier input contract.

    The classifier was trained behind ``Resize((S, S), antialias=True)``, so the resize here
    is antialiased too. Without it, downsampling 256 -> 224 aliases the speckle the teacher
    reads, which shifts the margin for reasons that have nothing to do with the lesion.
    """

    def __init__(self, classifier: nn.Module, image_size: int) -> None:
        super().__init__()
        self.classifier = classifier
        self.image_size = int(image_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return self.classifier(image)


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state-dict-like mapping")
    for key in ("state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", default="custom_resnet50_space")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--trace_size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    classifier = build_model(
        args.backbone,
        num_classes=args.num_classes,
        pretrained="none",
        device=device,
    )
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = {
        key.replace("module.", ""): value
        for key, value in extract_state_dict(checkpoint).items()
    }
    load_result = classifier.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Teacher checkpoint/model mismatch: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    wrapper = UNSBDiagnosticTeacher(classifier.eval(), args.image_size).to(device).eval()
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    example = torch.zeros(
        2,
        3,
        args.trace_size,
        args.trace_size,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        expected = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=True)
        actual = traced(example)
    if expected.shape != (2, args.num_classes):
        raise RuntimeError(f"Unexpected teacher output shape: {tuple(expected.shape)}")
    if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-4):
        raise RuntimeError("TorchScript validation failed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frozen = torch.jit.freeze(traced.eval())
    torch.jit.save(frozen, str(args.out))
    metadata = {
        "weights": str(args.weights.resolve()),
        "backbone": args.backbone,
        "num_classes": args.num_classes,
        "image_size": args.image_size,
        "input_contract": "B3HW float tensor in [-1,1]",
        "output_contract": "unnormalized class logits",
        "resize": "bilinear, antialias=True (matches the classifier's training transform)",
    }
    # Carry the per-rendering validation through, so anyone reading the exported teacher can
    # see whether it was ever validated on translated images.
    renderings = checkpoint.get("val_metrics") if isinstance(checkpoint, dict) else None
    if isinstance(renderings, dict):
        metadata["source_val_per_rendering"] = renderings
        aucs = [v.get("auc") for v in renderings.values() if isinstance(v, dict) and "auc" in v]
        if aucs:
            metadata["worst_rendering_auc"] = min(aucs)
            if len(aucs) == 1:
                print("[warn] teacher was validated on ONE rendering only; if that rendering is "
                      "raw source PNGs it is out of distribution on translated images")
    else:
        print("[warn] checkpoint carries no per-rendering validation. If this teacher was "
              "trained on raw source images only, the DOSC margin term will penalise "
              "appearance change rather than diagnostic damage. See "
              "scripts/train_render_robust_teacher.py")
    metadata_path = args.out.with_suffix(args.out.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved teacher: {args.out}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
