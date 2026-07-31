import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_trsc_downstream_manifest.py"
EVALUATOR = REPO_ROOT / "scripts" / "eval_trsc_downstream.py"


class TrscDownstreamTests(unittest.TestCase):
    def _write_image(self, path: Path, value: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.full((12, 12), value, dtype=np.uint8),
            mode="L",
        ).save(path)

    def test_builder_requires_matched_real_renders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "testA_manifest.csv"
            pd.DataFrame(
                [
                    {
                        "relative_path": "testA/source_eval_0.png",
                        "case_id": "s0",
                        "label": 0,
                        "source_split": "src_train",
                    },
                    {
                        "relative_path": "testA/source_eval_1.png",
                        "case_id": "s1",
                        "label": 1,
                        "source_split": "src_valid",
                    },
                ]
            ).to_csv(source, index=False)
            for variant_index, variant in enumerate(("core", "cidp")):
                images = root / variant
                for case_index in range(2):
                    value = 40 + case_index
                    self._write_image(
                        images / "real" / f"source_eval_{case_index}.png",
                        value,
                    )
                    self._write_image(
                        images / "fake_1" / f"source_eval_{case_index}.png",
                        value + 10 + variant_index,
                    )
                    self._write_image(
                        images / "fake_5" / f"source_eval_{case_index}.png",
                        value + 20 + variant_index,
                    )
            output = root / "downstream.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--source_manifest",
                    str(source),
                    "--variant",
                    f"core={root / 'core'}",
                    "--variant",
                    f"cidp={root / 'cidp'}",
                    "--out",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            frame = pd.read_csv(output)
            self.assertEqual(
                set(frame.columns),
                {
                    "case_id",
                    "label",
                    "split",
                    "raw",
                    "core_U1",
                    "core_U5",
                    "cidp_U1",
                    "cidp_U5",
                },
            )
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertFalse(metadata["target_labels_used"])

            self._write_image(
                root / "cidp" / "real" / "source_eval_1.png",
                255,
            )
            failed = subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--source_manifest",
                    str(source),
                    "--variant",
                    f"core={root / 'core'}",
                    "--variant",
                    f"cidp={root / 'cidp'}",
                    "--out",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("raw render mismatch", failed.stderr)

    def test_evaluator_unlocks_only_complete_label_free_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            sealed = root / "sealed"
            labels = root / "target.csv"
            cases = ["b0", "b1", "m0", "m1"]
            y = [0, 0, 1, 1]
            pd.DataFrame({"case_id": cases, "label": y}).to_csv(labels, index=False)

            probabilities = {
                "raw": [0.1, 0.8, 0.2, 0.9],
                "cidp_U1": [0.1, 0.2, 0.8, 0.9],
            }
            for condition, values in probabilities.items():
                for seed in (7, 16):
                    name = f"{condition}_s{seed}"
                    run = runs / name
                    run.mkdir(parents=True)
                    sealed.mkdir(parents=True, exist_ok=True)
                    (run / "config.json").write_text(
                        json.dumps(
                            {
                                "arm": condition,
                                "seed": seed,
                                "selected_epoch": 2,
                                "selected_score": 0.9,
                                "target_labels_used": False,
                            }
                        )
                    )
                    pd.DataFrame(
                        {
                            "epoch": [1, 2],
                            "select_score": [0.8, 0.9],
                        }
                    ).to_csv(run / "history.csv", index=False)
                    rows = []
                    for epoch in (1, 2):
                        for case_id, probability in zip(cases, values):
                            rows.append(
                                {
                                    "epoch": epoch,
                                    "case_id": case_id,
                                    "prob": probability,
                                }
                            )
                    pd.DataFrame(rows).to_csv(
                        sealed / f"{name}_target_percase.csv",
                        index=False,
                    )

            output = root / "report"
            subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--runs_root",
                    str(runs),
                    "--sealed_dir",
                    str(sealed),
                    "--target_labels",
                    str(labels),
                    "--conditions",
                    "raw,cidp_U1",
                    "--seeds",
                    "7,16",
                    "--baseline",
                    "raw",
                    "--bootstrap_draws",
                    "50",
                    "--out_dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            contrasts = pd.read_csv(output / "downstream_contrasts.csv")
            self.assertGreater(float(contrasts.iloc[0]["delta_auc"]), 0.0)
            summary = json.loads((output / "downstream_summary.json").read_text())
            self.assertFalse(
                summary["protocol"]["target_labels_used_during_training"]
            )

            invalid = sealed / "cidp_U1_s16_target_percase.csv"
            frame = pd.read_csv(invalid)
            frame["label"] = np.tile(y, 2)
            frame.to_csv(invalid, index=False)
            failed = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--runs_root",
                    str(runs),
                    "--sealed_dir",
                    str(sealed),
                    "--target_labels",
                    str(labels),
                    "--conditions",
                    "raw,cidp_U1",
                    "--seeds",
                    "7,16",
                    "--out_dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("must not contain target labels", failed.stderr)


if __name__ == "__main__":
    unittest.main()
