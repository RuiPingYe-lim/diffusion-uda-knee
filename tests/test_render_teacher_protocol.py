import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from train_render_robust_teacher import (
        assert_disjoint_cases,
        build_records,
    )
    from v9_calibration import assert_heldout_protocol, metrics


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class RenderTeacherProtocolTests(unittest.TestCase):
    def _manifest(self, root: Path, validation_case: str = "validation") -> Path:
        rows = []
        for split, case_id, label in (
            ("src_train", "training", 0),
            ("src_valid", validation_case, 1),
        ):
            row = {
                "split": split,
                "case_id": case_id,
                "label": label,
            }
            for view_index, view in enumerate(("src_path", "raw", "U1", "U5")):
                path = root / f"{split}_{view}.png"
                Image.new("L", (8, 8), color=20 + 20 * view_index).save(path)
                row[view] = path.name
            rows.append(row)
        manifest = root / "da_manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        return manifest

    def test_records_use_explicit_da_manifest_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root)
            records = build_records(
                manifest,
                "src_train",
                ["src_path", "raw", "U1", "U5"],
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["case_id"], "training")
            self.assertEqual(records[0]["label"], 0)
            self.assertEqual(
                set(records[0]["views"]),
                {"src_path", "raw", "U1", "U5"},
            )

    def test_train_validation_case_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root, validation_case="training")
            train = build_records(
                manifest,
                "src_train",
                ["src_path", "raw", "U1", "U5"],
            )
            validation = build_records(
                manifest,
                "src_valid",
                ["src_path", "raw", "U1", "U5"],
            )

            with self.assertRaisesRegex(ValueError, "case overlap"):
                assert_disjoint_cases(train, validation)

    def test_calibration_threshold_oracle_includes_outer_intervals(self):
        result = metrics(
            np.array([0.0, 1.0, 2.0]),
            np.array([1, 1, 0]),
            "test",
        )

        self.assertAlmostEqual(result["oracle_acc"], 2.0 / 3.0)

    def test_calibration_rejects_training_split_as_final_test(self):
        fit = pd.DataFrame({"case_id": ["fit"]})
        test = pd.DataFrame({"case_id": ["test"]})

        with self.assertRaisesRegex(ValueError, "held-out source test"):
            assert_heldout_protocol(
                fit,
                test,
                test_split="src_train",
                allow_nonheldout_test=False,
            )


if __name__ == "__main__":
    unittest.main()
