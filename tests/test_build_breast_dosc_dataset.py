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
BUILDER = REPO_ROOT / "scripts" / "build_breast_dosc_unsb_dataset.py"


class BuildBreastDoscDatasetTests(unittest.TestCase):
    def _make_csv(self, root: Path, name: str, cases, with_labels: bool) -> Path:
        rows = []
        for index, case_id in enumerate(cases):
            image_path = root / f"{name}_{case_id}.png"
            pixels = np.full((12, 12), 20 + index * 30, dtype=np.uint8)
            Image.fromarray(pixels, mode="L").save(image_path)
            row = {"image_path": str(image_path), "case_id": case_id}
            if with_labels:
                row["label"] = index % 2
            else:
                row["label"] = 99
            rows.append(row)
        csv_path = root / f"{name}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def test_builder_writes_only_source_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_train = self._make_csv(root, "source_train", ["s0", "s1"], True)
            source_val = self._make_csv(root, "source_val", ["v0", "v1"], True)
            target_train = self._make_csv(root, "target_train", ["t0", "t1"], False)
            output = root / "dataset"
            subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--source_train_csv",
                    str(source_train),
                    "--source_val_csv",
                    str(source_val),
                    "--target_train_csv",
                    str(target_train),
                    "--out_root",
                    str(output),
                    "--mode",
                    "copy",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            train_manifest = pd.read_csv(output / "trainA_manifest.csv")
            test_manifest = pd.read_csv(output / "testA_manifest.csv")
            summary = json.loads((output / "dataset_manifest.json").read_text())
            self.assertEqual(set(train_manifest["label"]), {0, 1})
            self.assertEqual(set(test_manifest["source_split"]), {"src_train", "src_valid"})
            self.assertFalse(summary["target_labels_used"])
            self.assertEqual(len(list((output / "trainB").iterdir())), 2)
            self.assertEqual(len(list((output / "testB").iterdir())), 2)
            self.assertFalse((output / "trainB_manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
