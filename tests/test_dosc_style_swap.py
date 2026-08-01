import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
UNSB_ROOT = REPO_ROOT / "src" / "unsb"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(UNSB_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSB_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from eval_dosc_style_swap import load_target_manifest
    from style_swap_metrics import (
        make_path_noises,
        summarize_style_swap,
        validate_complete_swap_grid,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class DoscStyleSwapTests(unittest.TestCase):
    def _frames(self, causal_slope: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        cases = (
            ("b0", 0, -2.0),
            ("b1", 0, -1.0),
            ("m0", 1, 1.0),
            ("m1", 1, 2.0),
        )
        references = (("r0", -1.0), ("r1", 0.0), ("r2", 1.0))
        swap_rows = []
        noise_rows = []
        for case_id, label, source_score in cases:
            for reference_id, probe_score in references:
                swap_rows.append(
                    {
                        "case_id": case_id,
                        "label": label,
                        "reference_id": reference_id,
                        "reference_probe_score": probe_score,
                        "source_score": source_score,
                        "output_score": source_score + causal_slope * probe_score,
                    }
                )
            for replicate, perturbation in enumerate((-0.01, 0.0, 0.01)):
                noise_rows.append(
                    {
                        "case_id": case_id,
                        "label": label,
                        "replicate_id": replicate,
                        "source_score": source_score,
                        "output_score": source_score + perturbation,
                    }
                )
        return pd.DataFrame(swap_rows), pd.DataFrame(noise_rows)

    def test_common_path_noise_has_one_broadcast_draw(self):
        source = torch.zeros(1, 3, 4, 4)
        common = make_path_noises(
            source,
            num_timesteps=5,
            batch_size=3,
            seed=7,
            common_across_batch=True,
        )
        independent = make_path_noises(
            source,
            num_timesteps=5,
            batch_size=3,
            seed=7,
            common_across_batch=False,
        )
        repeated = make_path_noises(
            source,
            num_timesteps=5,
            batch_size=3,
            seed=7,
            common_across_batch=False,
        )

        self.assertEqual(len(common), 4)
        self.assertEqual(tuple(common[0].shape), (1, 3, 4, 4))
        self.assertEqual(tuple(independent[0].shape), (3, 3, 4, 4))
        self.assertTrue(
            all(torch.equal(left, right) for left, right in zip(independent, repeated))
        )

    def test_causal_reference_effect_is_recovered(self):
        swap, noise = self._frames(causal_slope=0.5)
        summary = summarize_style_swap(
            swap,
            noise,
            bootstrap_draws=200,
            permutation_draws=200,
            seed=7,
        )

        self.assertAlmostEqual(summary["case_slope_mean"], 0.5, places=6)
        self.assertAlmostEqual(summary["reference_effect_pearson"], 1.0, places=6)
        self.assertGreater(
            summary["style_score_std_mean"],
            summary["path_noise_score_std_mean"],
        )
        self.assertAlmostEqual(summary["source_auc"], 1.0, places=6)
        self.assertAlmostEqual(summary["mean_over_references_auc"], 1.0, places=6)

    def test_no_reference_effect_has_zero_slope(self):
        swap, noise = self._frames(causal_slope=0.0)
        summary = summarize_style_swap(
            swap,
            noise,
            bootstrap_draws=50,
            permutation_draws=50,
            seed=7,
        )

        self.assertAlmostEqual(summary["case_slope_mean"], 0.0, places=6)
        self.assertAlmostEqual(summary["style_score_std_mean"], 0.0, places=6)
        self.assertIsNone(
            None
            if np.isnan(summary["reference_effect_pearson"])
            else summary["reference_effect_pearson"]
        )

    def test_incomplete_case_reference_grid_is_rejected(self):
        swap, _ = self._frames(causal_slope=0.5)
        incomplete = swap.iloc[:-1].copy()

        with self.assertRaisesRegex(ValueError, "complete case/reference grid"):
            validate_complete_swap_grid(incomplete)

    def test_target_manifest_diagnosis_column_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "target.png"
            Image.new("L", (8, 8), color=128).save(image_path)
            manifest = root / "target.csv"
            pd.DataFrame(
                [
                    {
                        "case_id": "target-1",
                        "image_path": image_path.name,
                        "label": 1,
                    }
                ]
            ).to_csv(manifest, index=False)

            frame = load_target_manifest(
                manifest,
                split="",
                id_col="case_id",
                path_col="image_path",
            )

            self.assertEqual(list(frame.columns), ["case_id", "path"])
            self.assertNotIn("label", frame.columns)


if __name__ == "__main__":
    unittest.main()
