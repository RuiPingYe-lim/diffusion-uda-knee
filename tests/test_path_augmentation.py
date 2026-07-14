import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
except ModuleNotFoundError:  # Lightweight code-review environments may omit ML dependencies.
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = REPO_ROOT / "src" / "bbdm_strict"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

if torch is not None:
    from path_augmentation import build_target_style_bank, make_style_path_view  # noqa: E402


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class PathAugmentationTests(unittest.TestCase):
    def test_path_endpoints_are_exact_and_noise_free(self):
        source = torch.linspace(0.2, 0.8, steps=16 * 16).reshape(1, 16, 16)
        kwargs = dict(target_mean=0.45, target_std=0.08, noise_std=0.5, blur_kernel=7)

        at_source, endpoint = make_style_path_view(
            source, alpha=0.0, generator=torch.Generator().manual_seed(1), **kwargs
        )
        at_endpoint, endpoint_again = make_style_path_view(
            source, alpha=1.0, generator=torch.Generator().manual_seed(2), **kwargs
        )
        self.assertTrue(torch.equal(at_source, source))
        self.assertTrue(torch.equal(at_endpoint, endpoint))
        self.assertTrue(torch.equal(endpoint_again, endpoint))

    def test_brownian_view_is_reproducible(self):
        source = torch.linspace(0.1, 0.9, steps=16 * 16).reshape(1, 16, 16)
        kwargs = dict(
            target_mean=0.55,
            target_std=0.12,
            alpha=0.5,
            noise_std=0.1,
            blur_kernel=7,
        )
        first, _ = make_style_path_view(source, generator=torch.Generator().manual_seed(123), **kwargs)
        second, _ = make_style_path_view(source, generator=torch.Generator().manual_seed(123), **kwargs)
        third, _ = make_style_path_view(source, generator=torch.Generator().manual_seed(124), **kwargs)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))

    def test_style_bank_does_not_require_or_parse_target_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index in range(3):
                grid = np.linspace(20 + index * 10, 180 + index * 10, num=16 * 16).reshape(16, 16)
                path = root / f"target_{index}.png"
                Image.fromarray(grid.astype(np.uint8), mode="L").save(path)
                paths.append(str(path))
            csv_path = root / "target_train.csv"
            pd.DataFrame(
                {"image_path": paths, "label": ["must", "not", "be parsed"]}
            ).to_csv(csv_path, index=False)
            cache_path = root / "style_bank.json"

            bank = build_target_style_bank(
                csv_path,
                image_size=16,
                max_samples=3,
                seed=7,
                cache_path=cache_path,
            )
            cached = build_target_style_bank(
                csv_path,
                image_size=16,
                max_samples=3,
                seed=7,
                cache_path=cache_path,
            )
            self.assertEqual(bank["n_styles"], 3)
            self.assertFalse(bank["target_labels_used"])
            self.assertEqual(bank["means"], cached["means"])


if __name__ == "__main__":
    unittest.main()
