import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "scripts" / "install_dosc_unsb_overlay.py"
CANONICAL_INSTALLER = REPO_ROOT / "scripts" / "install_trsc_unsb_overlay.py"


class InstallDoscOverlayTests(unittest.TestCase):
    def test_installer_adds_files_without_modifying_upstream_core(self):
        with tempfile.TemporaryDirectory() as temporary:
            unsb_root = Path(temporary) / "UNSB"
            (unsb_root / "models").mkdir(parents=True)
            (unsb_root / "data").mkdir()
            core_text = (
                "class SBModel:\n"
                "    def compute_G_loss(self): pass\n"
                "    def calculate_NCE_loss(self): pass\n"
            )
            (unsb_root / "models" / "sb_model.py").write_text(core_text)
            (unsb_root / "models" / "__init__.py").write_text("")
            (unsb_root / "data" / "base_dataset.py").write_text("")
            (unsb_root / "data" / "__init__.py").write_text("")
            (unsb_root / "train.py").write_text("")
            (unsb_root / "test.py").write_text("")

            command = [
                sys.executable,
                str(CANONICAL_INSTALLER),
                "--unsb_root",
                str(unsb_root),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run(
                [
                    sys.executable,
                    str(INSTALLER),
                    "--unsb_root",
                    str(unsb_root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                (unsb_root / "models" / "sb_model.py").read_text(),
                core_text,
            )
            self.assertTrue((unsb_root / "models" / "dosc_modules.py").is_file())
            self.assertTrue((unsb_root / "models" / "dosc_sb_model.py").is_file())
            self.assertTrue(
                (unsb_root / "data" / "dosc_unaligned_dataset.py").is_file()
            )
            self.assertTrue((unsb_root / "models" / "trsc_sb_model.py").is_file())
            self.assertTrue(
                (unsb_root / "data" / "trsc_unaligned_dataset.py").is_file()
            )
            model_overlay = (unsb_root / "models" / "dosc_sb_model.py").read_text()
            self.assertIn("def translate_with_condition(", model_overlay)
            self.assertIn("def translate_with_reference(", model_overlay)
            self.assertIn('"--dosc_enable_projection"', model_overlay)
            self.assertIn('"--lambda_DOSC_diag"', model_overlay)
            self.assertIn("default=0.0", model_overlay)
            canonical_model = (
                unsb_root / "models" / "trsc_sb_model.py"
            ).read_text()
            self.assertIn("class TrscSBModel", canonical_model)
            self.assertIn('dataset_mode="trsc_unaligned"', canonical_model)


if __name__ == "__main__":
    unittest.main()
