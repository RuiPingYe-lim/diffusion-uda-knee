import ast
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES = REPO_ROOT / "src" / "unsb" / "safebridge_modules.py"
MODEL = REPO_ROOT / "src" / "unsb" / "safebridge_joint_sb_model.py"
CLASSIFIER = REPO_ROOT / "src" / "unsb" / "trsc_joint_modules.py"
RUNNER = REPO_ROOT / "scripts" / "run_unsb_safebridge_breast.sh"
MATRIX = REPO_ROOT / "scripts" / "run_safebridge_matrix.sh"
INSTALLER = REPO_ROOT / "scripts" / "install_dosc_unsb_overlay.py"
SUMMARIZER = REPO_ROOT / "scripts" / "summarize_safebridge_audit.py"
SOURCE_SELECTOR = (
    REPO_ROOT / "scripts" / "select_safebridge_source_checkpoint.py"
)


class SafeBridgeProtocolTests(unittest.TestCase):
    def test_python_entry_points_parse(self):
        ast.parse(MODULES.read_text(encoding="utf-8"))
        ast.parse(MODEL.read_text(encoding="utf-8"))
        ast.parse(CLASSIFIER.read_text(encoding="utf-8"))
        ast.parse(SOURCE_SELECTOR.read_text(encoding="utf-8"))

    def test_classifier_task_gradient_cannot_enter_unsb(self):
        model = MODEL.read_text(encoding="utf-8")
        self.assertIn("@torch.no_grad()\n    def generate_state_candidates", model)
        self.assertIn("candidates.detach().reshape", model)
        self.assertIn("set --lambda_TRSC_task 0.0", model)
        self.assertNotIn("route_task_gradient", model)
        self.assertIn("DoscSBModel.compute_G_loss(self)", model)

    def test_dssr_considers_u1_u3_u5_and_can_reject(self):
        model = MODEL.read_text(encoding="utf-8")
        modules = MODULES.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('default="1,3,5"', model)
        self.assertIn('CANDIDATE_STATES="${CANDIDATE_STATES:-1,3,5}"', runner)
        self.assertIn("select_minimal_adequate_state", model)
        self.assertIn("selected_state, torch.full_like(selected_state, -1)", modules)
        self.assertIn("StateWiseDiagnosticCalibrator", model)

    def test_only_source_diagnosis_labels_are_used(self):
        model = MODEL.read_text(encoding="utf-8")
        modules = MODULES.read_text(encoding="utf-8")
        self.assertIn("self.real_A_label", model)
        self.assertNotIn("real_B_label", model)
        self.assertNotIn("target_label", model.lower())
        self.assertNotIn("target_label", modules.lower())

    def test_target_support_and_diagnostic_nullspace_are_both_explicit(self):
        model = MODEL.read_text(encoding="utf-8")
        modules = MODULES.read_text(encoding="utf-8")
        self.assertIn("target_subspace.project(stable[\"residual\"])", model)
        self.assertIn("project_diagnostic_nullspace", model)
        self.assertIn("torch.autograd.grad", model)
        self.assertIn("post_inner", modules)
        self.assertIn("source_pool, target_pool", model)

    def test_custom_methods_unwrap_unsb_data_parallel(self):
        model = MODEL.read_text(encoding="utf-8")
        self.assertIn('return network.module if hasattr(network, "module")', model)
        self.assertIn("self._classifier_module().forward_to_layer3", model)
        self.assertIn("target_subspace.is_ready()", model)
        self.assertIn("diagnostic_calibrator.evaluate_and_update", model)

    def test_layer3_split_preserves_checkpoint_container(self):
        classifier = CLASSIFIER.read_text(encoding="utf-8")
        self.assertIn("self.stem = nn.Sequential", classifier)
        self.assertIn("def forward_to_layer3", classifier)
        self.assertIn("self.stem[:7]", classifier)
        self.assertIn("self.stem[7]", classifier)

    def test_runner_forces_the_protocol_barrier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unsb_root = root / "UNSB"
            unsb_root.mkdir()
            capture = root / "args.json"
            (unsb_root / "train.py").write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['CAPTURE_ARGS']).write_text("
                "json.dumps(sys.argv[1:]), encoding='utf-8')\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "UNSB_ROOT": str(unsb_root),
                    "TRSC_DATA_ROOT": str(root / "data"),
                    "TRSC_INIT_DIR": str(root / "translator"),
                    "SOURCE_CLASSIFIER_CKPT": str(root / "source.pt"),
                    "SAFEBRIDGE_TEACHER": str(root / "teacher.pt"),
                    "CAPTURE_ARGS": str(capture),
                    "GPU_IDS": "-1",
                }
            )
            subprocess.run(
                ["bash", str(RUNNER)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            arguments = json.loads(capture.read_text(encoding="utf-8"))
            self.assertEqual(
                arguments[arguments.index("--model") + 1],
                "safebridge_joint_sb",
            )
            self.assertEqual(
                arguments[arguments.index("--lambda_TRSC_task") + 1],
                "0.0",
            )
            self.assertEqual(
                arguments[arguments.index("--safebridge_candidate_states") + 1],
                "1,3,5",
            )

    def test_matrix_contains_every_incremental_ablation(self):
        matrix = MATRIX.read_text(encoding="utf-8")
        for arm in (
            "raw_only",
            "dssr_only",
            "stable_residual",
            "target_projected",
            "full",
        ):
            self.assertIn(arm, matrix)

    def test_checkpoint_selection_uses_source_validation_only(self):
        selector = SOURCE_SELECTOR.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("source_val_csv", selector)
        self.assertIn('"selection_domain": "source_validation"', selector)
        self.assertIn('"target_labels_used": False', selector)
        self.assertNotIn("target_val", selector.lower())
        self.assertIn("select_safebridge_source_checkpoint.py", runner)
        self.assertIn("best_source_val_net_C.pth", runner)
        self.assertIn('--checkpoint "${CLASSIFIER_FOR_INFERENCE}"', runner)

    def test_overlay_installs_both_safebridge_files(self):
        installer = INSTALLER.read_text(encoding="utf-8")
        self.assertIn('OVERLAY_ROOT / "safebridge_modules.py"', installer)
        self.assertIn('OVERLAY_ROOT / "safebridge_joint_sb_model.py"', installer)

    def test_audit_summary_reports_class_conditioned_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = root / "audit.jsonl"
            rows = [
                {
                    "source_label": 0,
                    "selected_state": 1,
                    "dssr_accepted": True,
                    "final_accepted": True,
                    "alpha": 0.5,
                    "diagnostic_margin_drop": 0.02,
                    "diagnostic_rank_violation": 0.0,
                    "trajectory_stability": 0.9,
                    "domain_gain_retention": 0.7,
                    "selector_ready": True,
                },
                {
                    "source_label": 1,
                    "selected_state": -1,
                    "dssr_accepted": False,
                    "final_accepted": False,
                    "alpha": 0.0,
                    "diagnostic_margin_drop": 0.3,
                    "diagnostic_rank_violation": 0.2,
                    "trajectory_stability": 0.4,
                    "domain_gain_retention": 0.0,
                    "selector_ready": True,
                },
            ]
            audit.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "summary.json"
            subprocess.run(
                [
                    "python",
                    str(SUMMARIZER),
                    "--input",
                    str(audit),
                    "--out",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["final_acceptance"], 0.5)
            self.assertEqual(
                summary["final_acceptance_by_source_class"],
                {"0": 1.0, "1": 0.0},
            )


if __name__ == "__main__":
    unittest.main()
