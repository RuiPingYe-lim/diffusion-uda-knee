import ast
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES = REPO_ROOT / "src" / "unsb" / "dabrf_modules.py"
MODEL = REPO_ROOT / "src" / "unsb" / "trsc_dabrf_joint_sb_model.py"
RUNNER = REPO_ROOT / "scripts" / "run_unsb_trsc_joint_breast.sh"
WRAPPER = REPO_ROOT / "scripts" / "run_unsb_trsc_dabrf_breast.sh"
MATRIX = REPO_ROOT / "scripts" / "run_trsc_dabrf_matrix.sh"
JOINT_MATRIX = REPO_ROOT / "scripts" / "run_trsc_joint_matrix.sh"
INSTALLER = REPO_ROOT / "scripts" / "install_dosc_unsb_overlay.py"


class DABRFProtocolTests(unittest.TestCase):
    def test_python_entry_points_parse(self):
        ast.parse(MODULES.read_text(encoding="utf-8"))
        ast.parse(MODEL.read_text(encoding="utf-8"))

    def test_dabrf_repairs_only_the_existing_u1_candidate_set(self):
        model = MODEL.read_text(encoding="utf-8")
        self.assertIn("self.task_candidates", model)
        self.assertIn("self.task_repaired_candidates", model)
        self.assertIn('"raw_residual": raw_residual', MODULES.read_text())
        self.assertNotIn("U3", model)
        self.assertNotIn("U5", model)

    def test_constraint_gradient_cannot_move_the_generator(self):
        model = MODEL.read_text(encoding="utf-8")
        self.assertIn("self.task_candidates.detach().reshape(", model)
        self.assertIn(
            "route_task_gradient(\n            self.task_candidates,",
            model,
        )
        self.assertIn("if self.optimizer_R is not None:", model)
        self.assertIn("self.optimizer_R.step()", model)
        self.assertIn("self.optimizer_G.step()", model)

    def test_new_repair_weights_do_not_shift_the_random_stream(self):
        model = MODEL.read_text(encoding="utf-8")
        self.assertIn("cpu_rng_state = torch.random.get_rng_state()", model)
        self.assertIn("torch.random.set_rng_state(cpu_rng_state)", model)
        self.assertIn("torch.cuda.set_rng_state_all(cuda_rng_states)", model)

    def test_identity_is_a_true_base_model_bypass(self):
        model = MODEL.read_text(encoding="utf-8")
        matrix = MATRIX.read_text(encoding="utf-8")
        self.assertIn("if not self._dabrf_enabled:", model)
        self.assertIn("return super().optimize_parameters()", model)
        self.assertIn('MODEL_NAME="trsc_joint_sb"', matrix)
        self.assertIn(
            'bash "${SCRIPT_ROOT}/run_unsb_trsc_joint_breast.sh"',
            matrix,
        )
        identity_block = matrix.split(
            'if [[ "${repair_mode}" == "identity" ]]', 1
        )[1].split("return", 1)[0]
        self.assertNotIn("run_unsb_trsc_dabrf_breast.sh", identity_block)

    def test_identity_and_k3_emit_the_same_training_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unsb_root = root / "UNSB"
            unsb_root.mkdir()
            capture_script = (
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['CAPTURE_ARGS']).write_text("
                "json.dumps(sys.argv[1:]), encoding='utf-8')\n"
            )
            (unsb_root / "train.py").write_text(
                capture_script,
                encoding="utf-8",
            )

            common = os.environ.copy()
            common.update(
                {
                    "UNSB_ROOT": str(unsb_root),
                    "TRSC_DATA_ROOT": str(root / "data"),
                    "TRSC_INIT_DIR": str(root / "translator"),
                    "SOURCE_CLASSIFIER_CKPT": str(root / "source.pt"),
                    "GPU_IDS": "-1",
                    "SEED": "7",
                    "NUM_REFERENCES": "3",
                    "LAMBDA_TASK": "1.0",
                }
            )

            identity_capture = root / "identity.json"
            identity_env = common | {
                "ARMS": "identity",
                "SEEDS": "7",
                "EXPERIMENT_PREFIX": "identity_audit",
                "CAPTURE_ARGS": str(identity_capture),
            }
            subprocess.run(
                ["bash", str(MATRIX)],
                check=True,
                capture_output=True,
                text=True,
                env=identity_env,
            )

            baseline_capture = root / "baseline.json"
            baseline_env = common | {
                "ARMS": "k3_joint",
                "SEEDS": "7",
                "EXPERIMENT_PREFIX": "baseline_audit",
                "CAPTURE_ARGS": str(baseline_capture),
            }
            subprocess.run(
                ["bash", str(JOINT_MATRIX)],
                check=True,
                capture_output=True,
                text=True,
                env=baseline_env,
            )

            identity_args = json.loads(
                identity_capture.read_text(encoding="utf-8")
            )
            baseline_args = json.loads(
                baseline_capture.read_text(encoding="utf-8")
            )
            identity_args[identity_args.index("--name") + 1] = "<experiment>"
            baseline_args[baseline_args.index("--name") + 1] = "<experiment>"
            self.assertEqual(identity_args, baseline_args)
            self.assertEqual(
                identity_args[identity_args.index("--model") + 1],
                "trsc_joint_sb",
            )
            self.assertFalse(
                any(value.startswith("--dabrf_") for value in identity_args)
            )

    def test_zero_weight_constraints_skip_teacher_and_second_forward(self):
        model = MODEL.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("if self._dabrf_diag_enabled:", model)
        self.assertIn("if not self._dabrf_constraints_enabled:", model)
        self.assertIn("self.constraint_repaired_candidates = None", model)
        self.assertNotIn(': "${DABRF_TEACHER:?', runner)

    def test_only_source_labels_enter_diagnostic_repair(self):
        model = MODEL.read_text(encoding="utf-8")
        self.assertIn("self.real_A_label", model)
        self.assertNotIn("target_label", model.lower())
        self.assertNotIn("real_B_label", model)
        self.assertIn("references.detach()", model)

    def test_frozen_teacher_and_detached_calibration_are_required(self):
        model = MODEL.read_text(encoding="utf-8")
        self.assertIn("--dabrf_teacher_path", model)
        self.assertIn("torch.jit.load", model)
        self.assertIn("parameter.requires_grad_(False)", model)
        self.assertIn("CalibrationInvariantDiagnosticPreservation", model)
        self.assertIn("update_queue=False", model)

    def test_runner_selects_the_real_model_without_mode_prefix_collision(self):
        runner = RUNNER.read_text(encoding="utf-8")
        wrapper = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('--model "${MODEL_NAME}"', runner)
        self.assertIn('MODEL_NAME="trsc_dabrf_joint_sb"', wrapper)
        self.assertNotIn("  --mode ", runner)
        self.assertIn('--dabrf_teacher_path "${DABRF_TEACHER}"', runner)

    def test_matrix_contains_identity_simple_controls_and_full_repair(self):
        matrix = MATRIX.read_text(encoding="utf-8")
        self.assertIn('run_arm "identity"', matrix)
        self.assertIn('run_arm "fixed_scale08"', matrix)
        self.assertIn('run_arm "norm_clip08"', matrix)
        self.assertIn('run_arm "learned_task_only"', matrix)
        self.assertIn('run_arm "learned_full"', matrix)
        self.assertIn("NUM_REFERENCES=3", matrix)
        self.assertIn('ARMS="${ARMS:-identity fixed_scale08', matrix)
        self.assertIn('MODEL_NAME="trsc_joint_sb"', matrix)
        self.assertIn('MODEL_NAME="trsc_dabrf_joint_sb"', matrix)

    def test_overlay_installs_both_dabrf_files(self):
        installer = INSTALLER.read_text(encoding="utf-8")
        self.assertIn('OVERLAY_ROOT / "dabrf_modules.py"', installer)
        self.assertIn('OVERLAY_ROOT / "trsc_dabrf_joint_sb_model.py"', installer)


if __name__ == "__main__":
    unittest.main()
