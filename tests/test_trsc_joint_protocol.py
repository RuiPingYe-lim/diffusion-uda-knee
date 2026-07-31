import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = REPO_ROOT / "src" / "unsb" / "trsc_joint_sb_model.py"
BASE_MODEL = REPO_ROOT / "src" / "unsb" / "dosc_sb_model.py"
DATASET = REPO_ROOT / "src" / "unsb" / "trsc_unaligned_dataset.py"
RUNNER = REPO_ROOT / "scripts" / "run_unsb_trsc_joint_breast.sh"
MATRIX = REPO_ROOT / "scripts" / "run_trsc_joint_matrix.sh"
EVALUATOR = REPO_ROOT / "scripts" / "eval_trsc_joint_classifier.py"


class TrscJointProtocolTests(unittest.TestCase):
    def test_python_entry_points_parse(self):
        ast.parse(MODEL.read_text(encoding="utf-8"))
        ast.parse(DATASET.read_text(encoding="utf-8"))
        ast.parse(EVALUATOR.read_text(encoding="utf-8"))

    def test_default_path_is_core_u1_with_three_references(self):
        model = MODEL.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"--trsc_num_references", type=int, default=3', model)
        self.assertIn('NUM_REFERENCES="${NUM_REFERENCES:-3}"', runner)
        self.assertIn("--lambda_DOSC_safe 0.0", runner)
        self.assertIn("--lambda_DOSC_diag 0.0", runner)
        self.assertIn("--dosc_noise_ratio 0.0", runner)
        base_model = BASE_MODEL.read_text(encoding="utf-8")
        self.assertIn('"--seed",', base_model)
        self.assertIn('"--lambda_DOSC_safe",', base_model)
        self.assertIn(
            'default=0.0,\n            help="CIDP ablation weight',
            base_model,
        )
        self.assertIn(
            'LAMBDA_SAFE="${LAMBDA_SAFE:-0.0}"',
            (REPO_ROOT / "scripts" / "run_unsb_dosc_breast.sh").read_text(
                encoding="utf-8"
            ),
        )
        self.assertNotIn("topk", model.lower())
        self.assertNotIn("threshold", model.lower())

    def test_both_warm_starts_are_required_and_strict(self):
        model = MODEL.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('missing.append("--trsc_source_classifier_path")', model)
        self.assertIn('missing.append("--trsc_translator_init_dir")', model)
        self.assertIn("warm-start mismatch", model)
        self.assertIn(
            ': "${TRSC_INIT_DIR:?',
            runner,
        )
        self.assertIn(
            ': "${SOURCE_CLASSIFIER_CKPT:?',
            runner,
        )

    def test_task_gradient_is_routed_to_translator_without_scaling_classifier(self):
        model = MODEL.read_text(encoding="utf-8")
        self.assertIn(
            "route_task_gradient(\n            self.task_candidates,",
            model,
        )
        self.assertIn("self.optimizer_G.step()", model)
        self.assertIn("self.optimizer_S.step()", model)
        self.assertIn("self.optimizer_C.step()", model)

    def test_target_inference_reads_no_label_column(self):
        evaluator = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn("usecols=[args.path_col, args.case_col]", evaluator)
        self.assertIn('"target_labels_used": False', evaluator)
        self.assertNotIn("label_col", evaluator)
        self.assertIn('image.convert("L")', evaluator)
        self.assertLess(evaluator.index("T.ToTensor()"), evaluator.index("T.Resize("))

    def test_first_reference_is_shared_with_unsb_real_target(self):
        model = MODEL.read_text(encoding="utf-8")
        dataset = DATASET.read_text(encoding="utf-8")
        self.assertIn("torch.equal(references[:, 0], self.real_B)", model)
        self.assertIn('references = [item["B"]]', dataset)
        self.assertIn("random.sample(candidates, needed)", dataset)

    def test_minimal_matrix_is_k1_k3_and_task_gradient_only(self):
        matrix = MATRIX.read_text(encoding="utf-8")
        self.assertIn('run_arm "k1_joint" "${seed}" 1 1.0', matrix)
        self.assertIn(
            'run_arm "k3_no_task_gradient" "${seed}" 3 0.0',
            matrix,
        )
        self.assertIn('run_arm "k3_joint" "${seed}" 3 1.0', matrix)
        self.assertNotIn("CIDP", matrix)
        self.assertNotIn("topk", matrix.lower())


if __name__ == "__main__":
    unittest.main()
