import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.unsb.safebridge_modules import (
        StateWiseDiagnosticCalibrator,
        TargetGradientSubspace,
        choose_backtracking_alpha,
        masked_multi_reference_cross_entropy,
        parse_candidate_states,
        project_diagnostic_nullspace,
        select_minimal_adequate_state,
    )
    from src.unsb.trsc_joint_modules import SourceWarmStartResNet50


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SafeBridgeModuleTests(unittest.TestCase):
    def test_candidate_states_are_one_based_unique_and_increasing(self):
        self.assertEqual(parse_candidate_states("1,3,5", 5), (1, 3, 5))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_candidate_states("3,1", 5)
        with self.assertRaisesRegex(ValueError, r"\[1,5\]"):
            parse_candidate_states("1,6", 5)

    def test_target_subspace_projects_only_onto_observed_directions(self):
        module = TargetGradientSubspace(
            feature_dim=4,
            rank=2,
            queue_size=4,
            min_samples=2,
            minimum_domain_accuracy=0.5,
        )
        gradients = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        module.update(gradients, torch.tensor([1.0, 2.0]), 1.0)
        self.assertTrue(module.is_ready())
        residual = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])
        projected = module.project(residual)
        self.assertTrue(torch.allclose(projected[:, :2], residual[:, :2], atol=1e-5))
        self.assertTrue(torch.allclose(projected[:, 2:], torch.zeros_like(projected[:, 2:]), atol=1e-5))

    def test_state_calibration_absorbs_positive_scale_and_offset(self):
        module = StateWiseDiagnosticCalibrator(
            state_count=1,
            queue_size=4,
            min_per_class=1,
            affine_ridge=0.0,
        )
        source = torch.tensor([-2.0, 2.0])
        candidate = torch.tensor([[[-1.0]], [[1.0]]])
        result = module.evaluate_and_update(
            source,
            candidate,
            torch.tensor([0, 1]),
        )
        self.assertTrue(torch.all(result["ready"]))
        self.assertTrue(torch.all(result["safe"]))
        self.assertTrue(
            torch.allclose(
                result["calibrated_score"].reshape(-1),
                source,
                atol=1e-5,
            )
        )

    def test_selector_uses_the_shallowest_adequate_state(self):
        safe = torch.ones(1, 1, 3, dtype=torch.bool)
        result = select_minimal_adequate_state(
            safe,
            torch.tensor([[[0.4, 0.8, 0.9]]]),
            torch.tensor([0.0]),
            target_threshold=0.7,
            stability=torch.ones(1, 1, 3),
            state_ids=(1, 3, 5),
        )
        self.assertTrue(bool(result["accepted"].item()))
        self.assertEqual(int(result["selected_state"].item()), 3)

    def test_nullspace_projection_removes_diagnostic_component(self):
        residual = torch.tensor([[[[[1.0]], [[2.0]]]]])
        gradient = torch.tensor([[[[1.0]], [[0.0]]]])
        result = project_diagnostic_nullspace(residual, gradient)
        expected = torch.tensor([[[[[0.0]], [[2.0]]]]])
        self.assertTrue(torch.allclose(result["residual"], expected))
        self.assertLess(float(result["post_inner"].abs().max()), 1e-6)

    def test_backtracking_chooses_first_nonlinearly_valid_strength(self):
        result = choose_backtracking_alpha(
            domain_scores=torch.tensor([[[0.30, 0.20]]]),
            true_class_margins=torch.tensor([[[-0.50, 0.95]]]),
            source_domain_score=torch.tensor([0.0]),
            source_true_class_margin=torch.tensor([1.0]),
            accepted=torch.tensor([[True]]),
            alphas=(1.0, 0.5),
            minimum_domain_gain=0.15,
            margin_tolerance=0.10,
            required_domain_gain=torch.tensor([[0.18]]),
        )
        self.assertTrue(bool(result["accepted"].item()))
        self.assertAlmostEqual(float(result["alpha"].item()), 0.5)

    def test_rejected_cases_receive_unscaled_raw_loss(self):
        raw = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        candidates = torch.tensor(
            [[[1.0, 0.0]], [[2.0, -2.0]]],
        )
        labels = torch.tensor([0, 1])
        result = masked_multi_reference_cross_entropy(
            raw,
            candidates,
            labels,
            accepted=torch.tensor([[True], [False]]),
        )
        raw_per_case = torch.nn.functional.cross_entropy(
            raw, labels, reduction="none"
        )
        candidate_first = torch.nn.functional.cross_entropy(
            candidates[0], labels[:1]
        )
        expected = torch.stack(
            [0.5 * (raw_per_case[0] + candidate_first), raw_per_case[1]]
        ).mean()
        self.assertTrue(torch.allclose(result["total"], expected))

    def test_layer3_split_reproduces_the_original_classifier_forward(self):
        model = SourceWarmStartResNet50().eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            direct = model(image)
            split = model.forward_from_layer3(model.forward_to_layer3(image))
        self.assertTrue(torch.equal(direct, split))


if __name__ == "__main__":
    unittest.main()
