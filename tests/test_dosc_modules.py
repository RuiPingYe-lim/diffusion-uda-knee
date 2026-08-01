import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UNSB_ROOT = REPO_ROOT / "src" / "unsb"
if str(UNSB_ROOT) not in sys.path:
    sys.path.insert(0, str(UNSB_ROOT))

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

if torch is not None:
    from dosc_modules import (
        CalibrationInvariantDiagnosticPreservation,
        DiagnosticOrthogonalConditioner,
        DiagnosticSubspaceProjector,
        TargetReferenceStyleConditioner,
        diagnostic_non_degradation_loss,
        gradient_reverse,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class DiagnosticOrthogonalModuleTests(unittest.TestCase):
    def test_gradient_reverse_changes_only_backward_direction(self):
        value = torch.tensor([[1.0, 2.0]], requires_grad=True)
        output = gradient_reverse(value, strength=0.25)
        self.assertTrue(torch.equal(value, output))
        output.sum().backward()
        self.assertTrue(torch.allclose(value.grad, torch.full_like(value, -0.25)))

    def test_binary_projector_removes_class_mean_direction(self):
        projector = DiagnosticSubspaceProjector(
            style_dim=4,
            num_classes=2,
            momentum=0.0,
        )
        style = torch.tensor(
            [
                [-2.0, 1.0, 0.5, -0.5],
                [-1.0, 1.0, 0.5, -0.5],
                [1.0, 1.0, 0.5, -0.5],
                [2.0, 1.0, 0.5, -0.5],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])
        projector.update(style, labels)
        projected = projector(style)
        class_gap = projected[labels == 1].mean(0) - projected[labels == 0].mean(0)
        self.assertEqual(int(projector.basis_rank.item()), 1)
        self.assertLess(float(class_gap.norm()), 1e-5)
        self.assertGreater(float(projector.removed_energy_ratio(style)), 0.0)

    def test_conditioner_outputs_generator_style_and_gradients(self):
        torch.manual_seed(7)
        conditioner = DiagnosticOrthogonalConditioner(
            input_channels=3,
            style_dim=8,
            generator_style_dim=16,
            num_classes=2,
            encoder_widths=(4, 8),
            queue_size=8,
            enable_projection=True,
        )
        source = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        labels = torch.tensor([0, 1])
        conditioner.build_context(source, target, labels)
        context = conditioner.build_context(source, target, labels)
        self.assertEqual(tuple(context["condition"].shape), (2, 16))
        self.assertEqual(int(conditioner.projector.basis_rank.item()), 1)
        loss = (
            context["condition"].square().mean()
            + context["diagnostic_loss"]
            + context["domain_loss"]
            + context["instance_loss"]
        )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in conditioner.encoder.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_target_reference_conditioner_disables_projection_by_default(self):
        conditioner = TargetReferenceStyleConditioner(
            input_channels=3,
            style_dim=8,
            generator_style_dim=16,
            num_classes=2,
            encoder_widths=(4, 8),
            queue_size=8,
        )
        source = torch.randn(2, 3, 16, 16)
        target = torch.randn(2, 3, 16, 16)
        labels = torch.tensor([0, 1])
        context = conditioner.build_context(source, target, labels)

        self.assertFalse(conditioner.enable_projection)
        self.assertEqual(int(conditioner.projector.basis_rank.item()), 0)
        self.assertEqual(float(context["removed_energy"]), 0.0)

    def test_margin_loss_penalizes_diagnostic_degradation(self):
        class MeanTeacher(nn.Module):
            def forward(self, image):
                score = image.mean(dim=(1, 2, 3))
                return torch.stack([-score, score], dim=1)

        source = torch.ones(2, 3, 8, 8)
        translated = torch.zeros(2, 3, 8, 8, requires_grad=True)
        labels = torch.ones(2, dtype=torch.long)
        loss, drop = diagnostic_non_degradation_loss(
            MeanTeacher(),
            source,
            translated,
            labels,
            tolerance=0.1,
        )
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(drop), 0.0)
        loss.backward()
        self.assertIsNotNone(translated.grad)
        self.assertTrue(torch.isfinite(translated.grad).all())

    def test_cidp_ignores_positive_affine_score_drift(self):
        cidp = CalibrationInvariantDiagnosticPreservation(
            queue_size=8,
            min_per_class=1,
            affine_ridge=0.0,
        )
        source_score = torch.tensor([-3.0, -2.0, 2.0, 3.0])
        translated_score = ((source_score - 4.0) / 2.0).requires_grad_()
        labels = torch.tensor([0, 0, 1, 1])

        result = cidp.forward_scores(
            source_score,
            translated_score,
            labels,
            margin_tolerance=1e-5,
            rank_tolerance=1e-5,
        )

        self.assertEqual(float(result["ready"]), 1.0)
        self.assertAlmostEqual(float(result["affine_scale"]), 2.0, places=5)
        self.assertAlmostEqual(float(result["affine_bias"]), 4.0, places=5)
        self.assertLess(float(result["loss"]), 1e-5)
        self.assertLess(float(result["class0_charge"]), 1e-5)
        self.assertLess(float(result["class1_charge"]), 1e-5)
        self.assertFalse(result["affine_scale"].requires_grad)
        self.assertFalse(result["affine_bias"].requires_grad)
        result["loss"].backward()
        self.assertIsNotNone(translated_score.grad)
        self.assertTrue(torch.isfinite(translated_score.grad).all())

    def test_cidp_rank_term_penalizes_local_ordering_damage(self):
        cidp = CalibrationInvariantDiagnosticPreservation(
            queue_size=8,
            min_per_class=1,
            affine_ridge=0.0,
        )
        source_score = torch.tensor([-3.0, -2.0, 2.0, 3.0])
        translated_score = torch.tensor(
            [-3.0, 1.0, -1.0, 3.0],
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1])

        result = cidp.forward_scores(
            source_score,
            translated_score,
            labels,
            margin_tolerance=100.0,
            rank_tolerance=0.1,
        )

        self.assertEqual(float(result["ready"]), 1.0)
        self.assertEqual(float(result["margin_loss"]), 0.0)
        self.assertGreater(float(result["rank_loss"]), 0.0)
        result["loss"].backward()
        self.assertIsNotNone(translated_score.grad)
        self.assertGreater(float(translated_score.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(translated_score.grad).all())

    def test_cidp_waits_for_both_class_queues(self):
        cidp = CalibrationInvariantDiagnosticPreservation(
            queue_size=8,
            min_per_class=2,
            affine_ridge=0.0,
        )
        labels = torch.tensor([0, 1])
        first = cidp.forward_scores(
            torch.tensor([-2.0, 2.0]),
            torch.tensor([-4.0, 0.0], requires_grad=True),
            labels,
        )
        second = cidp.forward_scores(
            torch.tensor([-3.0, 3.0]),
            torch.tensor([-5.0, 1.0], requires_grad=True),
            labels,
        )

        self.assertEqual(float(first["ready"]), 0.0)
        self.assertEqual(float(first["loss"]), 0.0)
        self.assertEqual(float(second["ready"]), 1.0)
        self.assertTrue(torch.equal(cidp.score_queue.counts, torch.tensor([2, 2])))

    def test_cidp_queue_round_trips_through_state_dict(self):
        original = CalibrationInvariantDiagnosticPreservation(
            queue_size=8,
            min_per_class=1,
        )
        original.forward_scores(
            torch.tensor([-2.0, 2.0]),
            torch.tensor([-3.0, 1.0], requires_grad=True),
            torch.tensor([0, 1]),
        )
        restored = CalibrationInvariantDiagnosticPreservation(
            queue_size=8,
            min_per_class=1,
        )
        restored.load_state_dict(original.state_dict())

        self.assertTrue(
            torch.equal(
                original.score_queue.counts,
                restored.score_queue.counts,
            )
        )
        self.assertTrue(
            torch.allclose(
                original.score_queue.translated_scores,
                restored.score_queue.translated_scores,
            )
        )


if __name__ == "__main__":
    unittest.main()
