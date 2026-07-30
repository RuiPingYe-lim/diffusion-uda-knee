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
        DiagnosticOrthogonalConditioner,
        DiagnosticSubspaceProjector,
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


if __name__ == "__main__":
    unittest.main()
