import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.unsb.trsc_joint_modules import (
        multi_reference_cross_entropy,
        route_task_gradient,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrscJointModuleTests(unittest.TestCase):
    def test_gradient_router_changes_only_backward_scale(self):
        translated = torch.tensor([2.0], requires_grad=True)
        classifier_weight = torch.tensor([3.0], requires_grad=True)
        routed = route_task_gradient(translated, 0.25)
        self.assertTrue(torch.equal(routed, translated))
        (routed * classifier_weight).sum().backward()
        self.assertAlmostEqual(float(translated.grad), 0.75)
        self.assertAlmostEqual(float(classifier_weight.grad), 2.0)

    def test_equal_group_weight_is_invariant_to_duplicate_reference_count(self):
        raw = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        one_candidate = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        three_candidates = one_candidate.expand(-1, 3, -1).clone()
        labels = torch.tensor([0, 1])

        one = multi_reference_cross_entropy(
            raw,
            one_candidate,
            labels,
            weighting="equal_groups",
        )
        three = multi_reference_cross_entropy(
            raw,
            three_candidates,
            labels,
            weighting="equal_groups",
        )
        self.assertTrue(torch.allclose(one["candidate"], three["candidate"]))
        self.assertTrue(torch.allclose(one["total"], three["total"]))

    def test_every_candidate_receives_gradient(self):
        raw = torch.tensor([[0.5, -0.5]], requires_grad=True)
        candidates = torch.tensor(
            [[[0.2, -0.2], [0.1, -0.1], [-0.1, 0.1]]],
            requires_grad=True,
        )
        loss = multi_reference_cross_entropy(
            raw,
            candidates,
            torch.tensor([0]),
            weighting="equal_groups",
        )["total"]
        loss.backward()
        self.assertTrue(torch.all(candidates.grad.abs().sum(dim=2) > 0))

    def test_invalid_candidate_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[B,K,C\]"):
            multi_reference_cross_entropy(
                torch.zeros(2, 2),
                torch.zeros(2, 2),
                torch.zeros(2, dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()
