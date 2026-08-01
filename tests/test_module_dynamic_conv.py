import unittest

import torch
from torch import nn

from anytrain.module.moe.conv import (
    ADTRouter1d,
    ADTRouter2d,
    DynamicConv1d,
    DynamicConv2d,
    DynamicConvTranspose1d,
)
from anytrain.module.moe.conv.shape import (
    effective_kernel_size_1d,
    effective_kernel_size_2d,
    infer_padding_1d,
    infer_padding_2d,
    validate_dynamic_conv1d_args,
)
from anytrain.module.routing import DenseExpertWeights


class SimpleRouter(nn.Module):
    def __init__(self, in_channels: int, num_experts: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=-1)).softmax(dim=-1)


class SimpleRouter2d(nn.Module):
    def __init__(self, in_channels: int, num_experts: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=(-2, -1))).softmax(dim=-1)


class DynamicConv1dTest(unittest.TestCase):
    def test_initialization_shapes(self):
        conv = DynamicConv1d(
            16,
            32,
            kernel_size=3,
            num_experts=4,
            router=SimpleRouter(16, 4),
        )

        self.assertEqual(conv.kernel_size, torch.Size([3]))
        self.assertEqual(conv.weight.shape, (4, 32, 16, 3))
        self.assertEqual(conv.bias.shape, (4, 32))

    def test_forward_preserves_length_with_stride_one(self):
        conv = DynamicConv1d(
            16,
            32,
            kernel_size=3,
            num_experts=4,
            router=SimpleRouter(16, 4),
        )
        x = torch.randn(2, 16, 100)

        y = conv(x)

        self.assertEqual(y.shape, (2, 32, 100))
        self.assertTrue(torch.isfinite(y).all())

    def test_gradient_flow_to_router_and_experts(self):
        router = SimpleRouter(16, 4)
        conv = DynamicConv1d(16, 32, kernel_size=3, num_experts=4, router=router)
        x = torch.randn(2, 16, 24)

        conv(x).sum().backward()

        self.assertIsNotNone(conv.weight.grad)
        self.assertTrue(torch.isfinite(conv.weight.grad).all())
        self.assertIsNotNone(conv.bias.grad)
        self.assertTrue(torch.isfinite(conv.bias.grad).all())
        for parameter in router.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_segmented_forward_preserves_length(self):
        conv = DynamicConv1d(
            16,
            32,
            kernel_size=3,
            num_experts=4,
            segment_size=16,
            router=SimpleRouter(16, 4),
        )
        x = torch.randn(2, 16, 63)

        y = conv(x)

        self.assertEqual(y.shape, (2, 32, 63))
        self.assertTrue(torch.isfinite(y).all())

    def test_segmented_boundary_unaware_uses_local_zero_context(self):
        conv = DynamicConv1d(
            1,
            1,
            kernel_size=3,
            num_experts=1,
            segment_size=4,
            boundary_aware=False,
            router=None,
            bias=False,
        )
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[0, 0, 0, 1] = 1
        x = torch.arange(1, 9, dtype=torch.float32).view(1, 1, -1)

        y = conv.forward_manually(x, torch.ones(1, 1))

        self.assertTrue(torch.equal(y, x))

    def test_segmented_boundary_aware_matches_full_conv_padding_modes(self):
        x = torch.arange(1, 6, dtype=torch.float32).view(1, 1, -1)
        expert_weights = torch.ones(1, 1)

        for padding_mode in ("zeros", "replicate", "reflect", "circular"):
            with self.subTest(padding_mode=padding_mode):
                full_conv = DynamicConv1d(
                    1,
                    1,
                    kernel_size=3,
                    num_experts=1,
                    padding_mode=padding_mode,
                    router=None,
                    bias=False,
                )
                segmented_conv = DynamicConv1d(
                    1,
                    1,
                    kernel_size=3,
                    num_experts=1,
                    padding_mode=padding_mode,
                    segment_size=4,
                    router=None,
                    bias=False,
                )
                with torch.no_grad():
                    full_conv.weight.fill_(1.0)
                    segmented_conv.weight.copy_(full_conv.weight)

                full_y = full_conv.forward_manually(x, expert_weights)
                segmented_y = segmented_conv.forward_manually(x, expert_weights)

                self.assertTrue(torch.allclose(segmented_y, full_y))

    def test_even_stride_segmented_forward(self):
        conv = DynamicConv1d(
            16,
            32,
            kernel_size=4,
            num_experts=4,
            stride=2,
            segment_size=16,
            router=SimpleRouter(16, 4),
        )
        x = torch.randn(2, 16, 64)

        y = conv(x)

        self.assertEqual(y.shape, (2, 32, 32))
        self.assertTrue(torch.isfinite(y).all())

    def test_forward_manually_accepts_broadcast_weights(self):
        conv = DynamicConv1d(16, 32, kernel_size=3, num_experts=4, router=None)
        x = torch.randn(2, 16, 24)
        expert_weights = torch.tensor([[0.7, 0.1, 0.1, 0.1]])

        y = conv.forward_manually(x, expert_weights)

        self.assertEqual(y.shape, (2, 32, 24))
        self.assertTrue(torch.isfinite(y).all())

    def test_forward_manually_accepts_dense_expert_weights_object(self):
        conv = DynamicConv1d(16, 32, kernel_size=3, num_experts=4, router=None)
        x = torch.randn(2, 16, 24)
        expert_weights = DenseExpertWeights(torch.tensor([[0.7, 0.1, 0.1, 0.1]]))

        y = conv.forward_manually(x, expert_weights)

        self.assertEqual(y.shape, (2, 32, 24))
        self.assertTrue(torch.isfinite(y).all())

    def test_forward_manually_repeats_batch_weights_across_segments(self):
        conv = DynamicConv1d(
            1,
            1,
            kernel_size=3,
            num_experts=2,
            segment_size=4,
            router=None,
            bias=False,
        )
        x = torch.randn(2, 1, 8)

        y = conv.forward_manually(x, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        self.assertEqual(y.shape, (2, 1, 8))
        self.assertTrue(torch.isfinite(y).all())

    def test_forward_requires_router(self):
        conv = DynamicConv1d(16, 32, kernel_size=3, num_experts=4, router=None)

        with self.assertRaisesRegex(ValueError, "router"):
            conv(torch.randn(2, 16, 24))

    def test_forward_manually_rejects_mismatched_batch(self):
        conv = DynamicConv1d(16, 32, kernel_size=3, num_experts=4, router=None)

        with self.assertRaisesRegex(ValueError, "batch size"):
            conv.forward_manually(torch.randn(2, 16, 24), torch.randn(3, 4))

    def test_groups_change_weight_shape(self):
        conv = DynamicConv1d(
            16,
            32,
            kernel_size=3,
            num_experts=4,
            groups=2,
            router=SimpleRouter(16, 4),
        )

        self.assertEqual(conv.weight.shape, (4, 32, 8, 3))
        self.assertEqual(conv(torch.randn(2, 16, 24)).shape, (2, 32, 24))


class DynamicConvTranspose1dTest(unittest.TestCase):
    def test_initialization_shapes(self):
        conv = DynamicConvTranspose1d(
            16,
            32,
            kernel_size=3,
            num_experts=4,
            router=SimpleRouter(16, 4),
        )

        self.assertEqual(conv.kernel_size, torch.Size([3]))
        self.assertEqual(conv.weight.shape, (4, 16, 32, 3))
        self.assertEqual(conv.bias.shape, (4, 32))

    def test_forward_preserves_length_with_stride_one(self):
        conv = DynamicConvTranspose1d(
            16,
            32,
            kernel_size=3,
            num_experts=4,
            router=SimpleRouter(16, 4),
        )
        x = torch.randn(2, 16, 100)

        y = conv(x)

        self.assertEqual(y.shape, (2, 32, 100))
        self.assertTrue(torch.isfinite(y).all())

    def test_segmented_forward_preserves_length(self):
        conv = DynamicConvTranspose1d(
            16,
            32,
            kernel_size=3,
            num_experts=4,
            segment_size=16,
            router=SimpleRouter(16, 4),
        )
        x = torch.randn(2, 16, 63)

        y = conv(x)

        self.assertEqual(y.shape, (2, 32, 63))
        self.assertTrue(torch.isfinite(y).all())

    def test_segmented_forward_matches_full_conv_with_bias(self):
        torch.manual_seed(0)
        expert_weights = torch.ones(1, 1)
        for stride, kernel_size, segment_size in ((1, 3, 8), (2, 4, 8)):
            with self.subTest(stride=stride):
                full_conv = DynamicConvTranspose1d(
                    2,
                    3,
                    kernel_size=kernel_size,
                    num_experts=1,
                    stride=stride,
                    router=None,
                )
                segmented_conv = DynamicConvTranspose1d(
                    2,
                    3,
                    kernel_size=kernel_size,
                    num_experts=1,
                    stride=stride,
                    segment_size=segment_size,
                    router=None,
                )
                with torch.no_grad():
                    full_conv.weight.normal_()
                    full_conv.bias.normal_()
                    segmented_conv.weight.copy_(full_conv.weight)
                    segmented_conv.bias.copy_(full_conv.bias)
                x = torch.randn(4, 2, 23)

                full_y = full_conv.forward_manually(x, expert_weights)
                segmented_y = segmented_conv.forward_manually(x, expert_weights)

                self.assertTrue(torch.allclose(segmented_y, full_y, atol=1e-6))

    def test_stride_two_upsamples(self):
        conv = DynamicConvTranspose1d(
            16,
            32,
            kernel_size=4,
            num_experts=4,
            stride=2,
            router=SimpleRouter(16, 4),
        )
        x = torch.randn(2, 16, 50)

        y = conv(x)

        self.assertEqual(y.shape, (2, 32, 100))
        self.assertTrue(torch.isfinite(y).all())

    def test_output_padding(self):
        conv = DynamicConvTranspose1d(
            16,
            32,
            kernel_size=4,
            num_experts=4,
            stride=2,
            output_padding=1,
            router=SimpleRouter(16, 4),
        )

        y = conv(torch.randn(2, 16, 50))

        self.assertEqual(y.shape, (2, 32, 101))
        self.assertTrue(torch.isfinite(y).all())

    def test_forward_manually_validates_shape(self):
        conv = DynamicConvTranspose1d(16, 32, kernel_size=3, num_experts=4, router=None)

        with self.assertRaisesRegex(ValueError, "2D"):
            conv.forward_manually(torch.randn(2, 16, 24), torch.randn(2, 4, 1))


class DynamicConv2dTest(unittest.TestCase):
    def test_initialization_shapes(self):
        conv = DynamicConv2d(
            16,
            32,
            kernel_size=3,
            num_experts=4,
            router=SimpleRouter2d(16, 4),
        )

        self.assertEqual(conv.kernel_size, torch.Size([3, 3]))
        self.assertEqual(conv.weight.shape, (4, 32, 16, 3, 3))
        self.assertEqual(conv.bias.shape, (4, 32))

    def test_forward_preserves_size_with_stride_one(self):
        conv = DynamicConv2d(
            16,
            32,
            kernel_size=3,
            num_experts=4,
            router=SimpleRouter2d(16, 4),
        )
        x = torch.randn(2, 16, 13, 17)

        y = conv(x)

        self.assertEqual(y.shape, (2, 32, 13, 17))
        self.assertTrue(torch.isfinite(y).all())

    def test_gradient_flow_to_router_and_experts(self):
        router = SimpleRouter2d(16, 4)
        conv = DynamicConv2d(16, 32, kernel_size=3, num_experts=4, router=router)
        x = torch.randn(2, 16, 13, 17)

        conv(x).sum().backward()

        self.assertIsNotNone(conv.weight.grad)
        self.assertTrue(torch.isfinite(conv.weight.grad).all())
        self.assertIsNotNone(conv.bias.grad)
        self.assertTrue(torch.isfinite(conv.bias.grad).all())
        for parameter in router.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_forward_manually_accepts_broadcast_weights(self):
        conv = DynamicConv2d(16, 32, kernel_size=3, num_experts=4, router=None)
        x = torch.randn(2, 16, 13, 17)
        expert_weights = torch.tensor([[0.7, 0.1, 0.1, 0.1]])

        y = conv.forward_manually(x, expert_weights)

        self.assertEqual(y.shape, (2, 32, 13, 17))
        self.assertTrue(torch.isfinite(y).all())

    def test_forward_requires_router(self):
        conv = DynamicConv2d(16, 32, kernel_size=3, num_experts=4, router=None)

        with self.assertRaisesRegex(ValueError, "router"):
            conv(torch.randn(2, 16, 13, 17))

    def test_rejects_nonzero_padding_mode(self):
        with self.assertRaisesRegex(ValueError, "padding_mode"):
            DynamicConv2d(
                16,
                32,
                kernel_size=3,
                num_experts=4,
                padding_mode="reflect",
                router=None,
            )

    def test_forward_manually_rejects_mismatched_batch(self):
        conv = DynamicConv2d(16, 32, kernel_size=3, num_experts=4, router=None)

        with self.assertRaisesRegex(ValueError, "batch size"):
            conv.forward_manually(torch.randn(2, 16, 13, 17), torch.randn(3, 4))

    def test_groups_change_weight_shape(self):
        conv = DynamicConv2d(
            16,
            32,
            kernel_size=3,
            num_experts=4,
            groups=2,
            router=SimpleRouter2d(16, 4),
        )

        self.assertEqual(conv.weight.shape, (4, 32, 8, 3, 3))
        self.assertEqual(conv(torch.randn(2, 16, 13, 17)).shape, (2, 32, 13, 17))

    def test_all_experts_copy_matches_static_conv2d(self):
        torch.manual_seed(0)
        for groups in (1, 2):
            with self.subTest(groups=groups):
                conv = DynamicConv2d(
                    4,
                    6,
                    kernel_size=3,
                    num_experts=3,
                    groups=groups,
                    router=None,
                )
                static_weight = torch.randn(6, 4 // groups, 3, 3)
                static_bias = torch.randn(6)
                with torch.no_grad():
                    conv.weight.copy_(static_weight.unsqueeze(0).expand_as(conv.weight))
                    conv.bias.copy_(static_bias.unsqueeze(0).expand_as(conv.bias))
                x = torch.randn(5, 4, 11, 13)
                expert_weights = torch.rand(5, 3)
                expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

                y = conv.forward_manually(x, expert_weights)
                expected = torch.nn.functional.conv2d(
                    x,
                    static_weight,
                    static_bias,
                    padding=1,
                    groups=groups,
                )

                self.assertTrue(torch.allclose(y, expected, atol=1e-6))

    def test_longcat_subsampling_equivalence(self):
        torch.manual_seed(0)
        baseline = nn.Sequential(
            nn.ConstantPad2d((0, 0, 2, 0), 0),
            nn.Conv2d(1, 32, 3, (2, 1)),
        )
        dynamic = nn.Sequential(
            nn.ConstantPad2d((0, 0, 2, 0), 0),
            DynamicConv2d(1, 32, kernel_size=3, num_experts=4, stride=(2, 1), padding=0),
        )
        source_conv = baseline[1]
        target_conv = dynamic[1]
        with torch.no_grad():
            target_conv.weight.copy_(source_conv.weight.unsqueeze(0).expand_as(target_conv.weight))
            if source_conv.bias is None or target_conv.bias is None:
                raise RuntimeError("test convs must have bias.")
            target_conv.bias.copy_(source_conv.bias.unsqueeze(0).expand_as(target_conv.bias))
        x = torch.randn(3, 1, 20, 80)
        expert_weights = torch.rand(3, 4)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        y = target_conv.forward_manually(dynamic[0](x), expert_weights)
        expected = baseline(x)

        self.assertTrue(torch.allclose(y, expected, atol=1e-6))


class ADTRouter1dTest(unittest.TestCase):
    def test_router_returns_simplex_weights(self):
        router = ADTRouter1d(16, 4)

        weights = router(torch.randn(3, 16, 24))

        self.assertEqual(weights.shape, (3, 4))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6))
        self.assertTrue(torch.isfinite(weights).all())

    def test_router_supports_multiscale(self):
        router = ADTRouter1d(16, 12, multi_scale=True, hidden_size=4)

        weights = router(torch.randn(3, 16, 24))

        self.assertEqual(weights.shape, (3, 12))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6))


class ADTRouter2dTest(unittest.TestCase):
    def test_router_returns_simplex_weights(self):
        router = ADTRouter2d(16, 4)

        weights = router(torch.randn(3, 16, 13, 17))

        self.assertEqual(weights.shape, (3, 4))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6))
        self.assertTrue(torch.isfinite(weights).all())


class DynamicConvShapeTest(unittest.TestCase):
    def test_effective_kernel_and_padding(self):
        self.assertEqual(
            effective_kernel_size_1d(torch.Size([3]), torch.Size([2])), torch.Size([5])
        )
        self.assertEqual(infer_padding_1d(torch.Size([5]), torch.Size([1])), torch.Size([2]))
        self.assertEqual(infer_padding_1d(torch.Size([4]), torch.Size([2])), torch.Size([1]))
        self.assertEqual(
            effective_kernel_size_2d(torch.Size([3, 5]), torch.Size([2, 1])),
            torch.Size([5, 5]),
        )
        self.assertEqual(
            infer_padding_2d(torch.Size([5, 3]), torch.Size([1, 2])),
            torch.Size([2, 1]),
        )

    def test_segment_validation_only_applies_when_segment_size_is_set(self):
        validate_dynamic_conv1d_args(
            kernel_size=torch.Size([4]),
            stride=torch.Size([1]),
            padding=torch.Size([0]),
            dilation=torch.Size([1]),
            segment_size=None,
        )

        with self.assertRaisesRegex(ValueError, "odd effective kernel"):
            validate_dynamic_conv1d_args(
                kernel_size=torch.Size([4]),
                stride=torch.Size([1]),
                padding=torch.Size([0]),
                dilation=torch.Size([1]),
                segment_size=torch.Size([8]),
            )


if __name__ == "__main__":
    unittest.main()
