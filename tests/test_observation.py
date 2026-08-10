import copy
import unittest
import warnings
from collections.abc import Mapping
from typing import Any, cast
from unittest import mock

import torch
from torch import nn

from anytrain import observation
from anytrain.module.quantization import (
    AGRVQConfig,
    AutoGroupResidualVectorQuantizer,
    EmbeddingVectorQuantizer,
    FiniteScalarQuantizer,
    FSQConfig,
    GroupedVectorQuantizer,
    GVQConfig,
    ResidualFiniteScalarQuantizer,
    ResidualVectorQuantizer,
    RFSQConfig,
    RVQConfig,
    VQConfig,
)


def _curve(
    module: nn.Module,
    inputs: tuple[object, ...],
    kwargs: Mapping[str, Any],
    output: object,
):
    del module, inputs, kwargs
    if not isinstance(output, torch.Tensor):
        raise TypeError("test output must be a tensor.")
    return observation.Curve(output.float().mean())


class ObservationTypeTest(unittest.TestCase):
    def test_context_prefix_is_nested_and_restored(self):
        self.assertEqual(observation.context_prefix(), ())

        with observation.context("/train/"):
            self.assertEqual(observation.context_prefix(), ("train",))
            with observation.context("reconstruction/codec"):
                self.assertEqual(
                    observation.context_prefix(),
                    ("train", "reconstruction", "codec"),
                )
            self.assertEqual(observation.context_prefix(), ("train",))

        self.assertEqual(observation.context_prefix(), ())

    def test_context_prefix_requires_a_non_empty_string(self):
        with self.assertRaisesRegex(TypeError, "must be a string"):
            observation.context(cast(Any, 1)).__enter__()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            observation.context("/").__enter__()
        with self.assertRaisesRegex(ValueError, "empty path parts"):
            observation.context("train//loss").__enter__()

    def test_curve_requires_scalar(self):
        with self.assertRaisesRegex(ValueError, "must be scalar"):
            observation.Curve(torch.ones(2))

    def test_distribution_requires_1d_values_and_fixed_edges(self):
        with self.assertRaisesRegex(ValueError, "must be 1D"):
            observation.Distribution(torch.ones(2, 2), edges=(0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            observation.Distribution(torch.ones(2), edges=(0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            observation.Distribution(torch.ones(2), edges=(0.0, float("nan")))

    def test_profile_requires_aligned_floating_1d_pairs(self):
        with self.assertRaisesRegex(TypeError, "Profile.values must be a Tensor"):
            observation.Profile(
                cast(Any, [1.0]),
                torch.tensor([0.5]),
                edges=(0.0, 1.0),
            )
        with self.assertRaisesRegex(TypeError, "Profile.coordinates must be a Tensor"):
            observation.Profile(
                torch.tensor([1.0]),
                cast(Any, [0.5]),
                edges=(0.0, 1.0),
            )
        with self.assertRaisesRegex(ValueError, "both be 1D"):
            observation.Profile(
                torch.ones(1, 1),
                torch.ones(1, 1),
                edges=(0.0, 1.0),
            )
        with self.assertRaisesRegex(ValueError, "same shape"):
            observation.Profile(
                torch.ones(2),
                torch.ones(1),
                edges=(0.0, 1.0),
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            observation.Profile(
                torch.empty(0),
                torch.empty(0),
                edges=(0.0, 1.0),
            )
        with self.assertRaisesRegex(TypeError, "Profile.values must use a floating"):
            observation.Profile(
                torch.ones(1, dtype=torch.int64),
                torch.ones(1),
                edges=(0.0, 1.0),
            )
        with self.assertRaisesRegex(TypeError, "Profile.coordinates must use a floating"):
            observation.Profile(
                torch.ones(1),
                torch.ones(1, dtype=torch.int64),
                edges=(0.0, 1.0),
            )

    def test_profile_requires_strictly_increasing_edges_and_matching_device(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            observation.Profile(
                torch.ones(1),
                torch.ones(1),
                edges=(0.0, 0.0),
            )
        with self.assertRaisesRegex(ValueError, "same device"):
            observation.Profile(
                torch.ones(1),
                torch.ones(1, device="meta"),
                edges=(0.0, 1.0),
            )

    def test_ema_history_requires_finite_half_life(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            observation.EMAHistory(float("inf"))

    def test_repeated_forward_observations_require_an_explicit_reduction(self):
        with self.assertRaisesRegex(TypeError, "required positional argument: 'reduction'"):
            cast(Any, observation.OutputObservation)("value", _curve)
        with self.assertRaisesRegex(TypeError, "required positional argument: 'reduction'"):
            cast(Any, observation.ForwardEvent)("value")

    def test_observation_reduction_requires_the_enum_and_compatible_history(self):
        with self.assertRaisesRegex(TypeError, "must be a Reduction"):
            observation.OutputObservation("value", _curve, reduction=cast(Any, "mean"))
        with self.assertRaisesRegex(ValueError, "requires EMAHistory"):
            observation.OutputObservation(
                "value",
                _curve,
                reduction=observation.Reduction.Merge,
            )
        with self.assertRaisesRegex(ValueError, "only Merge"):
            observation.ForwardEvent(
                "value",
                reduction=observation.Reduction.Last,
                history=observation.EMAHistory(),
            )

    def test_bins_accept_multidimensional_codes_with_group_axis(self):
        value = observation.Bins(
            torch.tensor([[[0, 1], [1, -1]]]),
            sizes=(2, 3),
            group_dim=-1,
            active_mask=torch.tensor([[[True, True], [True, False]]]),
            group_names=("left", "right"),
        )
        self.assertEqual(value.values.shape, (1, 2, 2))

    def test_bins_reject_normalized_group_name_collisions(self):
        with self.assertRaisesRegex(ValueError, "unique after normalization"):
            observation.Bins(
                torch.tensor([[0, 1]]),
                sizes=(2, 2),
                group_dim=-1,
                group_names=("book", "/book/"),
            )

    def test_tensor_observation_equality_is_identity_based(self):
        left = observation.Bins(torch.tensor([0]), sizes=(2,))
        right = observation.Bins(torch.tensor([0]), sizes=(2,))
        self.assertNotEqual(left, right)

    def test_recommended_observation_warns_once_after_training_forward(self):
        class Recommended(nn.Module):
            def __init__(self):
                super().__init__()
                observation.registry.recommend(self)

            def forward(self, value):
                return value

        observation.registry.register(
            Recommended,
            (
                observation.ForwardObservation(
                    "value",
                    _curve,
                    reduction=observation.Reduction.Mean,
                    recommended=True,
                ),
            ),
        )
        module = Recommended().eval()
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with torch.no_grad():
                module(torch.tensor(1.0))
            self.assertEqual(records, [])
            module.train()
            module(torch.tensor(1.0))
            module(torch.tensor(2.0))
        matching = [
            item for item in records if issubclass(item.category, observation.RecommendationWarning)
        ]
        self.assertEqual(len(matching), 1)
        self.assertIn("ObservationCallback", str(matching[0].message))

    def test_all_public_quantizers_register_recommended_observation(self):
        vq_configs = (VQConfig(input_dim=2, codebook_size=4),) * 2
        modules = (
            FiniteScalarQuantizer(FSQConfig(input_dim=2, levels=(3, 3))),
            EmbeddingVectorQuantizer(VQConfig(input_dim=2, codebook_size=4)),
            GroupedVectorQuantizer(GVQConfig(input_dim=2, group_sizes=(2, 2))),
            ResidualVectorQuantizer(RVQConfig(vq_configs)),
            ResidualFiniteScalarQuantizer(
                RFSQConfig(input_dim=2, levels=(3, 3), scales=(1.0, 0.5))
            ),
            AutoGroupResidualVectorQuantizer(
                AGRVQConfig(input_dim=2, num_codebooks=2, group_size=2, codebook_dim=1)
            ),
        )
        for module in modules:
            specs: Any = observation.registry.specs(module)
            self.assertEqual(
                tuple(spec.name for spec in specs),
                ("quantization", "quantization/loss"),
            )
            self.assertTrue(specs[0].recommended)
            self.assertTrue(specs[0].covers_descendants)
            self.assertEqual(specs[0].reduction, observation.Reduction.Merge)
            self.assertEqual(specs[1].reduction, observation.Reduction.Mean)
            self.assertIsNone(specs[1].history)

    def test_recommendation_hint_never_requires_a_process_group(self):
        class Recommended(nn.Module):
            def __init__(self):
                super().__init__()
                observation.registry.recommend(self)

            def forward(self, value):
                return value

        observation.registry.register(
            Recommended,
            (
                observation.ForwardObservation(
                    "value",
                    _curve,
                    reduction=observation.Reduction.Mean,
                    recommended=True,
                ),
            ),
        )
        module = Recommended().train()
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch(
                "torch.distributed.get_rank",
                side_effect=ValueError("process group is unavailable"),
            ),
        ):
            self.assertEqual(module(torch.tensor(1.0)).item(), 1.0)

    def test_parent_recommendation_suppresses_descendant_hint(self):
        class Child(nn.Module):
            def __init__(self):
                super().__init__()
                observation.registry.recommend(self)

            def forward(self, value):
                return value

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()
                observation.registry.recommend(self)

            def forward(self, value):
                return self.child(value)

        observation.registry.register(
            Child,
            (
                observation.ForwardObservation(
                    "value",
                    _curve,
                    reduction=observation.Reduction.Mean,
                    recommended=True,
                ),
            ),
        )
        observation.registry.register(
            Parent,
            (
                observation.ForwardObservation(
                    "value",
                    _curve,
                    reduction=observation.Reduction.Mean,
                    recommended=True,
                    covers_descendants=True,
                ),
            ),
        )
        module = Parent().train()
        self.assertEqual(len(module.child._forward_pre_hooks), 0)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            module(torch.tensor(1.0))
        matching = [
            item for item in records if issubclass(item.category, observation.RecommendationWarning)
        ]
        self.assertEqual(len(matching), 1)
        self.assertIn("Parent", str(matching[0].message))

    def test_backward_recommendation_does_not_suppress_supported_descendant(self):
        class Child(nn.Module):
            def __init__(self):
                super().__init__()
                observation.registry.recommend(self)

            def forward(self, value):
                return value

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()

            def forward(self, value):
                return self.child(value)

        def collect_backward(module, grad_outputs):
            del module, grad_outputs
            return None

        observation.registry.register(
            Child,
            (
                observation.ForwardObservation(
                    "value",
                    _curve,
                    reduction=observation.Reduction.Mean,
                    recommended=True,
                ),
            ),
        )
        observation.registry.register(
            Parent,
            (
                observation.BackwardObservation(
                    "gradient",
                    collect_backward,
                    recommended=True,
                    covers_descendants=True,
                ),
            ),
        )
        module = Parent().train()
        self.assertEqual(len(module._forward_pre_hooks), 0)
        self.assertEqual(len(module.child._forward_pre_hooks), 1)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            module(torch.tensor(1.0))
        matching = [
            item for item in records if issubclass(item.category, observation.RecommendationWarning)
        ]
        self.assertEqual(len(matching), 1)
        self.assertIn("Child", str(matching[0].message))

    def test_registry_rejects_bound_module_collectors(self):
        class Registered(nn.Module):
            def _collect(self, module, inputs, kwargs, output):
                del module, inputs, kwargs, output
                return None

        module = Registered()
        with self.assertRaisesRegex(TypeError, "must not be bound"):
            observation.registry.register(
                Registered,
                (
                    observation.ForwardObservation(
                        "value",
                        module._collect,
                        reduction=observation.Reduction.Mean,
                    ),
                ),
            )

    def test_quantizer_observation_declaration_survives_deepcopy(self):
        module = FiniteScalarQuantizer(FSQConfig(input_dim=2, levels=(3, 3)))
        copied = copy.deepcopy(module)
        self.assertEqual(
            observation.registry.specs(copied),
            observation.registry.specs(module),
        )
        self.assertEqual(len(copied._forward_pre_hooks), 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", observation.RecommendationWarning)
            copied(torch.randn(1, 2))
        self.assertEqual(len(copied._forward_pre_hooks), 0)

    def test_registry_rejects_normalized_spec_name_collisions(self):
        class Registered(nn.Module):
            pass

        with self.assertRaisesRegex(ValueError, "unique after normalization"):
            observation.registry.register(
                Registered,
                (
                    observation.ForwardObservation(
                        "value",
                        _curve,
                        reduction=observation.Reduction.Mean,
                    ),
                    observation.ForwardObservation(
                        "/value/",
                        _curve,
                        reduction=observation.Reduction.Mean,
                    ),
                ),
            )

    def test_registry_rejects_duplicate_class_registration(self):
        class Registered(nn.Module):
            pass

        observation.registry.register(
            Registered,
            (
                observation.ForwardObservation(
                    "value",
                    _curve,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            observation.registry.register(
                Registered,
                (
                    observation.ForwardObservation(
                        "other",
                        _curve,
                        reduction=observation.Reduction.Mean,
                    ),
                ),
            )

    def test_registry_requires_explicit_concrete_subclass_registration(self):
        class Registered(nn.Module):
            pass

        class Child(Registered):
            pass

        observation.registry.register(
            Registered,
            (
                observation.ForwardObservation(
                    "value",
                    _curve,
                    reduction=observation.Reduction.Mean,
                ),
            ),
        )
        self.assertEqual(len(observation.registry.specs(Registered())), 1)
        self.assertEqual(observation.registry.specs(Child()), ())


if __name__ == "__main__":
    unittest.main()
