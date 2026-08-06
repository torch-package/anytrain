import copy
import gc
import unittest
import warnings
from unittest import mock
from weakref import ref

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
    kwargs: dict[str, object],
    output: object,
):
    del module, inputs, kwargs
    if not isinstance(output, torch.Tensor):
        raise TypeError("test output must be a tensor.")
    return observation.Curve(output.float().mean())


class ObservationTypeTest(unittest.TestCase):
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

    def test_ema_history_requires_finite_half_life(self):
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            observation.EMAHistory(float("inf"))

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
                observation.register(
                    self,
                    (
                        observation.ForwardObservation(
                            "value",
                            _curve,
                            recommended=True,
                        ),
                    ),
                )

            def forward(self, value):
                return value

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
            specs = observation.specs(module)
            self.assertEqual(len(specs), 1)
            self.assertTrue(specs[0].recommended)
            self.assertTrue(specs[0].covers_descendants)

    def test_recommendation_hint_never_requires_a_process_group(self):
        class Recommended(nn.Module):
            def __init__(self):
                super().__init__()
                observation.register(
                    self,
                    (
                        observation.ForwardObservation(
                            "value",
                            _curve,
                            recommended=True,
                        ),
                    ),
                )

            def forward(self, value):
                return value

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
                observation.register(
                    self,
                    (
                        observation.ForwardObservation(
                            "value",
                            _curve,
                            recommended=True,
                        ),
                    ),
                )

            def forward(self, value):
                return value

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()
                observation.register(
                    self,
                    (
                        observation.ForwardObservation(
                            "value",
                            _curve,
                            recommended=True,
                            covers_descendants=True,
                        ),
                    ),
                )

            def forward(self, value):
                return self.child(value)

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
                observation.register(
                    self,
                    (
                        observation.ForwardObservation(
                            "value",
                            _curve,
                            recommended=True,
                        ),
                    ),
                )

            def forward(self, value):
                return value

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()
                observation.register(
                    self,
                    (
                        observation.BackwardObservation(
                            "gradient",
                            self._collect,
                            recommended=True,
                            covers_descendants=True,
                        ),
                    ),
                )

            def forward(self, value):
                return self.child(value)

            def _collect(self, module, grad_outputs):
                del module, grad_outputs
                return None

        module = Parent().train()
        self.assertEqual(len(module._forward_pre_hooks), 0)
        self.assertEqual(len(module.child._forward_pre_hooks), 1)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            module(torch.tensor(1.0))
        matching = [
            item
            for item in records
            if issubclass(item.category, observation.RecommendationWarning)
        ]
        self.assertEqual(len(matching), 1)
        self.assertIn("Child", str(matching[0].message))

    def test_bound_collector_registration_does_not_keep_module_alive(self):
        class Registered(nn.Module):
            def __init__(self):
                super().__init__()
                observation.register(
                    self,
                    (
                        observation.ForwardObservation(
                            "value",
                            self._collect,
                            recommended=True,
                        ),
                    ),
                )

            def _collect(self, module, inputs, kwargs, output):
                del module, inputs, kwargs, output
                return None

        module = Registered()
        module_ref = ref(module)
        del module
        gc.collect()
        self.assertIsNone(module_ref())

    def test_instance_registered_bound_collector_rebinds_on_deepcopy(self):
        class Registered(nn.Module):
            def __init__(self):
                super().__init__()
                observation.register(
                    self,
                    (observation.ForwardObservation("value", self._collect),),
                )

            def _collect(self, module, inputs, kwargs, output):
                del module, inputs, kwargs, output
                return None

        module = Registered()
        copied = copy.deepcopy(module)
        collector = observation.specs(copied)[0].collect
        self.assertIs(getattr(collector, "__self__", None), copied)

    def test_quantizer_observation_declaration_survives_deepcopy(self):
        module = FiniteScalarQuantizer(FSQConfig(input_dim=2, levels=(3, 3)))
        copied = copy.deepcopy(module)
        self.assertEqual(observation.specs(copied), observation.specs(module))
        self.assertEqual(len(copied._forward_pre_hooks), 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", observation.RecommendationWarning)
            copied(torch.randn(1, 2))
        self.assertEqual(len(copied._forward_pre_hooks), 0)

    def test_register_rejects_normalized_spec_name_collisions(self):
        module = nn.Identity()
        with self.assertRaisesRegex(ValueError, "unique after normalization"):
            observation.register(
                module,
                (
                    observation.ForwardObservation("value", _curve),
                    observation.ForwardObservation("/value/", _curve),
                ),
            )


if __name__ == "__main__":
    unittest.main()
