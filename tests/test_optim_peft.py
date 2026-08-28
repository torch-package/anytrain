import importlib.util
import unittest
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import Linear as PeftLinear
    from transformers.pytorch_utils import Conv1D

    from anytrain.optim import CompositeOptimizer, LoRAMuon, create_llm_optimizer
    from anytrain.optim.peft import (
        create_lora_muon_adamw_optimizer,
        create_lora_muon_optimizer,
        lora_factor_pairs,
    )

    PEFT_AVAILABLE = True
else:
    PEFT_AVAILABLE = importlib.util.find_spec("peft") is not None
    if PEFT_AVAILABLE:
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora import Linear as PeftLinear
        from transformers.pytorch_utils import Conv1D

        from anytrain.optim import CompositeOptimizer, LoRAMuon, create_llm_optimizer
        from anytrain.optim.peft import (
            create_lora_muon_adamw_optimizer,
            create_lora_muon_optimizer,
            lora_factor_pairs,
        )


@unittest.skipUnless(PEFT_AVAILABLE, "PEFT is not installed")
class PeftLoRAMuonTest(unittest.TestCase):
    def test_llm_helper_routes_fsdp2_wrapped_peft_linear_to_lora_muon(self):
        from torch.distributed.fsdp import FSDPModule

        model = _linear_model(bias="all", init_lora_weights=False)
        layer = _linear_layers(model)[0]
        wrapped_type = type("FSDPLinear", (FSDPModule, PeftLinear), {})
        layer.__class__ = wrapped_type

        optimizer = cast(
            CompositeOptimizer,
            create_llm_optimizer(model, preset="sft", optimizer="muon", lr=0.03),
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        self.assertIsInstance(optimizer.optimizers["lora_muon"], LoRAMuon)
        pair = lora_factor_pairs(model)[0]
        self.assertIs(type(pair.left), nn.Parameter)
        self.assertIs(type(pair.right), nn.Parameter)

    def setUp(self):
        torch.manual_seed(11)

    def test_collects_active_linear_adapter_with_dropout(self):
        model = _linear_model(lora_dropout=0.25)
        layer = _linear_layers(model)[0]

        pairs = lora_factor_pairs(model)

        self.assertEqual(len(pairs), 1)
        self.assertIs(pairs[0].left, layer.lora_B["default"].weight)
        self.assertIs(pairs[0].right, layer.lora_A["default"].weight)
        self.assertEqual(pairs[0].scale, layer.scaling["default"])

    def test_factory_steps_a_real_peft_model(self):
        model = _linear_model()
        layer = _linear_layers(model)[0]
        left_before = layer.lora_B["default"].weight.detach().clone()
        right_before = layer.lora_A["default"].weight.detach().clone()
        optimizer = create_lora_muon_optimizer(
            model,
            lr=0.03,
            momentum=0.8,
            weight_decay=0.1,
        )

        loss = model(torch.randn(6, 4)).square().mean()
        loss.backward()
        optimizer.step()

        self.assertIsInstance(optimizer, LoRAMuon)
        self.assertFalse(torch.equal(layer.lora_B["default"].weight, left_before))
        self.assertFalse(torch.equal(layer.lora_A["default"].weight, right_before))

    def test_factory_rejects_peft_default_zero_initialization(self):
        model = _linear_model(init_lora_weights=True)

        with self.assertRaisesRegex(ValueError, "numerically full column rank"):
            create_lora_muon_optimizer(model, lr=0.03)

    def test_factory_accepts_pissa_full_rank_initialization(self):
        model = _linear_model(init_lora_weights="pissa")

        optimizer = create_lora_muon_optimizer(model, lr=0.03)

        self.assertIsInstance(optimizer, LoRAMuon)

    def test_maps_rslora_scale_and_fan_in_fan_out_coordinates(self):
        rslora_model = _linear_model(use_rslora=True)
        pair = lora_factor_pairs(rslora_model)[0]
        self.assertAlmostEqual(pair.scale, 4 / (2**0.5))

        conv1d_model = get_peft_model(
            cast(Any, nn.Sequential(Conv1D(5, 4))),
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["0"],
                init_lora_weights=False,
                fan_in_fan_out=True,
            ),
        )
        layer = _linear_layers(conv1d_model)[0]
        pair = lora_factor_pairs(conv1d_model)[0]
        inputs = torch.randn(3, 4)

        actual_delta = conv1d_model(inputs) - layer.get_base_layer()(inputs)
        expected_delta = pair.scale * inputs @ pair.right.mT @ pair.left.mT

        self.assertTrue(layer.fan_in_fan_out)
        torch.testing.assert_close(actual_delta, expected_delta)

    def test_collects_multiple_active_adapters_and_explicit_selection(self):
        config = _config()
        model = _linear_model(config=config)
        model.add_adapter("second", config)
        model.base_model.set_adapter(["default", "second"])
        layer = _linear_layers(model)[0]

        all_pairs = lora_factor_pairs(model)
        selected = lora_factor_pairs(model, adapter_names=["second"])

        self.assertEqual(len(all_pairs), 2)
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0].left, layer.lora_B["second"].weight)
        with self.assertRaisesRegex(ValueError, "without an optimizer"):
            create_lora_muon_optimizer(model, lr=0.03, adapter_names=["second"])

    def test_explicit_adapter_must_exist_and_be_active(self):
        config = _config()
        model = _linear_model(config=config)
        model.add_adapter("second", config)

        with self.assertRaisesRegex(ValueError, "must be active"):
            lora_factor_pairs(model, adapter_names=["second"])
        with self.assertRaisesRegex(ValueError, "were not found"):
            lora_factor_pairs(model, adapter_names=["missing"])

    def test_rejects_variant_merged_and_disabled_layers(self):
        variant_model = _linear_model(use_dora=True)
        with self.assertRaisesRegex(ValueError, "LoRA variant"):
            lora_factor_pairs(variant_model)

        merged_model = _linear_model()
        _linear_layers(merged_model)[0].merge()
        with self.assertRaisesRegex(ValueError, "must be unmerged"):
            lora_factor_pairs(merged_model)

        disabled_model = _linear_model()
        _linear_layers(disabled_model)[0].enable_adapters(False)
        with self.assertRaisesRegex(ValueError, "adapters disabled"):
            lora_factor_pairs(disabled_model)

    def test_rejects_lora_bias_and_unsupported_layer_types(self):
        bias_model = _linear_model(lora_bias=True)
        with self.assertRaisesRegex(ValueError, "must not use LoRA bias"):
            lora_factor_pairs(bias_model)

        conv_model = get_peft_model(
            cast(Any, nn.Sequential(nn.Conv2d(2, 3, kernel_size=1))),
            LoraConfig(r=1, target_modules=["0"], init_lora_weights=False),
        )
        with self.assertRaisesRegex(TypeError, "only vanilla PEFT Linear"):
            lora_factor_pairs(conv_model)

        embedding_model = get_peft_model(
            cast(Any, nn.Sequential(nn.Embedding(8, 4))),
            LoraConfig(r=2, target_modules=["0"], init_lora_weights=False),
        )
        with self.assertRaisesRegex(TypeError, "only vanilla PEFT Linear"):
            lora_factor_pairs(embedding_model)

    def test_factory_rejects_uncovered_trainable_parameters(self):
        model = _linear_model(bias="all")

        with self.assertRaisesRegex(ValueError, "without an optimizer"):
            create_lora_muon_optimizer(model, lr=0.03)

        model = get_peft_model(
            cast(
                Any,
                nn.Sequential(
                    nn.Linear(4, 4, bias=False),
                    nn.Linear(4, 5, bias=False),
                ),
            ),
            LoraConfig(
                r=2,
                target_modules=["0"],
                modules_to_save=["1"],
                init_lora_weights=False,
            ),
        )
        with self.assertRaisesRegex(ValueError, "without an optimizer"):
            create_lora_muon_optimizer(model, lr=0.03)

    def test_mixed_factory_covers_remaining_trainable_with_adamw(self):
        model = _linear_model(bias="all", init_lora_weights=False)
        layer = _linear_layers(model)[0]
        left_before = layer.lora_B["default"].weight.detach().clone()
        bias_before = layer.base_layer.bias.detach().clone()

        optimizer = cast(
            CompositeOptimizer,
            create_lora_muon_adamw_optimizer(
                model,
                lr=0.03,
                adamw={"lr": 0.03, "weight_decay": 0.0},
            ),
        )
        loss = model(torch.randn(6, 4)).square().mean()
        loss.backward()
        optimizer.step()

        self.assertIsInstance(optimizer, CompositeOptimizer)
        self.assertEqual(set(optimizer.optimizers), {"lora_muon", "adamw"})
        self.assertIsInstance(optimizer.optimizers["lora_muon"], LoRAMuon)
        self.assertFalse(torch.equal(layer.lora_B["default"].weight, left_before))
        self.assertFalse(torch.equal(layer.base_layer.bias, bias_before))

    def test_mixed_factory_returns_pure_lora_muon_without_remaining(self):
        model = _linear_model(bias="none", init_lora_weights=False)
        for name, parameter in model.named_parameters():
            if "lora_" not in name:
                parameter.requires_grad_(False)

        optimizer = create_lora_muon_adamw_optimizer(
            model,
            lr=0.03,
            adamw={"lr": 0.03},
        )

        self.assertIsInstance(optimizer, LoRAMuon)

    def test_llm_helper_routes_muon_to_lora_muon_for_peft(self):
        model = _linear_model(bias="all", init_lora_weights=False)

        optimizer = cast(
            CompositeOptimizer,
            create_llm_optimizer(model, preset="sft", optimizer="muon", lr=0.03),
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        self.assertEqual(set(optimizer.optimizers), {"lora_muon", "adamw"})
        self.assertIsInstance(optimizer.optimizers["lora_muon"], LoRAMuon)

    def test_llm_helper_adamw_does_not_use_lora_muon(self):
        model = _linear_model(bias="all", init_lora_weights=False)

        optimizer = create_llm_optimizer(model, preset="sft", optimizer="adamw", lr=0.03)

        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_rejects_tensor_parallel_markers(self):
        layer_model = _linear_model()
        layer = _linear_layers(layer_model)[0]
        layer._tp_info = object()
        with self.assertRaisesRegex(ValueError, "tensor parallelism"):
            lora_factor_pairs(layer_model)

        base_model = _linear_model()
        base_layer = _linear_layers(base_model)[0].get_base_layer()
        base_layer._hf_tp_plan = "colwise"
        with self.assertRaisesRegex(ValueError, "tensor parallelism"):
            lora_factor_pairs(base_model)

    def test_rejects_factor_storage_aliases(self):
        model = get_peft_model(
            cast(
                Any,
                nn.Sequential(
                    nn.Linear(4, 4, bias=False),
                    nn.Linear(4, 4, bias=False),
                ),
            ),
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["0", "1"],
                init_lora_weights=False,
            ),
        )
        first, second = _linear_layers(model)
        second.lora_A["default"].weight = nn.Parameter(first.lora_A["default"].weight.detach())

        with self.assertRaisesRegex(ValueError, "shares storage"):
            lora_factor_pairs(model)


def _config(**overrides: Any) -> Any:
    values = {
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["0"],
        "init_lora_weights": False,
    }
    values.update(overrides)
    return LoraConfig(**values)


def _linear_model(*, config: Any = None, **overrides: Any) -> Any:
    if config is None:
        config = _config(**overrides)
    return get_peft_model(cast(Any, nn.Sequential(nn.Linear(4, 5, bias=True))), config)


def _linear_layers(module: nn.Module) -> list[Any]:
    return [child for child in module.modules() if type(child) is PeftLinear]
