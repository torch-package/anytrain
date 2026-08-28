import unittest
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import nn

import anytrain.optim as optim_api
import anytrain.optim.scheduler as scheduler_api
from anytrain.optim import (
    CompositeOptimizer,
    checkpoint_optimizers,
    create_adamw_optimizer,
    create_llm_optimizer,
    create_muon_adamw_optimizer,
    create_scheduler,
    muon_available,
    split_adamw_decay_params,
    split_muon_params,
)
from anytrain.optim.options import (
    AdamWOptions,
    MuonAdjustLRFn,
    MuonOptions,
)

MUON_AVAILABLE = muon_available()


def _param_ids(params: list[nn.Parameter]) -> set[int]:
    return {id(param) for param in params}


def _group_param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(param) for group in optimizer.param_groups for param in group["params"]}


def _group_lrs_by_param_id(optimizer: torch.optim.Optimizer) -> dict[int, float]:
    return {id(param): group["lr"] for group in optimizer.param_groups for param in group["params"]}


def _adamw_options(lr: float, weight_decay: float = 0.1) -> AdamWOptions:
    return {"lr": lr, "weight_decay": weight_decay}


def _muon_options(lr: float, weight_decay: float = 0.1) -> MuonOptions:
    return {"lr": lr, "weight_decay": weight_decay}


class CheckpointOptimizersTest(unittest.TestCase):
    def test_checkpoint_optimizers_returns_regular_optimizer(self):
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)

        self.assertEqual(checkpoint_optimizers(optimizer), (optimizer,))

    def test_checkpoint_optimizers_flattens_composite_in_child_order(self):
        first = torch.optim.SGD([nn.Parameter(torch.tensor(1.0))], lr=1.0)
        second = torch.optim.AdamW([nn.Parameter(torch.tensor(2.0))], lr=1.0)
        optimizer = CompositeOptimizer({"first": first, "second": second})

        self.assertEqual(checkpoint_optimizers(optimizer), (first, second))

    def test_checkpoint_optimizers_recursively_flattens_composites(self):
        first = torch.optim.SGD([nn.Parameter(torch.tensor(1.0))], lr=1.0)
        second = torch.optim.AdamW([nn.Parameter(torch.tensor(2.0))], lr=1.0)
        third = torch.optim.SGD([nn.Parameter(torch.tensor(3.0))], lr=1.0)
        optimizer = CompositeOptimizer(
            {
                "nested": CompositeOptimizer({"first": first, "second": second}),
                "third": third,
            }
        )

        self.assertEqual(checkpoint_optimizers(optimizer), (first, second, third))

    def test_checkpoint_optimizers_rejects_duplicate_child_instances(self):
        parameter = nn.Parameter(torch.tensor(1.0))
        child = torch.optim.SGD([parameter], lr=1.0)
        optimizer = CompositeOptimizer({"first": child})
        optimizer.optimizers["duplicate"] = child

        with self.assertRaisesRegex(ValueError, "appears more than once"):
            checkpoint_optimizers(optimizer)

    def test_checkpoint_optimizers_supports_distributed_checkpoint_state_dict(self):
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        model = nn.Linear(2, 2)
        sgd = torch.optim.SGD([model.weight], lr=0.1, momentum=0.9)
        adamw = torch.optim.AdamW([model.bias], lr=0.1)
        optimizer = CompositeOptimizer({"sgd": sgd, "adamw": adamw})

        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        state = get_optimizer_state_dict(model, checkpoint_optimizers(optimizer))

        self.assertEqual(set(state), {"state", "param_groups"})
        self.assertEqual(len(state["state"]), 2)
        self.assertEqual(len(state["param_groups"]), 2)


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)


class SplitMuonParamsTest(unittest.TestCase):
    def test_split_muon_params_uses_2d_weight_rule(self):
        model = nn.Sequential(
            nn.Linear(4, 3),
            nn.LayerNorm(3),
        )

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model[0].weight)})
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model[0].bias), id(model[1].weight), id(model[1].bias)},
        )

    def test_split_muon_params_keeps_module_biases_non_muon(self):
        class ModuleWithMatrixBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 4))
                self.bias = nn.Parameter(torch.randn(3, 4))

        model = ModuleWithMatrixBias()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model.weight)})
        self.assertEqual(_param_ids(non_muon_parameters), {id(model.bias)})

    def test_split_muon_params_excludes_embedding_weights_by_default(self):
        model = nn.Sequential(
            nn.Embedding(8, 3),
            nn.Linear(3, 4),
        )

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model[1].weight)})
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model[0].weight), id(model[1].bias)},
        )

    def test_split_muon_params_uses_embedding_type_not_name(self):
        class FakeEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 3))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = FakeEmbedding()

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model.embed_tokens.weight)})
        self.assertEqual(non_muon_parameters, [])

    def test_split_muon_params_includes_head_named_weights_by_default(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)
                self.lm_head = nn.Linear(4, 8, bias=False)

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(
            _param_ids(muon_parameters),
            {id(model.proj.weight), id(model.lm_head.weight)},
        )
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model.proj.bias)},
        )

    def test_split_muon_params_excludes_explicit_modules(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)
                self.lm_head = nn.Linear(4, 8, bias=False)

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(
            model,
            excluded_modules=(model.lm_head,),
        )

        self.assertEqual(
            _param_ids(muon_parameters),
            {id(model.proj.weight)},
        )
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model.proj.bias), id(model.lm_head.weight)},
        )

    def test_split_muon_params_excludes_explicit_module_subtree(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Linear(4, 8, bias=False)
                self.output = nn.Sequential(
                    nn.Linear(4, 4, bias=False),
                    nn.Linear(4, 8, bias=False),
                )

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(
            model,
            excluded_modules=(model.output,),
        )

        self.assertEqual(_param_ids(muon_parameters), {id(model.decoder.weight)})
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model.output[0].weight), id(model.output[1].weight)},
        )

    def test_split_muon_params_excludes_tied_embedding_head_weight(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(8, 4)
                self.proj = nn.Linear(4, 4)
                self.lm_head = nn.Linear(4, 8, bias=False)
                self.lm_head.weight = self.embed_tokens.weight

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model.proj.weight)})
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model.embed_tokens.weight), id(model.proj.bias)},
        )

    def test_split_muon_params_uses_non_muon_for_any_shared_parameter_owner(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 8, bias=False)
                self.embed_tokens = nn.Embedding(8, 4)
                self.proj.weight = self.embed_tokens.weight

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(muon_parameters, [])
        self.assertEqual(_param_ids(non_muon_parameters), {id(model.embed_tokens.weight)})

    def test_split_muon_params_excludes_custom_norm_class_by_default(self):
        class TinyRMSNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(3, 3))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(3, 3, bias=False)
                self.rms_norm = TinyRMSNorm()

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model.proj.weight)})
        self.assertEqual(_param_ids(non_muon_parameters), {id(model.rms_norm.weight)})

    def test_split_muon_params_skips_frozen_params_by_default(self):
        model = nn.Linear(4, 3, bias=False)
        model.weight.requires_grad_(False)

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(muon_parameters, [])
        self.assertEqual(non_muon_parameters, [])

    def test_split_muon_params_can_include_frozen_params(self):
        model = nn.Linear(4, 3, bias=False)
        model.weight.requires_grad_(False)

        muon_parameters, non_muon_parameters = split_muon_params(model, requires_grad_only=False)

        self.assertEqual(_param_ids(muon_parameters), {id(model.weight)})
        self.assertEqual(non_muon_parameters, [])

    def test_split_muon_params_requires_boolean_controls(self):
        model = nn.Linear(4, 3, bias=False)

        with self.assertRaisesRegex(TypeError, "requires_grad_only must be a bool"):
            split_muon_params(model, requires_grad_only=cast(Any, 1))

    def test_split_muon_params_rejects_non_module(self):
        with self.assertRaisesRegex(TypeError, "torch.nn.Module"):
            split_muon_params(cast(Any, torch.tensor(1.0)))

    def test_split_muon_params_rejects_excluded_module_outside_root(self):
        with self.assertRaisesRegex(ValueError, "belong"):
            split_muon_params(nn.Linear(4, 4), excluded_modules=(nn.Linear(4, 4),))


class AdamWOptimizerTest(unittest.TestCase):
    def test_split_adamw_decay_params_excludes_norm_embedding_and_bias(self):
        model = TinyLLM()

        decay_params, no_decay_params = split_adamw_decay_params(model)

        self.assertEqual(
            _param_ids(decay_params), {id(model.proj.weight), id(model.lm_head.weight)}
        )
        self.assertEqual(
            _param_ids(no_decay_params),
            {
                id(model.embed_tokens.weight),
                id(model.proj.bias),
                id(model.norm.weight),
                id(model.norm.bias),
            },
        )

    def test_split_adamw_decay_params_excludes_explicit_modules(self):
        model = TinyLLM()

        decay_params, no_decay_params = split_adamw_decay_params(
            model,
            excluded_modules=(model.lm_head,),
        )

        self.assertEqual(_param_ids(decay_params), {id(model.proj.weight)})
        self.assertEqual(
            _param_ids(no_decay_params),
            {
                id(model.embed_tokens.weight),
                id(model.proj.bias),
                id(model.norm.weight),
                id(model.norm.bias),
                id(model.lm_head.weight),
            },
        )

    def test_split_adamw_decay_params_decays_standard_weight_parameters_by_default(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(2, 4, kernel_size=3, bias=False)
                self.proj = nn.Linear(4, 4, bias=False)

        model = Model()

        decay_params, no_decay_params = split_adamw_decay_params(model)

        self.assertEqual(_param_ids(decay_params), {id(model.conv.weight), id(model.proj.weight)})
        self.assertEqual(_param_ids(no_decay_params), set())

    def test_create_adamw_optimizer_uses_decay_and_no_decay_groups(self):
        model = TinyLLM()
        optimizer = create_adamw_optimizer(model, _adamw_options(3e-4))

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.1, 0.0])
        self.assertEqual(_group_param_ids(optimizer), {id(param) for param in model.parameters()})

    def test_create_adamw_optimizer_applies_lr_scale_rules_by_module_subtree(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(4, 4),
                    nn.Linear(4, 4),
                )
                self.head = nn.Linear(4, 2, bias=False)

        model = Model()
        optimizer = create_adamw_optimizer(
            model,
            _adamw_options(1.0),
            lr_scale_rules=[
                {"name": "encoder", "lr_scale": 0.5},
                {"name": "encoder.1", "lr_scale": 0.25},
                {"name": "head", "lr_scale": 2.0},
            ],
        )

        lrs = _group_lrs_by_param_id(optimizer)
        self.assertEqual(lrs[id(model.encoder[0].weight)], 0.5)
        self.assertEqual(lrs[id(model.encoder[0].bias)], 0.5)
        self.assertEqual(lrs[id(model.encoder[1].weight)], 0.25)
        self.assertEqual(lrs[id(model.encoder[1].bias)], 0.25)
        self.assertEqual(lrs[id(model.head.weight)], 2.0)

    def test_create_adamw_optimizer_rejects_zero_lr_scale(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            create_adamw_optimizer(
                nn.Sequential(nn.Linear(4, 4)),
                _adamw_options(1.0),
                lr_scale_rules=[{"name": "0", "lr_scale": 0.0}],
            )

    def test_create_adamw_optimizer_rejects_nonfinite_lr_scale(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                create_adamw_optimizer(
                    TinyLLM(),
                    _adamw_options(3e-4),
                    lr_scale_rules=[{"name": "proj", "lr_scale": value}],
                )

    def test_create_adamw_optimizer_rejects_unknown_lr_scale_rule_name(self):
        with self.assertRaisesRegex(ValueError, "belong"):
            create_adamw_optimizer(
                nn.Sequential(nn.Linear(4, 4)),
                _adamw_options(1.0),
                lr_scale_rules=[{"name": "missing", "lr_scale": 0.5}],
            )

    def test_create_adamw_optimizer_rejects_same_specificity_shared_param_scales(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.Linear(4, 4, bias=False)
                self.right = nn.Linear(4, 4, bias=False)
                self.right.weight = self.left.weight

        model = Model()

        with self.assertRaisesRegex(ValueError, "conflicting"):
            create_adamw_optimizer(
                model,
                _adamw_options(1.0),
                lr_scale_rules=[
                    {"name": "left", "lr_scale": 0.5},
                    {"name": "right", "lr_scale": 2.0},
                ],
            )

    def test_split_adamw_decay_params_rejects_selected_param_outside_module(self):
        with self.assertRaisesRegex(ValueError, "belong"):
            split_adamw_decay_params(
                nn.Linear(4, 4),
                selected_params=[nn.Parameter(torch.randn(4, 4))],
            )

    def test_split_adamw_decay_params_requires_boolean_controls(self):
        model = nn.Linear(4, 4)

        with self.assertRaisesRegex(TypeError, "requires_grad_only must be a bool"):
            split_adamw_decay_params(model, requires_grad_only=cast(Any, 1))
        with self.assertRaisesRegex(TypeError, "decay_selected_params must be a bool"):
            split_adamw_decay_params(model, decay_selected_params=cast(Any, 1))


@unittest.skipUnless(MUON_AVAILABLE, "torch.optim.Muon is not available")
class MuonAdamWOptimizerTest(unittest.TestCase):
    def test_composite_optimizer_plans_parameters_in_one_traversal(self):
        class CountingModel(TinyLLM):
            named_modules_calls = 0

            def named_modules(self, *args, **kwargs):
                self.named_modules_calls += 1
                return super().named_modules(*args, **kwargs)

        model = CountingModel()

        create_muon_adamw_optimizer(
            model,
            muon=_muon_options(3e-4),
            adamw=_adamw_options(3e-4),
        )

        self.assertEqual(model.named_modules_calls, 1)

    def test_create_muon_adamw_optimizer_returns_composite_optimizer(self):
        model = TinyLLM()

        optimizer = create_muon_adamw_optimizer(
            model,
            muon=_muon_options(3e-4),
            adamw=_adamw_options(3e-4),
            excluded_modules=(model.lm_head,),
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        self.assertEqual(set(optimizer.optimizers), {"muon", "adamw"})
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(
            optimizer.optimizers["muon"].param_groups[0]["adjust_lr_fn"],
            MuonAdjustLRFn.MATCH_RMS_ADAMW,
        )
        self.assertEqual(_group_param_ids(optimizer), {id(param) for param in model.parameters()})
        self.assertNotIn(id(model.lm_head.weight), _group_param_ids(optimizer.optimizers["muon"]))
        self.assertIn(id(model.lm_head.weight), _group_param_ids(optimizer.optimizers["adamw"]))

    def test_create_muon_adamw_optimizer_uses_no_decay_for_adamw_fallback(self):
        model = TinyLLM()

        optimizer = create_muon_adamw_optimizer(
            model,
            muon=_muon_options(3e-4),
            adamw=_adamw_options(3e-4),
            excluded_modules=(model.lm_head,),
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        adamw_optimizer = optimizer.optimizers["adamw"]
        self.assertTrue(all(group["weight_decay"] == 0.0 for group in adamw_optimizer.param_groups))

    def test_create_muon_adamw_optimizer_requires_child_option_lrs(self):
        with self.assertRaisesRegex(KeyError, "lr"):
            create_muon_adamw_optimizer(TinyLLM(), muon=cast(Any, {}), adamw=_adamw_options(1.0))
        with self.assertRaisesRegex(KeyError, "lr"):
            create_muon_adamw_optimizer(TinyLLM(), muon=_muon_options(1.0), adamw=cast(Any, {}))

    def test_create_muon_adamw_optimizer_applies_lr_scale_rules_to_child_optimizers(self):
        model = TinyLLM()

        optimizer = create_muon_adamw_optimizer(
            model,
            muon=_muon_options(1.0),
            adamw=_adamw_options(1.0),
            excluded_modules=(model.lm_head,),
            lr_scale_rules=[
                {"name": "proj", "lr_scale": 0.5},
                {"name": "lm_head", "lr_scale": 2.0},
            ],
        )

        lrs = _group_lrs_by_param_id(optimizer)
        self.assertEqual(lrs[id(model.proj.weight)], 0.5)
        self.assertEqual(lrs[id(model.proj.bias)], 0.5)
        self.assertEqual(lrs[id(model.lm_head.weight)], 2.0)
        self.assertEqual(lrs[id(model.embed_tokens.weight)], 1.0)

    def test_composite_optimizer_scheduler_updates_child_param_groups(self):
        model = TinyLLM()
        optimizer = create_muon_adamw_optimizer(
            model,
            muon=_muon_options(1.0),
            adamw=_adamw_options(1.0),
            excluded_modules=(model.lm_head,),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)

        optimizer.step()
        scheduler.step()

        self.assertTrue(all(group["lr"] == 0.5 for group in optimizer.param_groups))
        self.assertTrue(
            all(group["lr"] == 0.5 for group in optimizer.optimizers["muon"].param_groups)
        )
        self.assertTrue(
            all(group["lr"] == 0.5 for group in optimizer.optimizers["adamw"].param_groups)
        )

    def test_composite_optimizer_state_dict_round_trip_uses_optimizer_names(self):
        model = TinyLLM()
        optimizer = create_muon_adamw_optimizer(
            model,
            muon=_muon_options(3e-4),
            adamw=_adamw_options(3e-4),
            excluded_modules=(model.lm_head,),
        )

        state = optimizer.state_dict()

        self.assertEqual(set(state["optimizers"]), {"muon", "adamw"})
        optimizer.load_state_dict(state)

        first_child_group = optimizer.optimizers["muon"].param_groups[0]
        self.assertIs(optimizer.param_groups[0], first_child_group)

    def test_composite_optimizer_restore_reorders_children_for_scheduler_state(self):
        first = nn.Parameter(torch.tensor(1.0))
        second = nn.Parameter(torch.tensor(2.0))
        optimizer = CompositeOptimizer(
            {
                "first": torch.optim.SGD([first], lr=1.0),
                "second": torch.optim.SGD([second], lr=2.0),
            }
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.5)
        scheduler.step()
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        restored_first = nn.Parameter(torch.tensor(1.0))
        restored_second = nn.Parameter(torch.tensor(2.0))
        restored = CompositeOptimizer(
            {
                "second": torch.optim.SGD([restored_second], lr=2.0),
                "first": torch.optim.SGD([restored_first], lr=1.0),
            }
        )
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored, lambda _: 0.5)
        restored.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)

        scheduler.step()
        restored_scheduler.step()

        self.assertEqual(restored.optimizers["first"].param_groups[0]["lr"], 0.5)
        self.assertEqual(restored.optimizers["second"].param_groups[0]["lr"], 1.0)
        self.assertEqual(
            [group["lr"] for group in restored.param_groups],
            [group["lr"] for group in optimizer.param_groups],
        )

    def test_composite_optimizer_rejects_mismatched_state_names(self):
        model = TinyLLM()
        optimizer = create_muon_adamw_optimizer(
            model,
            muon=_muon_options(3e-4),
            adamw=_adamw_options(3e-4),
            excluded_modules=(model.lm_head,),
        )

        with self.assertRaisesRegex(ValueError, "state names"):
            optimizer.load_state_dict({"optimizers": {"muon": {}}})

    def test_composite_optimizer_rejects_shared_child_parameters(self):
        model = nn.Linear(4, 4)
        optimizer_a = torch.optim.AdamW([model.weight], lr=1e-3)
        optimizer_b = torch.optim.SGD([model.weight], lr=1e-3)

        with self.assertRaisesRegex(ValueError, "share parameters"):
            CompositeOptimizer({"adamw": optimizer_a, "sgd": optimizer_b})

    def test_composite_optimizer_rejects_add_param_group_after_composition(self):
        model = nn.Linear(4, 4)
        optimizer = CompositeOptimizer({"adamw": torch.optim.AdamW([model.weight], lr=1e-3)})

        with self.assertRaisesRegex(RuntimeError, "add_param_group"):
            optimizer.add_param_group({"params": [model.bias]})


class LLMOptimizerTest(unittest.TestCase):
    def test_create_llm_optimizer_uses_sft_adamw_defaults(self):
        optimizer = create_llm_optimizer(TinyLLM(), preset="sft")

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["lr"], 2e-5)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.01)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))

    def test_create_llm_optimizer_uses_cpt_defaults(self):
        optimizer = create_llm_optimizer(TinyLLM(), preset="cpt")

        self.assertEqual(optimizer.defaults["lr"], 5e-5)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.01)

    def test_create_llm_optimizer_rejects_unknown_optimizer(self):
        with self.assertRaisesRegex(ValueError, "optimizer"):
            create_llm_optimizer(TinyLLM(), optimizer="sgd")

    def test_create_llm_optimizer_rejects_unknown_preset(self):
        with self.assertRaisesRegex(ValueError, "preset"):
            create_llm_optimizer(TinyLLM(), preset="dpo")

    def test_create_llm_optimizer_uses_flat_adamw_options(self):
        model = TinyLLM()
        optimizer = create_llm_optimizer(model, preset="sft", optimizer="adamw", lr=1e-4)

        self.assertIsInstance(optimizer, torch.optim.AdamW)

    @unittest.skipUnless(MUON_AVAILABLE, "torch.optim.Muon is not available")
    def test_create_llm_optimizer_uses_composite_for_muon_options(self):
        model = TinyLLM()
        optimizer = create_llm_optimizer(model, preset="pretrain", optimizer="muon")

        self.assertIsInstance(optimizer, CompositeOptimizer)

    @unittest.skipUnless(MUON_AVAILABLE, "torch.optim.Muon is not available")
    def test_create_llm_optimizer_passes_excluded_modules(self):
        model = TinyLLM()
        optimizer = create_llm_optimizer(
            model,
            preset="pretrain",
            optimizer="muon",
            excluded_modules=(model.lm_head,),
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        assert isinstance(optimizer, CompositeOptimizer)
        self.assertNotIn(id(model.lm_head.weight), _group_param_ids(optimizer.optimizers["muon"]))
        self.assertIn(id(model.lm_head.weight), _group_param_ids(optimizer.optimizers["adamw"]))

    def test_create_llm_optimizer_passes_lr_scale_rules(self):
        model = TinyLLM()
        optimizer = create_llm_optimizer(
            model,
            preset="sft",
            optimizer="adamw",
            lr=1.0,
            lr_scale_rules=[{"name": "lm_head", "lr_scale": 2.0}],
        )

        self.assertEqual(_group_lrs_by_param_id(optimizer)[id(model.lm_head.weight)], 2.0)

    @unittest.skipUnless(MUON_AVAILABLE, "torch.optim.Muon is not available")
    def test_create_llm_muon_optimizer_passes_lr_scale_rules_to_child_optimizers(self):
        model = TinyLLM()
        optimizer = create_llm_optimizer(
            model,
            preset="pretrain",
            optimizer="muon",
            lr=1.0,
            excluded_modules=(model.lm_head,),
            lr_scale_rules=[
                {"name": "proj", "lr_scale": 0.5},
                {"name": "lm_head", "lr_scale": 2.0},
            ],
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        lrs = _group_lrs_by_param_id(optimizer)
        self.assertEqual(lrs[id(model.proj.weight)], 0.5)
        self.assertEqual(lrs[id(model.proj.bias)], 0.5)
        self.assertEqual(lrs[id(model.lm_head.weight)], 2.0)

    @unittest.skipUnless(MUON_AVAILABLE, "torch.optim.Muon is not available")
    def test_composite_optimizer_scheduler_state_round_trip(self):
        model = TinyLLM()
        optimizer = create_muon_adamw_optimizer(
            model,
            muon=_muon_options(1.0),
            adamw=_adamw_options(1.0),
            excluded_modules=(model.lm_head,),
        )
        scheduler = create_scheduler(
            optimizer,
            schedule="warmup_cosine",
            warmup_steps=2,
            total_steps=6,
            min_lr_ratio=0.1,
        )

        for _ in range(3):
            optimizer.step()
            scheduler.step()

        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        expected_lrs = [group["lr"] for group in optimizer.param_groups]

        restored_model = TinyLLM()
        restored_optimizer = create_muon_adamw_optimizer(
            restored_model,
            muon=_muon_options(1.0),
            adamw=_adamw_options(1.0),
            excluded_modules=(restored_model.lm_head,),
        )
        restored_scheduler = create_scheduler(
            restored_optimizer,
            schedule="warmup_cosine",
            warmup_steps=2,
            total_steps=6,
            min_lr_ratio=0.1,
        )

        restored_optimizer.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)

        self.assertEqual(restored_scheduler.last_epoch, scheduler.last_epoch)
        self.assertEqual([group["lr"] for group in restored_optimizer.param_groups], expected_lrs)

        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()

        self.assertEqual(
            [group["lr"] for group in restored_optimizer.param_groups],
            [group["lr"] for group in optimizer.param_groups],
        )

    def test_create_scheduler_warmup_and_decay(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = create_scheduler(
            optimizer,
            schedule="warmup_cosine",
            warmup_steps=2,
            total_steps=10,
            min_lr_ratio=0.1,
        )

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)
        for _ in range(20):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_create_scheduler_supports_wsd_composition(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = create_scheduler(
            optimizer,
            schedule="wsd",
            warmup_steps=2,
            stable_steps=3,
            decay_steps=5,
            min_lr_ratio=0.1,
        )

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_create_scheduler_supports_constant(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = create_scheduler(optimizer, schedule="constant")

        for _ in range(20):
            optimizer.step()
            scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)

    def test_create_scheduler_validates_named_schedule_inputs(self):
        optimizer = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1.0)

        with self.assertRaisesRegex(ValueError, "schedule"):
            create_scheduler(optimizer, schedule="linear")
        with self.assertRaisesRegex(ValueError, "only valid"):
            create_scheduler(optimizer, schedule="constant", warmup_steps=1)
        with self.assertRaisesRegex(ValueError, "only valid for wsd"):
            create_scheduler(
                optimizer,
                schedule="warmup_cosine",
                total_steps=10,
                decay_steps=2,
            )
        with self.assertRaisesRegex(TypeError, "total_steps"):
            create_scheduler(
                optimizer,
                schedule="wsd",
                stable_steps=1,
                decay_steps=1,
                total_steps=cast(Any, 3.0),
            )
        with self.assertRaisesRegex(ValueError, "must equal"):
            create_scheduler(
                optimizer,
                schedule="wsd",
                warmup_steps=1,
                stable_steps=1,
                decay_steps=1,
                total_steps=4,
            )

    @unittest.skipUnless(MUON_AVAILABLE, "torch.optim.Muon is not available")
    def test_llm_muon_defaults_share_lr_but_split_weight_decay(self):
        model = TinyLLM()
        optimizer = create_llm_optimizer(
            model,
            preset="pretrain",
            optimizer="muon",
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        assert isinstance(optimizer, CompositeOptimizer)
        muon_group = optimizer.optimizers["muon"].param_groups[0]
        adamw_optimizer = optimizer.optimizers["adamw"]
        self.assertEqual(muon_group["lr"], 3e-4)
        self.assertEqual(muon_group["weight_decay"], 0.0)
        self.assertEqual(adamw_optimizer.defaults["weight_decay"], 0.01)
        self.assertEqual(muon_group["adjust_lr_fn"], MuonAdjustLRFn.MATCH_RMS_ADAMW)

    def test_llm_flat_optimizer_does_not_accept_muon_options(self):
        with self.assertRaisesRegex(TypeError, "muon_lr"):
            cast(Any, create_llm_optimizer)(
                TinyLLM(),
                preset="pretrain",
                optimizer="muon",
                muon_lr=1.0,
            )


class OptimConfigTest(unittest.TestCase):
    def test_muon_availability_matches_torch(self):
        self.assertEqual(MUON_AVAILABLE, getattr(torch.optim, "Muon", None) is not None)

    def test_muon_creation_reports_optional_dependency(self):
        with (
            patch.object(torch.optim, "Muon", None, create=True),
            self.assertRaisesRegex(RuntimeError, r"anytrain\[muon\].*torch>=2\.9"),
        ):
            create_muon_adamw_optimizer(
                TinyLLM(),
                muon=_muon_options(1e-4),
                adamw=_adamw_options(1e-4),
            )

    def test_top_level_optim_api_hides_config_classes(self):
        self.assertNotIn("AdamWConfig", optim_api.__all__)
        self.assertNotIn("AdamWOptions", optim_api.__all__)
        self.assertNotIn("Schedule", optim_api.__all__)
        self.assertFalse(hasattr(optim_api, "create_llm_lightning_optimizers"))

    def test_scheduler_api_does_not_export_legacy_aliases(self):
        self.assertEqual(scheduler_api.__all__, ["create_scheduler"])
        self.assertFalse(hasattr(scheduler_api, "Phase"))
        self.assertFalse(hasattr(scheduler_api, "Schedule"))
        self.assertFalse(hasattr(scheduler_api, "SchedulerConfig"))
        self.assertFalse(hasattr(scheduler_api, "SchedulerPhaseConfig"))
        self.assertFalse(hasattr(scheduler_api, "SchedulerPhaseLike"))

    def test_scheduler_api_hides_internal_helpers(self):
        self.assertNotIn("SchedulerOption", scheduler_api.__all__)
        self.assertNotIn("lr_ratio", scheduler_api.__all__)
        self.assertFalse(hasattr(scheduler_api, "normalize_curve_shape"))

    def test_top_level_optim_api_exports_lr_scale_rules(self):
        self.assertIn("LRScaleRule", optim_api.__all__)
        self.assertIn("LRScaleRules", optim_api.__all__)

    def test_adamw_options_delegate_invalid_betas_to_torch(self):
        with self.assertRaisesRegex(ValueError, "Invalid beta"):
            create_llm_optimizer(
                nn.Linear(2, 2),
                betas=(0.9, 1.0),
            )

    @unittest.skipUnless(MUON_AVAILABLE, "torch.optim.Muon is not available")
    def test_muon_options_default_to_match_rms_adamw(self):
        optimizer = create_muon_adamw_optimizer(
            TinyLLM(),
            muon=_muon_options(1e-4),
            adamw=_adamw_options(1e-4),
        )

        self.assertEqual(
            optimizer.optimizers["muon"].param_groups[0]["adjust_lr_fn"],
            MuonAdjustLRFn.MATCH_RMS_ADAMW,
        )

    @unittest.skipUnless(MUON_AVAILABLE, "torch.optim.Muon is not available")
    def test_muon_options_accept_adjust_lr_fn_string(self):
        optimizer = create_muon_adamw_optimizer(
            TinyLLM(),
            muon={**_muon_options(1e-4), "adjust_lr_fn": "original"},
            adamw=_adamw_options(1e-4),
        )

        self.assertEqual(optimizer.optimizers["muon"].param_groups[0]["adjust_lr_fn"], "original")

    @unittest.skipUnless(MUON_AVAILABLE, "torch.optim.Muon is not available")
    def test_muon_options_delegate_invalid_adjust_lr_fn_to_torch(self):
        with self.assertRaisesRegex(ValueError, "Adjust learning rate"):
            create_muon_adamw_optimizer(
                TinyLLM(),
                muon={**_muon_options(1e-4), "adjust_lr_fn": "scaled"},
                adamw=_adamw_options(1e-4),
            )


if __name__ == "__main__":
    unittest.main()
