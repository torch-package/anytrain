from __future__ import annotations

import hashlib
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from anytrain.lightning.checkpoint import (
    STATE_DICT_SCHEMA_GRAMMAR,
    Difference,
    StepCheckpoint,
    canonical_sha256,
    canonical_value,
    discover_step_checkpoints,
    evaluate_checkpoints,
    first_difference,
    load_checkpoint,
    stable_json,
    state_dict_schema,
)


class _Kind(Enum):
    FIRST = "first"


class _StateModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 3))
        self.register_buffer("offset", torch.ones(3))
        self.register_buffer("temporary", torch.ones(1), persistent=False)


class _ExtraStateModule(nn.Module):
    def get_extra_state(self) -> object:
        return {"version": 1}


class _HookModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.loaded_step: int | None = None

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        step = checkpoint.get("global_step")
        self.loaded_step = step if isinstance(step, int) else None


class CheckpointContractTest(unittest.TestCase):
    def test_canonical_value_normalizes_supported_contract_values(self) -> None:
        value = {
            "dtype": torch.float32,
            "enum": _Kind.FIRST,
            "sequence": (3, 2),
            "set": frozenset({"z", "a"}),
        }

        self.assertEqual(
            canonical_value(value),
            {
                "dtype": "float32",
                "enum": "first",
                "sequence": [3, 2],
                "set": ["a", "z"],
            },
        )

    def test_canonical_value_rejects_ambiguous_or_non_json_values(self) -> None:
        with self.assertRaisesRegex(TypeError, "mapping keys must be strings"):
            canonical_value(cast(Any, {1: "value"}))
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                canonical_value(value)
        with self.assertRaisesRegex(TypeError, "unsupported checkpoint contract value"):
            canonical_value(object())

    def test_stable_json_and_digest_are_order_independent(self) -> None:
        first = {"unicode": "\u00e9", "nested": {"b": 2, "a": 1}}
        second = {"nested": {"a": 1, "b": 2}, "unicode": "\u00e9"}

        serialized = stable_json(first)

        self.assertEqual(serialized, '{"nested":{"a":1,"b":2},"unicode":"\u00e9"}')
        self.assertEqual(stable_json(second), serialized)
        self.assertEqual(
            canonical_sha256(first),
            hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )

    def test_first_difference_reports_stable_paths_and_values(self) -> None:
        self.assertIsNone(first_difference({"a": [1]}, {"a": (1,)}))
        self.assertEqual(
            first_difference({"a": [1, 2]}, {"a": [1]}, path="contract"),
            Difference("contract.a.length", 2, 1),
        )
        self.assertEqual(
            first_difference({"a": 1}, {"a": 1, "b": 2}, path="contract"),
            Difference("contract.b", "<missing>", 2),
        )

    def test_first_difference_supports_domain_key_order(self) -> None:
        actual = {"alpha": 1, "runtime": 1}
        expected = {"alpha": 2, "runtime": 2}

        difference = first_difference(
            actual,
            expected,
            path="components",
            key_order={"components": ("runtime", "alpha")},
        )

        self.assertEqual(difference, Difference("components.runtime", 1, 2))

    def test_first_difference_requires_a_non_empty_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            first_difference({}, {}, path="")

    def test_state_dict_schema_hashes_sorted_parameter_and_buffer_entries(self) -> None:
        schema = state_dict_schema(_StateModule())
        entries = [
            {"name": "offset", "kind": "buffer", "shape": [3]},
            {"name": "weight", "kind": "parameter", "shape": [2, 3]},
        ]

        self.assertEqual(
            schema,
            {
                "grammar": STATE_DICT_SCHEMA_GRAMMAR,
                "entry_count": 2,
                "parameter_entries": 1,
                "buffer_entries": 1,
                "schema_sha256": canonical_sha256(entries),
            },
        )

    def test_state_dict_schema_rejects_non_tensor_extra_state(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a tensor"):
            state_dict_schema(_ExtraStateModule())

    def test_state_dict_schema_requires_a_module(self) -> None:
        with self.assertRaisesRegex(TypeError, "torch module"):
            state_dict_schema(cast(Any, object()))


class LightningCheckpointTest(unittest.TestCase):
    def test_loader_validates_control_arguments_before_reading(self) -> None:
        missing = Path("missing.ckpt")

        with self.assertRaisesRegex(TypeError, "torch.nn.Module"):
            load_checkpoint(cast(Any, object()), missing)
        with self.assertRaisesRegex(TypeError, "call_module_hook must be a bool"):
            load_checkpoint(
                nn.Linear(1, 1),
                missing,
                call_module_hook=cast(Any, 1),
            )

    def test_loader_rejects_malformed_root_and_state_dict(self) -> None:
        module = nn.Linear(1, 1)
        malformed = (
            (["not", "a", "mapping"], "root must be a mapping"),
            ({1: "not a string key"}, "root keys must be strings"),
            ({}, "state_dict must be a mapping"),
            ({"state_dict": []}, "state_dict must be a mapping"),
            ({"state_dict": {1: torch.zeros(())}}, "state_dict keys must be strings"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (payload, message) in enumerate(malformed):
                with self.subTest(message=message):
                    path = Path(directory) / f"malformed-{index}.ckpt"
                    torch.save(payload, path)
                    with self.assertRaisesRegex(TypeError, message):
                        load_checkpoint(module, path)

    def test_loader_calls_optional_hook_and_restores_state(self) -> None:
        source = _HookModule()
        source.weight.data.fill_(3)
        target = _HookModule()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save(
                {
                    "global_step": 7,
                    "state_dict": source.state_dict(),
                },
                path,
            )

            checkpoint = load_checkpoint(target, path)

        self.assertEqual(target.loaded_step, 7)
        self.assertEqual(target.weight.item(), 3)
        self.assertEqual(checkpoint["global_step"], 7)

    def test_loader_can_skip_module_hook(self) -> None:
        source = _HookModule()
        target = _HookModule()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save({"global_step": 7, "state_dict": source.state_dict()}, path)

            load_checkpoint(target, path, call_module_hook=False)

        self.assertIsNone(target.loaded_step)

    def test_loader_uses_strict_state_dict_matching(self) -> None:
        module = nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save({"state_dict": {}}, path)

            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_checkpoint(module, path)


class CheckpointEvaluationTest(unittest.TestCase):
    def test_discovers_durable_step_checkpoints_in_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "step-00000010.ckpt",
                "last.ckpt",
                ".step-00000020.ckpt.part",
                "step-00000002.ckpt",
                "notes.txt",
            ):
                (root / name).touch()

            checkpoints = discover_step_checkpoints(root)

        self.assertEqual([checkpoint.step for checkpoint in checkpoints], [2, 10])
        self.assertEqual(
            [checkpoint.name for checkpoint in checkpoints],
            ["step-00000002", "step-00000010"],
        )

    def test_rejects_duplicate_numeric_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "step-1.ckpt").touch()
            (root / "step-01.ckpt").touch()

            with self.assertRaisesRegex(ValueError, "multiple files for step 1"):
                discover_step_checkpoints(root)

    def test_requires_a_directory_with_step_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty"
            empty.mkdir()
            file = root / "checkpoint.ckpt"
            file.touch()

            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                discover_step_checkpoints(root / "missing")
            with self.assertRaisesRegex(NotADirectoryError, "not a directory"):
                discover_step_checkpoints(file)
            with self.assertRaisesRegex(FileNotFoundError, "contains no"):
                discover_step_checkpoints(empty)

    def test_evaluates_one_snapshot_in_step_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "step-20.ckpt").touch()
            (root / "step-3.ckpt").touch()
            evaluated: list[int] = []

            checkpoints = evaluate_checkpoints(
                root,
                lambda checkpoint: evaluated.append(checkpoint.step),
            )

        self.assertEqual(evaluated, [3, 20])
        self.assertEqual([checkpoint.step for checkpoint in checkpoints], [3, 20])

    def test_evaluation_is_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (1, 2, 3):
                (root / f"step-{step}.ckpt").touch()
            evaluated: list[int] = []

            def evaluate(checkpoint: StepCheckpoint) -> None:
                evaluated.append(checkpoint.step)
                if checkpoint.step == 2:
                    raise RuntimeError("failed")

            with self.assertRaisesRegex(RuntimeError, "failed"):
                evaluate_checkpoints(root, evaluate)

        self.assertEqual(evaluated, [1, 2])

    def test_evaluation_validates_callback_before_directory_access(self) -> None:
        with self.assertRaisesRegex(TypeError, "evaluate must be callable"):
            evaluate_checkpoints("missing", cast(Any, None))


if __name__ == "__main__":
    unittest.main()
