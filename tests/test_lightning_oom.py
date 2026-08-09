from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace

import torch

from anytrain.lightning import OOMDiagnostics, oom


class OOMDiagnosticsTest(unittest.TestCase):
    def test_callback_uses_downstream_input_reporter(self) -> None:
        callback = OOMDiagnostics(lambda batch: {"batch_size": len(batch)})
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))

        callback.on_train_batch_start(trainer, module, ("a", "b"), 7)
        callback.on_before_backward(trainer, module, torch.tensor(1.0))
        payload = _payload(callback, trainer, module, torch.OutOfMemoryError("CUDA OOM"))

        self.assertEqual(payload["context"]["phase"], "train_backward")
        self.assertEqual(payload["context"]["batch_idx"], 7)
        self.assertEqual(payload["context"]["inputs"], {"batch_size": 2})

    def test_explicit_report_replaces_active_batch_context(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))
        callback.on_train_batch_start(trainer, module, object(), 3)

        error = torch.OutOfMemoryError("generation allocation failed")
        self.assertTrue(
            oom.report_oom(
                trainer,
                module,
                error,
                phase="generation",
                inputs={"max_new_tokens": 128},
            )
        )
        payload = _payload(callback, trainer, module, error)

        self.assertEqual(payload["context"]["phase"], "generation")
        self.assertEqual(payload["context"]["inputs"], {"max_new_tokens": 128})

    def test_annotated_context_wins_over_callback_context(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))
        callback.on_train_batch_start(trainer, module, object(), 3)
        error = torch.OutOfMemoryError("annotated")
        self.assertTrue(oom.annotate(error, phase="inner", inputs={"shape": [1, 2]}))
        self.assertEqual(
            oom.context(error),
            {"phase": "inner", "inputs": {"shape": [1, 2]}},
        )

        payload = _payload(callback, trainer, module, error)

        self.assertEqual(payload["context"], {"phase": "inner", "inputs": {"shape": [1, 2]}})

    def test_runtime_error_with_oom_message_is_reported(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))

        payload = _payload(callback, trainer, module, RuntimeError("CUDA out of memory"))

        self.assertEqual(payload["event"], "out_of_memory")

    def test_non_oom_is_ignored(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))
        output = io.StringIO()

        with redirect_stderr(output):
            callback.on_exception(trainer, module, RuntimeError("boom"))

        self.assertEqual(output.getvalue(), "")


def _trainer(callback: OOMDiagnostics) -> SimpleNamespace:
    return SimpleNamespace(
        callbacks=[callback],
        current_epoch=2,
        global_step=11,
        global_rank=1,
        local_rank=0,
    )


def _payload(
    callback: OOMDiagnostics,
    trainer: SimpleNamespace,
    module: SimpleNamespace,
    error: BaseException,
) -> dict[str, object]:
    output = io.StringIO()
    with redirect_stderr(output):
        callback.on_exception(trainer, module, error)
    lines = output.getvalue().splitlines()
    if len(lines) != 1:
        raise AssertionError(f"expected one diagnostic line, received {lines!r}")
    return json.loads(lines[0])


if __name__ == "__main__":
    unittest.main()
