"""Offline probes for review-2026-09-18.md; no models or network required.

Run with the desired checkout's src on PYTHONPATH. These print observations,
including the known failures; they are not a replacement for regression tests.
The Chat shutdown probe runs in a subprocess with a five-second timeout.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable

import torch

from anytrain.evaluator.weighted import Metric
from anytrain.framework.flow_matching.objective import masked_mse_velocity_loss
from anytrain.framework.rl import GRPOLoss, group_relative_advantages
from anytrain.framework.rl.logprob import masked_mean, sequence_logps
from anytrain.loss.frame import masked_frame_mse_loss
from anytrain.module.stream import (
    Align,
    AlignMerge,
    AlignMethod,
    AlignSpec,
    Features,
    Materializer,
    bind,
    compile_layout,
)


def observe(operation: Callable[[], object]) -> str:
    try:
        return str(operation())
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def grpo(dtype: torch.dtype) -> torch.Tensor:
    return GRPOLoss()(
        policy_token_logps=torch.zeros(1, 2, 2),
        old_token_logps=torch.zeros(1, 2, 2),
        response_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        rewards=torch.tensor([[1, 1]], dtype=dtype),
    )


def broadcast_padding() -> torch.Tensor:
    carrier = Features("carrier", torch.zeros(1, 2, 1), torch.tensor([2]))
    source = Features("source", torch.tensor([[[3.0], [float("nan")]]]), torch.tensor([1]))
    expression = Align(
        "aligned", "carrier", "source", AlignSpec(AlignMethod.BROADCAST, AlignMerge.ADD)
    )
    layout = compile_layout(expression, {"carrier": carrier.timeline, "source": source.timeline})
    return Materializer()(bind(layout, {"carrier": carrier, "source": source})).values


def chat_shutdown() -> str:
    # Constructor bypass avoids creating SDK clients or reading credentials.
    code = '''
import asyncio
import threading
from anytrain.chat import Chat
chat = object.__new__(Chat)
chat._lock = threading.Lock()
chat._lock.acquire()
threading.Timer(0.2, chat._lock.release).start()
async def waiter():
    async with chat._async_text_lock():
        pass
async def main():
    asyncio.create_task(waiter())
    await asyncio.sleep(0.01)
print("before asyncio.run", flush=True)
asyncio.run(main())
print("shutdown complete", flush=True)
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=5, check=False
        )
    except subprocess.TimeoutExpired as error:
        return f"timeout after 5s; child output: {error.stdout!r}"
    return f"exit={result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"


def main() -> None:
    results: dict[str, str] = {}
    for dtype in (torch.int64, torch.float16, torch.float32):
        results[f"equal_rewards_{dtype}"] = observe(
            lambda dtype=dtype: group_relative_advantages(torch.tensor([[1, 1]], dtype=dtype))
        )
        results[f"grpo_equal_rewards_{dtype}"] = observe(lambda dtype=dtype: grpo(dtype))
    results["fp16_masked_mean_expected_1"] = observe(
        lambda: masked_mean(torch.ones(65536, dtype=torch.float16), torch.ones(65536, dtype=torch.bool))
    )
    results["fp16_sequence_mean_expected_minus_1"] = observe(
        lambda: sequence_logps(
            torch.full((1, 65536), -1.0, dtype=torch.float16),
            torch.ones(1, 65536, dtype=torch.bool),
            reduction="mean",
        )
    )
    results["fp16_flow_mse_expected_025"] = observe(
        lambda: masked_mse_velocity_loss(
            torch.full((1, 256, 256), 0.5, dtype=torch.float16),
            torch.zeros(1, 256, 256, dtype=torch.float32),
            {"mask": torch.ones(1, 256, dtype=torch.bool)},
        )
    )
    results["fp16_frame_mse_expected_025"] = observe(
        lambda: masked_frame_mse_loss(
            torch.full((1, 65536, 1), 0.5, dtype=torch.float16),
            torch.zeros(1, 65536, 1, dtype=torch.float16),
            torch.ones(1, 65536, dtype=torch.bool),
        )
    )
    results["bf16_metric_expected_mean_1_count_257"] = observe(
        lambda: Metric(torch.ones(257), torch.ones(257, dtype=torch.bfloat16)).reduced()
    )
    results["fp16_metric_expected_mean_1_count_65536"] = observe(
        lambda: Metric(torch.ones(65536), torch.ones(65536, dtype=torch.float16)).reduced()
    )
    results["broadcast_padding_expected_3_3"] = observe(broadcast_padding)
    results["chat_shutdown"] = chat_shutdown()
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
