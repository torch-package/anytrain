"""Regression tests for reward, reduction, and exact-count dtype boundaries."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from anytrain.evaluator.weighted import Metric, reduce
from anytrain.framework.flow_matching.objective import masked_mse_velocity_loss
from anytrain.framework.rl import ContinuousGRPOLoss, GRPOLoss, NeighborGRPOLoss
from anytrain.framework.rl.advantage import group_relative_advantages
from anytrain.framework.rl.logprob import masked_mean, sequence_logps
from anytrain.loss.frame import masked_frame_mse_loss


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64]
)
def test_equal_and_masked_rewards_have_finite_advantages(dtype: torch.dtype) -> None:
    rewards = torch.tensor([[1, 1, 99], [1, 3, 99]], dtype=dtype)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    if rewards.is_floating_point():
        rewards[:, -1] = float("nan")
    result = group_relative_advantages(rewards, mask)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert result.dtype == expected_dtype
    torch.testing.assert_close(result, torch.tensor([[0, 0, 0], [-1, 1, 0]], dtype=expected_dtype))


def _grpo(kind: str, policy: Tensor, rewards: Tensor, mask: Tensor, kl: Tensor) -> Tensor:
    if kind.startswith("token"):
        return GRPOLoss(
            kl_beta=0.1, token_reduction="token_mean" if kind.endswith("flat") else "sequence_mean"
        )(
            policy_token_logps=policy,
            old_token_logps=torch.zeros_like(policy),
            rewards=rewards,
            response_mask=mask,
            ref_token_logps=kl,
        )
    if kind.startswith("continuous"):
        return ContinuousGRPOLoss(
            kl_beta=0.1, step_reduction="step_mean" if kind.endswith("flat") else "trajectory_mean"
        )(
            policy_step_logps=policy,
            old_step_logps=torch.zeros_like(policy),
            rewards=rewards,
            step_mask=mask,
            kl_values=kl,
        )
    return NeighborGRPOLoss(
        kl_beta=0.1,
        advantage_norm_p=1.0 if kind.endswith("flat") else 2.0,
        neighbor_reduction="neighbor_mean" if kind.endswith("flat") else "anchor_step_mean",
    )(
        policy_neighbor_logps=policy,
        old_neighbor_logps=torch.zeros_like(policy),
        rewards=rewards,
        neighbor_mask=mask,
        kl_values=kl,
    )


_KINDS = [
    "token_flat",
    "token_group",
    "continuous_flat",
    "continuous_group",
    "neighbor_flat",
    "neighbor_group",
]


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64, torch.int64])
@pytest.mark.parametrize("equal", [False, True])
def test_grpo_rewards_and_long_reductions_match_fp64(
    kind: str, dtype: torch.dtype, equal: bool
) -> None:
    shape = (1, 1, 32769, 2) if kind.startswith("neighbor") else (1, 2, 32769)
    policy_dtype = torch.float32 if dtype == torch.int64 else dtype
    policy = torch.zeros(shape, dtype=policy_dtype, requires_grad=True)
    mask = torch.ones(shape, dtype=torch.bool)
    mask[..., -1] = kind.startswith("neighbor")
    # Different valid counts and invalid values exercise numerator/mask behavior.
    if kind.startswith("neighbor"):
        mask[:, :, -1, :] = False
    with torch.no_grad():
        policy[~mask] = float("nan")
    rewards = torch.ones(1, 2, dtype=dtype)
    if not equal:
        rewards[0, 1] = 3
    kl = torch.full(shape, 0.5, dtype=policy_dtype).masked_fill(~mask, float("inf"))
    actual = _grpo(kind, policy, rewards, mask, kl)
    reference_policy = policy.detach().double().requires_grad_()
    expected = _grpo(kind, reference_policy, rewards.double(), mask, kl.double())
    assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-3, atol=1e-7)
    actual.backward()
    expected.backward()
    assert policy.grad is not None and reference_policy.grad is not None
    assert bool(torch.isfinite(policy.grad).all())
    assert torch.count_nonzero(policy.grad[~mask]) == 0
    torch.testing.assert_close(policy.grad.double(), reference_policy.grad, rtol=0.02, atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_masked_mean_and_sequence_mean_accumulate_without_half_overflow(dtype: torch.dtype) -> None:
    value = torch.ones(1, 65537, dtype=dtype, requires_grad=True)
    mask = torch.ones_like(value, dtype=torch.bool)
    mask[:, -1] = False
    with torch.no_grad():
        value[:, -1] = float("nan")
    mean = masked_mean(value, mask)
    seq = sequence_logps(value, mask, reduction="mean")
    total = sequence_logps(value, mask)
    assert mean.item() == seq.item() == 1
    assert total.item() == 65536
    assert mean.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    (mean + seq.sum()).backward()
    assert value.grad is not None
    assert bool(torch.isfinite(value.grad).all())
    assert value.grad[0, -1] == 0
    torch.testing.assert_close(
        value.grad[0, 0].double(), torch.tensor(2 / 65536, dtype=torch.float64)
    )


@pytest.mark.parametrize("flow", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_masked_mse_preserves_loss_and_gradients_at_half_count_boundary(
    flow: bool, dtype: torch.dtype
) -> None:
    shape = (1, 257, 256) if flow else (1, 65537, 1)
    prediction = torch.full(shape, 0.5, dtype=dtype, requires_grad=True)
    target = torch.zeros(shape, dtype=dtype, requires_grad=True)
    mask = torch.ones(shape[:2], dtype=torch.bool)
    mask[:, -1] = False
    with torch.no_grad():
        prediction[:, -1] = float("nan")
        target[:, -1] = float("inf")
    loss = (
        masked_mse_velocity_loss(prediction, target, {"mask": mask})
        if flow
        else masked_frame_mse_loss(prediction, target, mask)
    )
    assert loss.item() == 0.25
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    loss.backward()
    assert prediction.grad is not None and target.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool(torch.isfinite(target.grad).all())
    assert torch.count_nonzero(prediction.grad[:, -1]) == 0
    assert torch.count_nonzero(target.grad[:, -1]) == 0
    torch.testing.assert_close(
        prediction.grad[0, 0, 0].double(), torch.tensor(1 / 65536, dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "dtype,size", [(torch.bfloat16, 257), (torch.float16, 65536), (torch.int64, 65536)]
)
def test_metric_exact_count_shared_cache_and_gradients(dtype: torch.dtype, size: int) -> None:
    values = torch.ones(size, requires_grad=True)
    weights = torch.ones(size, dtype=dtype)
    metrics = reduce({"first": Metric(values, weights), "second": Metric(values * 2, weights)})
    assert metrics["first"][1] == metrics["second"][1] == size
    assert metrics["first"][0].item() == 1
    assert metrics["second"][0].item() == 2
    metrics["first"][0].backward()
    assert values.grad is not None
    torch.testing.assert_close(values.grad, torch.full_like(values, 1 / size))


def test_metric_count_rejects_unsupported_total_before_integer_overflow() -> None:
    for weights in (
        torch.tensor([2**53], dtype=torch.int64),
        torch.tensor([2**62, 2**62], dtype=torch.int64),
    ):
        with pytest.raises(ValueError, match="2\\*\\*53"):
            Metric(torch.ones_like(weights, dtype=torch.float32), weights).count()
    assert (
        Metric(torch.ones(2), torch.tensor([2**53 - 2, 1], dtype=torch.int64)).count() == 2**53 - 1
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_equal_reward_group_backward_is_finite(dtype: torch.dtype) -> None:
    rewards = torch.ones(1, 2, dtype=dtype, requires_grad=True)
    group_relative_advantages(rewards).sum().backward()
    assert rewards.grad is not None
    assert bool(torch.isfinite(rewards.grad).all())
