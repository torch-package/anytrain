# RL Framework Design

## 目标

anytrain.framework.rl 提供偏好学习和离线 rollout 训练中可直接组合进下游
LightningModule / objective 的 tensor-level 组件。第一版面向 two-stage RLHF/RLAIF
工作流：

1. 下游项目离线生成推理数据，并对候选响应打 reward。
2. 训练时把 batch 转成 log-prob、reward 和 mask tensor，再调用 DPO / GRPO objective。

anytrain 不接管采样入口、reward 模型、JSONL schema、dataset/collate、batch schema 或
训练配置；这些仍由 anycodec、speech-to-speech 或其它下游项目拥有。

## Public API

根 public API：

    from anytrain.framework.rl import (
        ContinuousGRPOLoss,
        ContinuousRolloutBatch,
        DPOLoss,
        GRPOLoss,
        NeighborGRPOLoss,
        NeighborRolloutBatch,
        RolloutBatch,
        gather_token_logps,
        group_relative_advantages,
        neighbor_logps_from_distances,
        sequence_logps,
    )

anytrain.framework.__init__ 不默认导出 RL 子模块，用户需要显式导入
anytrain.framework.rl。

## Rollout tensor contracts

`RolloutBatch`、`ContinuousRolloutBatch` 和 `NeighborRolloutBatch` 只描述进入 loss
前的 tensor 字段，不描述下游项目的 rollout record、reward source、JSONL schema 或
collate 结果。它们用于给 adapter 的返回值加类型边界，而不是要求下游 batch 直接继承
anytrain 的数据结构。

- `RolloutBatch` 对应 token/logit GRPO：`policy_token_logps`、`old_token_logps`、
  `rewards`、`response_mask`，以及可选 `ref_token_logps`、`group_mask`。
- `ContinuousRolloutBatch` 对应连续 latent transition GRPO：
  `policy_step_logps`、`old_step_logps`、`rewards`、`step_mask`，以及可选
  `kl_values`、`group_mask`。
- `NeighborRolloutBatch` 对应 neighbor/symmetric flow GRPO：
  `policy_neighbor_logps`、`old_neighbor_logps`、`rewards`、`neighbor_mask`，
  以及可选 `kl_values`、`group_mask`、`anchor_mask`。

## Log-prob helpers

gather_token_logps(logits, token_ids) 接收：

- logits: [..., vocab]
- token_ids: [...]

返回选中 token 的 log-prob，形状与 token_ids 一致。

sequence_logps(token_logps, mask, reduction="sum") 把 token log-prob 聚合成
sequence log-prob。默认 sum 适配 DPO 的 response sequence log-prob；如果下游需要长度
归一化，需要显式传 reduction="mean"。

## DPO

DPOLoss 只接收 sequence log-prob，不负责重新 forward 模型：

    loss_fn = DPOLoss(beta=0.1)

    loss, details = loss_fn(
        policy_chosen_logps=policy_chosen,
        policy_rejected_logps=policy_rejected,
        ref_chosen_logps=ref_chosen,
        ref_rejected_logps=ref_rejected,
    )

公式：

    policy_margin = logpi(chosen|x) - logpi(rejected|x)
    ref_margin    = logpiref(chosen|x) - logpiref(rejected|x)
    loss          = -logsigmoid(beta * (policy_margin - ref_margin))

reference_free=True 时 reference margin 视为 0。reference log-prob 在 loss 内 detach，
只让 policy log-prob 回传梯度。

details：

- policy_margin
- reference_margin
- preference_logit
- accuracy

## GRPO

GRPOLoss 固定第一版 tensor contract：

    loss_fn = GRPOLoss(clip_range=0.2, kl_beta=0.04)

    loss, details = loss_fn(
        policy_token_logps=policy_logps,
        old_token_logps=old_logps,
        ref_token_logps=ref_logps,
        rewards=rewards,
        response_mask=mask,
        group_mask=group_mask,
    )

这里 B 是 prompt batch，G 是同一 prompt 下的候选数量，T 是 padded response
token 数。每个有效 group 至少需要两个候选，用于 group-relative reward normalization。

公式：

    advantage = (reward - group_mean) / (group_std + eps)
    ratio     = exp(policy_token_logp - old_token_logp)
    pg_loss   = -min(ratio * advantage, clipped_ratio * advantage)
    kl        = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1
    total     = pg_loss + kl_beta * kl

old_token_logps、ref_token_logps、rewards 和 computed advantages 在 loss 内 detach；
policy log-prob 保留梯度。

details：

- pg_loss
- kl_loss
- reward_mean
- reward_std
- advantage_mean
- clip_fraction
- token_count
- candidate_count

## Continuous / Flow-GRPO

`ContinuousGRPOLoss` 面向 Flow-GRPO 风格的连续 latent transition 训练：下游已完成
group rollout、reward 打分，并能提供每个 denoising / flow step 的 Gaussian log-prob。
它不负责采样、reward model、prompt schema 或重新 forward 模型，只接收 tensor：

    loss_fn = ContinuousGRPOLoss(clip_range=1e-4, adv_clip_max=5.0)

    loss, details = loss_fn(
        policy_step_logps=policy_logps,
        old_step_logps=old_logps,
        rewards=rewards,
        step_mask=step_mask,
        kl_values=kl_values,
        group_mask=group_mask,
    )

这里 `policy_step_logps` / `old_step_logps` / `step_mask` 形状为 `[batch, group, steps]`，
`rewards` 形状为 `[batch, group]`。默认 `trajectory_mean` 先按有效 step 聚合每条轨迹，
再在有效候选间平均；`step_mean` 则直接按所有有效 step 平均。

论文引用：Liu et al., “Flow-GRPO: Training Flow Matching Models via Online RL”,
arXiv:2505.05470。SDE transition log-prob 和 KL helper 放在
`anytrain.framework.flow_matching`，RL loss 只消费 log-prob / KL tensor。

## Neighbor / Symmetric Flow-GRPO

`NeighborGRPOLoss` 面向 Neighbor GRPO 风格的 deterministic flow / diffusion
训练：下游已经为同一 prompt 生成一组 candidate trajectories，并在若干 anchor timestep
上构造“anchor 选择哪个 candidate neighbor”的 surrogate log-prob。它不负责 anchor
sampling、ODE sampler、reward model、trajectory cache 或重新 forward 模型，只接收 tensor：

    loss_fn = NeighborGRPOLoss(clip_range=0.2, advantage_norm_p=1.0)

    loss, details = loss_fn(
        policy_neighbor_logps=policy_logps,
        old_neighbor_logps=old_logps,
        rewards=rewards,
        neighbor_mask=neighbor_mask,
        kl_values=kl_values,
        group_mask=group_mask,
        anchor_mask=anchor_mask,
    )

这里 `policy_neighbor_logps` / `old_neighbor_logps` / `neighbor_mask` 形状为
`[batch, anchors, steps, group]`，`rewards` 形状为 `[batch, group]`。最后一维的
`group` 对应同一 prompt 下的候选轨迹；`anchors` 和 `steps` 由下游 symmetric anchor
sampling 或其它策略决定。可选 `anchor_mask` 形状为 `[batch, anchors]`，用于过滤无效
或未采样的 anchor。

公式：

    advantage = group_relative(reward, norm_p=advantage_norm_p)
    ratio     = exp(policy_neighbor_logp - old_neighbor_logp)
    pg_loss   = -min(ratio * advantage, clipped_ratio * advantage)
    total     = pg_loss + kl_beta * kl

`neighbor_logps_from_distances(distances, mask, temperature=...)` 提供最小公共 helper：
它把非负 neighbor distance 转成 `log_softmax(-distance / temperature)`。下游可以用当前
policy 的 anchor / candidate latent 距离生成 `policy_neighbor_logps`，用 rollout 时保存的
旧距离或旧 logits 生成 `old_neighbor_logps`。

与 `ContinuousGRPOLoss` 的边界区别：

- `ContinuousGRPOLoss` 消费 SDE Gaussian transition log-prob，形状为 `[batch, group, steps]`。
- `NeighborGRPOLoss` 消费 distance-softmax surrogate log-prob，形状为
  `[batch, anchors, steps, group]`。
- 二者共享 group-relative reward、PPO-style clipping、mask reduction 和 optional KL 思路，
  但 policy 概念不同，因此保持为两个平级 objective。

## 离线数据边界

离线生成建议由下游项目实现。以 speech-to-speech 为例，下游已有 generation request/result
概念，可以在其上记录：

    sample_id
    group_id
    candidate_id
    prompt_ids
    response_ids
    response_mask
    old_token_logps
    reward
    reward_components
    decode_ok
    metadata

进入 anytrain RL objective 前，下游应完成：

- 把 prompt + response 重新 forward current policy 得到 policy_token_logps。
- 从生成阶段保存的值读取 old_token_logps。
- 可选 forward frozen reference model 得到 ref_token_logps。
- 把 task-specific reward 映射到 rewards。
- 生成 response_mask 和可选 group_mask。

如果使用 Neighbor GRPO，下游还应完成：

- 为同一 prompt 生成 grouped candidate trajectories。
- 采样 symmetric anchors / timesteps，并构造 neighbor distance 或 surrogate logits。
- 保存或重算 old_neighbor_logps，并生成 neighbor_mask。

## semantic-acoustic-codec / s2s 适配边界

强化学习能力按依赖方向分成三层：

```text
anytrain.framework.rl
  -> semantic-acoustic-codec.rl
      -> s2s.rl
```

`anytrain.framework.rl` 只保留 tensor-level objective 和最小 typed contract，不出现
semantic code、acoustic code、source speech、ASR、speaker 或任务 prompt 字段。

`semantic-acoustic-codec.rl` 拥有 semantic/acoustic codec rollout source。它可以定义
SAC 自己的 request、candidate、reward components 和 adapter，例如：

```python
class SACRLAdapter:
    def rollout(self, source_batch) -> list[SACCandidate]: ...
    def score(self, candidates) -> SACRewardBatch: ...
    def to_grpo_batch(self, candidates, rewards) -> RolloutBatch: ...
    def to_continuous_grpo_batch(
        self,
        candidates,
        rewards,
    ) -> ContinuousRolloutBatch: ...
    def to_neighbor_grpo_batch(self, candidates, rewards) -> NeighborRolloutBatch: ...
```

SAC adapter 可以知道 structured codec 的 semantic ids、acoustic ids/features、decode
audio、decode_ok、old log-prob / old distance、codec metadata 和 codec-specific reward。
它最终只把这些领域对象降维成 `anytrain.framework.rl` 的 tensor batch。

`s2s.rl` 依赖 SAC adapter，但拥有 s2s source 和 task reward。它可以在 SAC candidate
之上定义 source speech、generation request/result、target text/semantic intent、ASR
结果、speaker preservation、translation/semantic correctness 等字段，例如：

```python
class S2SRLAdapter:
    def __init__(self, sac_adapter: SACRLAdapter): ...
    def rollout(self, requests) -> list[S2SGenerationResult]: ...
    def score(self, results) -> S2SRewardBatch: ...
    def to_grpo_batch(self, results, rewards) -> RolloutBatch: ...
```

因此 s2s 可以复用 SAC 的 codec substrate 和 candidate construction，但 SAC 不理解 s2s
任务语义；二者也都不把领域 schema 上推到 anytrain。

## 非目标

第一版不做：

- 完整 rollout runner、dataset、collate 或 JSONL writer。
- reward model / evaluator / ASR / MT / audio quality 打分。
- 下游 batch schema、task schema 或 prompt 模板。
- LightningModule、Trainer、optimizer 或配置系统。
- online PPO / vLLM / distributed generation runtime。

这些能力如果需要复用，后续也应以显式 helper 形式进入 framework.rl，不能把下游任务语义
塞进 anytrain core。

## 测试策略

第一版覆盖：

- gather_token_logps 和 sequence_logps 的 shape / mask 行为。
- DPO reference-free、with-reference、梯度边界和错误路径。
- GRPO group advantage、clip、KL、mask、梯度边界和错误路径。
- Continuous / Neighbor GRPO 的 log-prob 语义、mask、KL、advantage 和梯度边界。
- anytrain.framework 根包保持空导出。
