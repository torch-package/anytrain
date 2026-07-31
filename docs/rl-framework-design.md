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
        DPOLoss,
        GRPOLoss,
        gather_token_logps,
        group_relative_advantages,
        sequence_logps,
    )

anytrain.framework.__init__ 不默认导出 RL 子模块，用户需要显式导入
anytrain.framework.rl。

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
- anytrain.framework 根包保持空导出。
