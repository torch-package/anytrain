"""Composable wrappers around Facebook flow_matching."""

from .continuous import ContinuousFlowRuntime
from .discrete import DiscreteFlowRuntime
from .objective import (
    ContinuousVelocityObjective,
    DiscreteGeneralizedKLObjective,
    masked_mse_velocity_loss,
    mse_velocity_loss,
)
from .sampler import DiscreteEulerSampler, ODESampler
from .source import GaussianSource, MaskTokenSource, UniformSource, UniformTokenSource
from .stochastic import (
    LogProbReduction,
    RectifiedFlowSDE,
    RectifiedFlowSDETransition,
    gaussian_kl_from_means,
    gaussian_log_prob,
)
from .time import DEFAULT_TIME_EPS, LogitNormalTimeSampler, UniformTimeSampler
from .types import (
    ContinuousTrainingSample,
    FlowLossFn,
    FlowSampleOutput,
    ModelCaller,
    ModelExtras,
    Source,
    TimeSampler,
    default_call_model,
)

__all__ = [
    "ContinuousFlowRuntime",
    "ContinuousTrainingSample",
    "ContinuousVelocityObjective",
    "DEFAULT_TIME_EPS",
    "DiscreteEulerSampler",
    "DiscreteFlowRuntime",
    "DiscreteGeneralizedKLObjective",
    "FlowSampleOutput",
    "FlowLossFn",
    "GaussianSource",
    "LogProbReduction",
    "MaskTokenSource",
    "ModelCaller",
    "ModelExtras",
    "ODESampler",
    "LogitNormalTimeSampler",
    "RectifiedFlowSDE",
    "RectifiedFlowSDETransition",
    "Source",
    "TimeSampler",
    "UniformSource",
    "UniformTimeSampler",
    "UniformTokenSource",
    "default_call_model",
    "gaussian_kl_from_means",
    "gaussian_log_prob",
    "masked_mse_velocity_loss",
    "mse_velocity_loss",
]
