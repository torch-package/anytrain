from typing import TYPE_CHECKING

from .condition import (
    ConditionGainKey,
    ConditionGainProbe,
    ConditionGainResult,
    ConditionGainStat,
    condition_gain,
    condition_gain_metrics,
)
from .data import (
    IGNORE_INDEX,
    Audio,
    AudioCondition,
    Content,
    EmbeddedAudioCondition,
    GenerationRequest,
    GenerationResponse,
    Modality,
    Route,
    Sample,
    TokenAudioCondition,
    TokenBatch,
    TokenInput,
    TokenTargets,
)
from .metrics import generation_metrics
from .model import (
    GenerationError,
    GenerationOptions,
    GenerationOutput,
    GenerationStatus,
    GenerationTiming,
    Generator,
    LossStat,
    StageTiming,
    TokenLossStats,
    TokenScorer,
    TokenStepInput,
    TokenStepOutput,
    TokenStepper,
    generate,
    score_tokens,
    step_tokens,
)
from .training import (
    REPORT_KEY,
    BatchUnits,
    RouteReport,
    StepReport,
    TrainingOutput,
    step_report,
    token_objective,
)

__all__ = [
    "IGNORE_INDEX",
    "REPORT_KEY",
    "Audio",
    "AudioCondition",
    "BatchUnits",
    "ConditionGainCallback",
    "ConditionGainKey",
    "ConditionGainProbe",
    "ConditionGainResult",
    "ConditionGainStat",
    "Content",
    "EmbeddedAudioCondition",
    "GenerationError",
    "GenerationOptions",
    "GenerationOutput",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationStatus",
    "GenerationTiming",
    "Generator",
    "LossStat",
    "Modality",
    "Route",
    "RouteReport",
    "S2SMetricsCallback",
    "Sample",
    "StageTiming",
    "StepReport",
    "TokenBatch",
    "TokenAudioCondition",
    "TokenInput",
    "TokenLossStats",
    "TokenScorer",
    "TokenStepInput",
    "TokenStepOutput",
    "TokenStepper",
    "TokenTargets",
    "TrainingOutput",
    "condition_gain",
    "condition_gain_metrics",
    "generate",
    "generation_metrics",
    "score_tokens",
    "step_tokens",
    "step_report",
    "token_objective",
]


if TYPE_CHECKING:
    from .integration import ConditionGainCallback as ConditionGainCallback
    from .integration import S2SMetricsCallback as S2SMetricsCallback


def __getattr__(name: str) -> object:
    if name == "ConditionGainCallback":
        from .integration import ConditionGainCallback
        return ConditionGainCallback
    if name == "S2SMetricsCallback":
        from .integration import S2SMetricsCallback
        return S2SMetricsCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
