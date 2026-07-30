from __future__ import annotations

INSTALL_HINT = (
    "Install the optional flow dependencies with `python -m pip install 'anytrain[flow]'`."
)

try:
    from flow_matching.loss import MixturePathGeneralizedKL
    from flow_matching.path import CondOTProbPath, MixtureDiscreteProbPath, ProbPath
    from flow_matching.path.path_sample import DiscretePathSample
    from flow_matching.path.scheduler import PolynomialConvexScheduler
    from flow_matching.solver import MixtureDiscreteEulerSolver, ODESolver
except ImportError as exc:  # pragma: no cover - exercised in environments without the extra.
    raise ImportError(
        f"`anytrain.framework.flow_matching` requires `flow_matching`. {INSTALL_HINT}"
    ) from exc


__all__ = [
    "CondOTProbPath",
    "DiscretePathSample",
    "MixtureDiscreteEulerSolver",
    "MixtureDiscreteProbPath",
    "MixturePathGeneralizedKL",
    "ODESolver",
    "PolynomialConvexScheduler",
    "ProbPath",
]
