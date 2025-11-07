"""Training helpers exposed under the :mod:`rag_t5.train` namespace."""

from .branching import (
    BranchComparisonConfig,
    BranchResult,
    BranchStrategy,
    run_three_branch_comparison,
)
from .trainer import TrainConfig, train

__all__ = [
    "TrainConfig",
    "train",
    "BranchStrategy",
    "BranchComparisonConfig",
    "BranchResult",
    "run_three_branch_comparison",
]
