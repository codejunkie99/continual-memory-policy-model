"""Memory Policy Model (mpm).

A tiny policy model that learns WRITE / UPDATE / DELETE / LINK / COMPACT / NOOP
decisions for an external memory system from delayed downstream consequences,
while keeping user facts out of the model weights.
"""

from __future__ import annotations

from .types import Action, Op, ALL_OPS
from .store import MemoryStore
from .baseline import BaselinePolicy
from .mlx_policy import MLXPolicy, FakeMLXBackend, MlxBackend

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Op",
    "ALL_OPS",
    "MemoryStore",
    "BaselinePolicy",
    "MLXPolicy",
    "FakeMLXBackend",
    "MlxBackend",
    "__version__",
]
