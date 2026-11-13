"""Public abstention API.

Kept as a stable import location for the reliability component while the
implementation lives with the evaluation utilities.
"""

from ..evaluation.abstention import AbstentionEngine

__all__ = ["AbstentionEngine"]
