"""The `mel` domain-specific language.

This package exposes the `mel` DSL.
"""

from .ast import MelProgram
from .evaluator import MelProgramEvaluator

__all__ = [MelProgramEvaluator, MelProgram]
