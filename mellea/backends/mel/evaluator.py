"""Protocol for mel evaluators."""

from typing import Protocol, Tuple

from ...core import Context
from .ast import MelleaContent, MelProgram


class MelProgramEvaluator(Protocol):
    """Protocol for evaluation of mel programs"""

    def mel_eval(
        self, mexpr: MelProgram, ctx_init: Context, c_init: MelleaContent
    ) -> tuple[Context, MelleaContent]: ...
