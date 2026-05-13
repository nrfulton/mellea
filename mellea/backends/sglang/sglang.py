from collections.abc import Coroutine, Sequence

from ...core import BaseModelSubclass, C, CBlock, Component, Context, ModelOutputThunk
from .. import FormatterBackend
from ..mel import MelProgramEvaluator, ast as mast


class SGLangBackend(FormatterBackend, MelProgramEvaluator): ...
