import copy

from mellea.security.core import Classified, Taintable, TaintedBy, Unclassified
from mellea.stdlib.base import CBlock, Component, ModelOutputThunk


def declassify(
    c: ModelOutputThunk | CBlock | Component,
) -> ModelOutputThunk | CBlock | Component:
    """Create a declassified version of a CBlock (non-mutating).

    This function creates a new CBlock with the same content but marked
    as safe (SecLevel.none()). The original CBlock is not modified.

    Args:
        cblock: The CBlock to declassify

    Returns:
        A new CBlock with safe security level
    """
    match c:
        case ModelOutputThunk():
            c = copy.copy(c)
