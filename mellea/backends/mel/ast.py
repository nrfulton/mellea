"""The Mel Program ast.

# Definition of the Mel Program Syntax

## Values

ctx: a Mellea Context
c: a Mellea component or cblock or MOT
s: a span

We then need to define a function (ctx,c) -> s.

## Terms

h,j ::= x (var)
        c, ctx (component or cblock and context)

## Programs

p,q ::= x,y := ctx, c           assign
        fill(ctx, c)            fill
        gen(ctx, c)             generate
        p ; q                   sequential composition
        p || q                  parallel composition
        p ++ q                  non-det choice
        p^n                     repeat n times
        p*                      repeat non-det number of times.
        ?P()                    statically evaluated predcicate
        ?P(H)                   dynamically evaluated preciate H=h_0,...,h_n for finite n
        if P() then p else q    if/else with static predicates

## Predicates

P(h), Q(h) ::= h.value == str           where str is a static string
               h_0.value == h_1.value
               not P()
               P() and Q()
               P() or Q()
"""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

from ...core import CBlock, Component, Context, ModelOutputThunk

MelleaContent = Union[Component, CBlock, ModelOutputThunk]


@dataclass(frozen=True)
class MelNode:
    @abstractmethod
    def __wfe__(self) -> bool: ...


@dataclass(frozen=True)
class Prefill(MelNode):
    span: Span


@dataclass(frozen=True)
class Generate(MelNode):
    span: Span


@dataclass(frozen=True)
class SeqCompose(MelNode):
    left: MelNode
    right: MelNode

    def __wfe__(self) -> bool:
        return self.left.__wfe__() and self.right.__wfe__()


@dataclass(frozen=True)
class ParallelCompose(MelNode):
    left: MelNode
    right: MelNode

    def __wfe__(self) -> bool:
        return self.left.__wfe__() and self.right.__wfe__()


@dataclass(frozen=True)
class Choice(MelNode):
    left: MelNode
    right: MelNode

    def __wfe__(self) -> bool:
        return self.left.__wfe__() and self.right.__wfe__()


@dataclass(frozen=True)
class BoundedRepeat(MelNode):
    child: MelNode
    n: int

    def __wfe__(self) -> bool:
        return self.child.__wfe__() and self.n > 0


@dataclass(frozen=True)
class UnboundedRepeat(MelNode):
    child: MelNode

    def __wfe__(self) -> bool:
        return self.child.__wfe__()


@dataclass(frozen=True)
class Guard(MelNode):
    prop: Callable[[Context], bool]

    def __wfe__(self) -> bool:
        return True


MelProgram = Union[
    Prefill,
    Generate,
    SeqCompose,
    ParallelCompose,
    Choice,
    BoundedRepeat,
    UnboundedRepeat,
    Guard,
]
