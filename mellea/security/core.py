"""Core security functionality for mellea.

This module provides the fundamental security classes and functions for
tracking security levels of content blocks and enforcing security policies.
"""

from collections.abc import Callable
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class Unclassified:
    """Indicates no taint."""



class Classified(Generic[T]):
    """Indicates taint with respect to a particular"""

    def __init__(self, entitlement_checker: Callable[[T], bool] | None, _meta: dict):
        self._entitlement_checker: Callable[[T], bool] = entitlement_checker
        self._meta = _meta

    def has_access(self, entitlement: T) -> bool:
        if self._entitlement_checker is None:
            raise SecurityError(
                "No entitlement checker was provided for this classification level."
            )
        return self._entitlement_checker(entitlement)


class TaintedBy:
    def __init__(self, source: "Taintable"):
        self.source = source


type SecLevel = Unclassified | Classified | TaintedBy


class SecurityError(Exception):
    """Exception raised for security-related errors."""


class Taintable(Protocol):
    def security_level(self) -> SecLevel:
        raise NotImplementedError("No security_level was specified.")
