"""Method registry.

A thin container that holds the set of methods to be evaluated in an
experiment.  The runner iterates over the registry rather than hard-coding
method-specific branches.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterator, List

from src.methods.base import Method


class MethodRegistry:
    """Ordered collection of scheduling methods."""

    def __init__(self) -> None:
        self._methods: OrderedDict[str, Method] = OrderedDict()

    def register(self, method: Method) -> None:
        if method.name in self._methods:
            raise ValueError(f"Method '{method.name}' is already registered.")
        self._methods[method.name] = method

    def __iter__(self) -> Iterator[Method]:
        return iter(self._methods.values())

    def __len__(self) -> int:
        return len(self._methods)

    @property
    def names(self) -> List[str]:
        return list(self._methods.keys())
