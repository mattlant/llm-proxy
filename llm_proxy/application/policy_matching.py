from __future__ import annotations

from collections.abc import Callable, Collection
from fnmatch import fnmatchcase


def matches_any(
    source: Collection[str],
    predicates: Collection[str],
    comparator: Callable[[str, str], bool],
) -> bool:
    return bool(source) and any(comparator(value, predicate) for predicate in predicates for value in source)


def matches_all(
    source: Collection[str],
    predicates: Collection[str],
    comparator: Callable[[str, str], bool],
) -> bool:
    return not predicates or (bool(source) and all(any(comparator(value, predicate) for value in source) for predicate in predicates))


def equals_casefold(value: str, pattern: str) -> bool:
    return value.casefold() == pattern.casefold()


def equals_sensitive(value: str, pattern: str) -> bool:
    return value == pattern


def contains_casefold(value: str, pattern: str) -> bool:
    return pattern.casefold() in value.casefold()


def wildcard_casefold(value: str, pattern: str) -> bool:
    return fnmatchcase(value.casefold(), pattern.casefold())


def wildcard_sensitive(value: str, pattern: str) -> bool:
    return fnmatchcase(value, pattern)
