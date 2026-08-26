from __future__ import annotations


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def round_down(x: int, y: int) -> int:
    return (x // y) * y
