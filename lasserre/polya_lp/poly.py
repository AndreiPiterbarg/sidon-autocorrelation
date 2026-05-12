"""Monomial enumeration and binomial coefficient utilities."""
from __future__ import annotations
from math import comb
from typing import List, Tuple, Dict
import numpy as np


def enum_monomials_le(d: int, R: int) -> List[Tuple[int, ...]]:
    """All multi-indices alpha in N^d with |alpha| <= R, sorted lexicographically."""
    out: List[Tuple[int, ...]] = []
    cur: List[int] = [0] * d

    def rec(pos: int, remaining: int):
        if pos == d - 1:
            for v in range(remaining + 1):
                cur[pos] = v
                out.append(tuple(cur))
            cur[pos] = 0
            return
        for v in range(remaining + 1):
            cur[pos] = v
            rec(pos + 1, remaining - v)
        cur[pos] = 0

    rec(0, R)
    return out


def enum_monomials_eq(d: int, R: int) -> List[Tuple[int, ...]]:
    """All multi-indices alpha in N^d with |alpha| == R."""
    return [m for m in enum_monomials_le(d, R) if sum(m) == R]


def multinomial(beta: Tuple[int, ...]) -> int:
    """Multinomial coefficient (|beta| choose beta_1, beta_2, ...)."""
    n = sum(beta)
    out = 1
    rem = n
    for b in beta:
        out *= comb(rem, b)
        rem -= b
    return out


def index_map(monos: List[Tuple[int, ...]]) -> Dict[Tuple[int, ...], int]:
    return {m: i for i, m in enumerate(monos)}


def shift_minus(beta: Tuple[int, ...], j: int) -> Tuple[int, ...] | None:
    """Return beta - e_j if non-negative, else None."""
    if beta[j] == 0:
        return None
    out = list(beta)
    out[j] -= 1
    return tuple(out)


def shift_minus2(beta: Tuple[int, ...], i: int, j: int) -> Tuple[int, ...] | None:
    """Return beta - e_i - e_j if non-negative."""
    if i == j:
        if beta[i] < 2:
            return None
        out = list(beta)
        out[i] -= 2
        return tuple(out)
    if beta[i] == 0 or beta[j] == 0:
        return None
    out = list(beta)
    out[i] -= 1
    out[j] -= 1
    return tuple(out)
