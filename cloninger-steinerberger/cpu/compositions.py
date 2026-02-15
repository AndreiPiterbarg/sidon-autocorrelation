"""Composition generators for branch-and-prune algorithm.

Generates non-negative integer vectors summing to S, with optional
canonical (reversal-symmetric) filtering.
"""
import numpy as np
import numba


@numba.njit(cache=True)
def _fill_batch_generic(buf, d, S, state, remaining, depth_arr):
    """Generic Numba composition generator for any d."""
    pos = 0
    batch_size = buf.shape[0]
    depth = depth_arr[0]

    while depth >= 0:
        if depth == d - 1:
            state[d - 1] = remaining[d - 1]
            for i in range(d):
                buf[pos, i] = state[i]
            pos += 1
            if pos == batch_size:
                depth -= 1
                if depth >= 0:
                    state[depth] += 1
                depth_arr[0] = depth
                return pos
            depth -= 1
            if depth >= 0:
                state[depth] += 1
        elif state[depth] <= remaining[depth]:
            remaining[depth + 1] = remaining[depth] - state[depth]
            depth += 1
            state[depth] = 0
        else:
            depth -= 1
            if depth >= 0:
                state[depth] += 1

    depth_arr[0] = -1
    return pos


@numba.njit(cache=True)
def _fill_batch_d4(buf, S, c0, c1, c2):
    """Fill buf with d=4 compositions starting from state (c0, c1, c2).

    Returns (rows_written, next_c0, next_c1, next_c2, done).
    """
    pos = 0
    batch_size = buf.shape[0]

    while c0 <= S:
        r0 = S - c0
        while c1 <= r0:
            r1 = r0 - c1
            while c2 <= r1:
                buf[pos, 0] = c0
                buf[pos, 1] = c1
                buf[pos, 2] = c2
                buf[pos, 3] = r1 - c2
                pos += 1
                if pos == batch_size:
                    c2 += 1
                    return pos, c0, c1, c2, False
                c2 += 1
            c1 += 1
            c2 = 0
        c0 += 1
        c1 = 0
        c2 = 0

    return pos, c0, c1, c2, True


@numba.njit(cache=True)
def _fill_batch_d6(buf, S, c0, c1, c2, c3, c4):
    """Fill buf with d=6 compositions starting from state (c0..c4).

    Returns (rows_written, c0, c1, c2, c3, c4, done).
    """
    pos = 0
    batch_size = buf.shape[0]

    while c0 <= S:
        r0 = S - c0
        while c1 <= r0:
            r1 = r0 - c1
            while c2 <= r1:
                r2 = r1 - c2
                while c3 <= r2:
                    r3 = r2 - c3
                    while c4 <= r3:
                        buf[pos, 0] = c0
                        buf[pos, 1] = c1
                        buf[pos, 2] = c2
                        buf[pos, 3] = c3
                        buf[pos, 4] = c4
                        buf[pos, 5] = r3 - c4
                        pos += 1
                        if pos == batch_size:
                            c4 += 1
                            return pos, c0, c1, c2, c3, c4, False
                        c4 += 1
                    c3 += 1
                    c4 = 0
                c2 += 1
                c3 = 0
                c4 = 0
            c1 += 1
            c2 = 0
            c3 = 0
            c4 = 0
        c0 += 1
        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 0

    return pos, c0, c1, c2, c3, c4, True


@numba.njit(cache=True)
def _fill_batch_d4_canonical(buf, S, c0, c1, c2):
    """Fill buf with CANONICAL d=4 compositions (b <= rev(b) lex).

    For (c0, c1, c2, c3) with c3 = S-c0-c1-c2:
      canonical iff c0 < c3, or (c0 == c3 and c1 <= c2).

    Loop-bound tightening:
      - c0 capped at S//2 (c0 > S/2 => c3 < c0 always)
      - c1 capped at S - 2*c0 (beyond => no canonical c2 exists)
      - c2 capped at S - 2*c0 - c1 (beyond => c0 > c3)
    """
    pos = 0
    batch_size = buf.shape[0]
    half_S = S // 2

    while c0 <= half_S:
        r0 = S - c0
        c1_max = r0 - c0  # S - 2*c0
        while c1 <= c1_max:
            r1 = r0 - c1
            c2_max = r1 - c0  # S - 2*c0 - c1; guaranteed >= 0
            while c2 <= c2_max:
                c3 = r1 - c2
                # At c2 == c2_max: c0 == c3, palindrome — canonical iff c1 <= c2
                if c0 == c3 and c1 > c2:
                    c2 += 1
                    continue
                buf[pos, 0] = c0
                buf[pos, 1] = c1
                buf[pos, 2] = c2
                buf[pos, 3] = c3
                pos += 1
                if pos == batch_size:
                    c2 += 1
                    return pos, c0, c1, c2, False
                c2 += 1
            c1 += 1
            c2 = 0
        c0 += 1
        c1 = 0
        c2 = 0

    return pos, c0, c1, c2, True


@numba.njit(cache=True)
def _fill_batch_d6_canonical(buf, S, c0, c1, c2, c3, c4):
    """Fill buf with CANONICAL d=6 compositions (b <= rev(b) lex).

    For (c0,..,c5) with c5 = S-c0-c1-c2-c3-c4:
      canonical iff c0 < c5, or (c0==c5 and c1 < c4),
                    or (c0==c5 and c1==c4 and c2 <= c3).

    Loop-bound tightening on c0 and c4.
    """
    pos = 0
    batch_size = buf.shape[0]
    half_S = S // 2

    while c0 <= half_S:
        r0 = S - c0
        while c1 <= r0:
            r1 = r0 - c1
            while c2 <= r1:
                r2 = r1 - c2
                while c3 <= r2:
                    r3 = r2 - c3
                    # c5 = r3 - c4; canonical needs c0 <= c5, i.e. c4 <= r3 - c0
                    c4_max = r3 - c0
                    while c4 <= c4_max:
                        c5 = r3 - c4
                        # At c0 == c5: check (c1,c2) vs (c4,c3)
                        if c0 == c5:
                            if c1 > c4:
                                c4 += 1
                                continue
                            if c1 == c4 and c2 > c3:
                                c4 += 1
                                continue
                        buf[pos, 0] = c0
                        buf[pos, 1] = c1
                        buf[pos, 2] = c2
                        buf[pos, 3] = c3
                        buf[pos, 4] = c4
                        buf[pos, 5] = c5
                        pos += 1
                        if pos == batch_size:
                            c4 += 1
                            return pos, c0, c1, c2, c3, c4, False
                        c4 += 1
                    c3 += 1
                    c4 = 0
                c2 += 1
                c3 = 0
                c4 = 0
            c1 += 1
            c2 = 0
            c3 = 0
            c4 = 0
        c0 += 1
        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 0

    return pos, c0, c1, c2, c3, c4, True


@numba.njit(cache=True)
def _fill_batch_generic_canonical(buf, d, S, state, remaining, depth_arr):
    """Generic Numba canonical composition generator — only emits b <= rev(b)."""
    pos = 0
    batch_size = buf.shape[0]
    depth = depth_arr[0]

    while depth >= 0:
        if depth == d - 1:
            state[d - 1] = remaining[d - 1]
            # Canonical check: state <= state[::-1] lex
            is_canon = True
            for i in range(d // 2):
                j = d - 1 - i
                if state[i] < state[j]:
                    break
                elif state[i] > state[j]:
                    is_canon = False
                    break
            if not is_canon:
                depth -= 1
                if depth >= 0:
                    state[depth] += 1
                continue
            for i in range(d):
                buf[pos, i] = state[i]
            pos += 1
            if pos == batch_size:
                depth -= 1
                if depth >= 0:
                    state[depth] += 1
                depth_arr[0] = depth
                return pos
            depth -= 1
            if depth >= 0:
                state[depth] += 1
        elif state[depth] <= remaining[depth]:
            remaining[depth + 1] = remaining[depth] - state[depth]
            depth += 1
            state[depth] = 0
        else:
            depth -= 1
            if depth >= 0:
                state[depth] += 1

    depth_arr[0] = -1
    return pos


def generate_canonical_compositions_batched(d, S, batch_size=100000):
    """Yield batches of CANONICAL compositions (b <= rev(b) lex) as int32 arrays.

    Only generates ~half the compositions, skipping non-canonical ones.
    For d=4 and d=6, uses loop-bound-tightened Numba generators.
    """
    if d == 1:
        yield np.array([[S]], dtype=np.int32)
        return
    if d == 2:
        # Canonical: c0 <= c1, i.e. c0 <= S/2
        c0 = np.arange(S // 2 + 1, dtype=np.int32)
        yield np.column_stack([c0, S - c0])
        return

    if d == 4:
        c0, c1, c2, done = 0, 0, 0, False
        while not done:
            buf = np.empty((batch_size, d), dtype=np.int32)
            n, c0, c1, c2, done = _fill_batch_d4_canonical(
                buf, S, c0, c1, c2)
            if n > 0:
                yield buf[:n]
        return

    if d == 6:
        c0, c1, c2, c3, c4, done = 0, 0, 0, 0, 0, False
        while not done:
            buf = np.empty((batch_size, d), dtype=np.int32)
            n, c0, c1, c2, c3, c4, done = _fill_batch_d6_canonical(
                buf, S, c0, c1, c2, c3, c4)
            if n > 0:
                yield buf[:n]
        return

    # Generic path
    state = np.zeros(d, dtype=np.int32)
    remaining = np.zeros(d, dtype=np.int32)
    remaining[0] = S
    depth_arr = np.array([0], dtype=np.int32)
    while depth_arr[0] >= 0:
        buf = np.empty((batch_size, d), dtype=np.int32)
        n = _fill_batch_generic_canonical(buf, d, S, state, remaining,
                                          depth_arr)
        if n > 0:
            yield buf[:n]


def generate_compositions_batched(d, S, batch_size=100000):
    """Yield batches of compositions as numpy int32 arrays.

    Streaming generator: fills a fixed-size buffer and yields it when full.
    Peak memory is O(batch_size) instead of O(N_total).

    For d=4 and d=6, uses specialized Numba JIT with explicit nested loops.
    For all other d, uses a generic Numba JIT stack-based generator.
    """
    if d == 1:
        yield np.array([[S]], dtype=np.int32)
        return
    if d == 2:
        c0 = np.arange(S + 1, dtype=np.int32)
        yield np.column_stack([c0, S - c0])
        return

    if d == 4:
        c0, c1, c2, done = 0, 0, 0, False
        while not done:
            buf = np.empty((batch_size, d), dtype=np.int32)
            n, c0, c1, c2, done = _fill_batch_d4(buf, S, c0, c1, c2)
            if n > 0:
                yield buf[:n]
        return

    if d == 6:
        c0, c1, c2, c3, c4, done = 0, 0, 0, 0, 0, False
        while not done:
            buf = np.empty((batch_size, d), dtype=np.int32)
            n, c0, c1, c2, c3, c4, done = _fill_batch_d6(
                buf, S, c0, c1, c2, c3, c4)
            if n > 0:
                yield buf[:n]
        return

    # Generic path: Numba JIT stack-based generator (d=3,5,7+)
    state = np.zeros(d, dtype=np.int32)
    remaining = np.zeros(d, dtype=np.int32)
    remaining[0] = S
    depth_arr = np.array([0], dtype=np.int32)
    while depth_arr[0] >= 0:
        buf = np.empty((batch_size, d), dtype=np.int32)
        n = _fill_batch_generic(buf, d, S, state, remaining, depth_arr)
        if n > 0:
            yield buf[:n]
