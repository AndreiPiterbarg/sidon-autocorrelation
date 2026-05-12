"""Persistent box-state journal with content-addressed identity.

DESIGN
------
A box B in our pipeline is identified by the canonical hash of its
integer endpoints (lo_int, hi_int) — these are the same `D_SHIFT=60`
dyadic integer arrays used by the cascade BnB (`interval_bnb.box.Box`).
Two boxes with the same endpoints are the SAME box mathematically and
we deduplicate by hash. This lets us safely re-encounter the same box
across BnB iterations or SDP retries without double-counting volume.

The journal is an append-only JSONL file. Each line is one event:

    {"ts": "2026-05-02T12:34:56.789Z",
     "event": "box_seen|cascade_cert|sdp_cert|sdp_fail|split|infeasible|...",
     "hash": "...",                # 16-char hex prefix of sha256(lo_int||hi_int)
     "lo_int": [...],              # full int endpoints (for reconstruction)
     "hi_int": [...],
     "depth": 0,
     "volume": 1.23e-7,            # float volume for sanity
     "iter": 1,                    # which BnB+SDP iteration this came from
     "phase": "bnb|sdp|init|...",
     "extra": {...}                # method-specific fields
    }

The journal is read on startup to RESUME a partially-completed run:
boxes already CERT in the journal are skipped on re-injection.

INTEGRITY
---------
* Append-only (open mode 'a', flush after every line).
* Hash is canonical-hash on (sorted_lo_int_bytes || sorted_hi_int_bytes).
* Each line is self-describing — file is parseable even if truncated.
* Aggregate state (BoxJournal.summary()) is reconstructed from the
  full event log, never from cached counters that could drift.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Canonical box hash
# --------------------------------------------------------------------------

def canonical_box_hash(lo_int: Iterable[int], hi_int: Iterable[int]) -> str:
    """Return a stable hex hash of a box. The hash treats the box as the
    pair of integer arrays (lo_int, hi_int). Two boxes with element-wise
    equal int endpoints have the same hash; otherwise different.

    We use SHA-256 and take the first 16 hex chars (64 bits of entropy
    — enough to dedup ~10^9 boxes with negligible collision risk).
    """
    h = hashlib.sha256()
    # Length prefix prevents (a||b) == (a'||b') ambiguity.
    lo_list = list(int(x) for x in lo_int)
    hi_list = list(int(x) for x in hi_int)
    h.update(len(lo_list).to_bytes(4, 'little'))
    for v in lo_list:
        # int.to_bytes requires non-negative. Box endpoints in
        # interval_bnb are 0 ≤ v ≤ 2^60. We encode as 8-byte unsigned.
        h.update(int(v).to_bytes(8, 'little', signed=False))
    h.update(len(hi_list).to_bytes(4, 'little'))
    for v in hi_list:
        h.update(int(v).to_bytes(8, 'little', signed=False))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# Journal events
# --------------------------------------------------------------------------

# Event-name vocabulary. Add new events at the END of the list to keep
# old journals parseable.
EVENT_INIT_BOX        = 'init_box'         # initial-split box recorded
EVENT_CASCADE_CERT    = 'cascade_cert'     # closed by BnB cascade tier
EVENT_CASCADE_INFEAS  = 'cascade_infeas'   # box doesn't intersect simplex
EVENT_BOX_DUMPED      = 'box_dumped'       # box captured to dump file
EVENT_SDP_ATTEMPT     = 'sdp_attempt'      # SDP started on this box
EVENT_SDP_CERT        = 'sdp_cert'         # SDP returned infeas (cert)
EVENT_SDP_FAIL        = 'sdp_fail'         # SDP could not cert
EVENT_BOX_SPLIT       = 'box_split'        # parent split into 2 children
EVENT_ITER_START      = 'iter_start'       # iteration boundary
EVENT_ITER_END        = 'iter_end'
EVENT_RUN_START       = 'run_start'
EVENT_RUN_END         = 'run_end'
EVENT_AUDIT_VERDICT   = 'audit_verdict'    # final soundness check result


# --------------------------------------------------------------------------
# Journal class
# --------------------------------------------------------------------------

def _utc_iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='milliseconds').replace('+00:00', 'Z')


class BoxJournal:
    """Append-only JSONL log of box state transitions.

    Thread-safe: all writes go through one Lock so concurrent emits from
    SDP pool callbacks don't interleave bytes. JSONL line atomicity is
    guaranteed by the lock + immediate flush.

    Usage:
        j = BoxJournal('/path/to/run/journal.jsonl')
        j.emit(EVENT_INIT_BOX, lo_int=lo, hi_int=hi, depth=0, volume=v)
        j.emit(EVENT_CASCADE_CERT, hash='abcd...', extra={'method': 'autoconv'})
        j.close()

    Read existing journal:
        for ev in BoxJournal.iter_events('/path/to/journal.jsonl'):
            ...
    """

    def __init__(self, path: str | os.PathLike, *, run_tag: str = ''):
        self.path = str(Path(path))
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        self._fh = open(self.path, 'a', encoding='utf-8', buffering=1)
        self._lock = threading.Lock()
        self._run_tag = run_tag

    def emit(self, event: str, *,
             hash: Optional[str] = None,
             lo_int: Optional[Iterable[int]] = None,
             hi_int: Optional[Iterable[int]] = None,
             depth: Optional[int] = None,
             volume: Optional[float] = None,
             iter: Optional[int] = None,
             phase: Optional[str] = None,
             extra: Optional[Dict[str, Any]] = None) -> None:
        """Append one event. Hash is auto-computed if absent and lo/hi given."""
        if hash is None and lo_int is not None and hi_int is not None:
            hash = canonical_box_hash(lo_int, hi_int)
        rec: Dict[str, Any] = {
            'ts': _utc_iso_now(),
            'event': event,
        }
        if self._run_tag:
            rec['run'] = self._run_tag
        if hash is not None:
            rec['hash'] = hash
        if lo_int is not None:
            rec['lo_int'] = [int(x) for x in lo_int]
        if hi_int is not None:
            rec['hi_int'] = [int(x) for x in hi_int]
        if depth is not None:
            rec['depth'] = int(depth)
        if volume is not None:
            rec['volume'] = float(volume)
        if iter is not None:
            rec['iter'] = int(iter)
        if phase is not None:
            rec['phase'] = phase
        if extra:
            rec['extra'] = extra
        line = json.dumps(rec, separators=(',', ':')) + '\n'
        with self._lock:
            self._fh.write(line)
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass

    @staticmethod
    def iter_events(path: str | os.PathLike) -> Iterable[Dict[str, Any]]:
        """Yield each parsed event from a journal file. Skips malformed lines."""
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    @staticmethod
    def summary(path: str | os.PathLike) -> Dict[str, Any]:
        """Reconstruct aggregate state from the full event log.

        Returns:
            {
                'init_volume': float,            # sum of init_box volumes
                'cascade_cert_volume': float,
                'sdp_cert_volume': float,
                'cascade_infeas_volume': float,
                'sdp_fail_count': int,
                'pending_count': int,            # init_box - certs - infeas - splits
                'last_iter': int,
                'unique_boxes_seen': int,
            }
        """
        init_vol = 0.0
        cascade_cert_vol = 0.0
        sdp_cert_vol = 0.0
        cascade_infeas_vol = 0.0
        sdp_fail_count = 0
        last_iter = 0
        seen_init: set = set()
        seen_cascade_cert: set = set()
        seen_sdp_cert: set = set()
        seen_infeas: set = set()
        seen_split: set = set()
        all_seen: set = set()
        for ev in BoxJournal.iter_events(path):
            h = ev.get('hash')
            v = ev.get('volume', 0.0) or 0.0
            ename = ev.get('event')
            if h:
                all_seen.add(h)
            if ename == EVENT_INIT_BOX and h not in seen_init:
                seen_init.add(h)
                init_vol += v
            elif ename == EVENT_CASCADE_CERT and h not in seen_cascade_cert:
                seen_cascade_cert.add(h)
                cascade_cert_vol += v
            elif ename == EVENT_SDP_CERT and h not in seen_sdp_cert:
                seen_sdp_cert.add(h)
                sdp_cert_vol += v
            elif ename == EVENT_CASCADE_INFEAS and h not in seen_infeas:
                seen_infeas.add(h)
                cascade_infeas_vol += v
            elif ename == EVENT_SDP_FAIL:
                sdp_fail_count += 1
            elif ename == EVENT_BOX_SPLIT:
                seen_split.add(h)
            elif ename == EVENT_ITER_END:
                last_iter = max(last_iter, ev.get('iter', 0) or 0)
        # A box is "settled" if we've recorded a cert (cascade or SDP)
        # or marked it infeasible or split. Otherwise still pending.
        settled = seen_cascade_cert | seen_sdp_cert | seen_infeas | seen_split
        pending = seen_init - settled
        return {
            'init_volume': init_vol,
            'cascade_cert_volume': cascade_cert_vol,
            'sdp_cert_volume': sdp_cert_vol,
            'cascade_infeas_volume': cascade_infeas_vol,
            'sdp_fail_count': sdp_fail_count,
            'pending_count': len(pending),
            'pending_hashes': sorted(pending),
            'last_iter': last_iter,
            'unique_boxes_seen': len(all_seen),
            'init_count': len(seen_init),
            'cascade_cert_count': len(seen_cascade_cert),
            'sdp_cert_count': len(seen_sdp_cert),
            'split_count': len(seen_split),
            'infeas_count': len(seen_infeas),
        }
