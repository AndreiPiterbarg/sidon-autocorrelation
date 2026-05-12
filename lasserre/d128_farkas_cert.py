"""d=128 sparse-clique Farkas certification.

Thin wrapper around ``lasserre.d64_farkas_cert.certify_sparse_farkas``.
Writes the cert to ``lasserre/certs/d128_cert.json``. d=128 is corroborative
of the d=64 result; a failed d=128 run does not invalidate C_{1a} ≥ 1.281
provided the d=64 cert is in hand.
"""
from __future__ import annotations

import os
from pathlib import Path

from lasserre.d64_farkas_cert import (
    certify_sparse_farkas, bisect_certified_lb, SparseFarkasCertResult,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def certify_d128(t_test: float = 1.281, *,
                   order: int = 2,
                   bandwidth: int = 16,
                   cert_path: str | None = None,
                   **kwargs) -> SparseFarkasCertResult:
    if cert_path is None:
        cert_path = str(Path(_HERE) / 'certs' / 'd128_cert.json')
    return certify_sparse_farkas(
        d=128, order=order, bandwidth=bandwidth, t_test=t_test,
        cert_path=cert_path, **kwargs,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--order', type=int, default=2)
    ap.add_argument('--bandwidth', type=int, default=16)
    ap.add_argument('--t_test', type=float, default=1.281)
    ap.add_argument('--bisect', action='store_true')
    ap.add_argument('--t_hi', type=float, default=1.30)
    ap.add_argument('--n_threads', type=int, default=0)
    ap.add_argument('--cert_path', type=str, default=None)
    args = ap.parse_args()

    if args.bisect:
        r = bisect_certified_lb(
            d=128, order=args.order, bandwidth=args.bandwidth,
            t_lo=args.t_test, t_hi=args.t_hi,
            n_threads=args.n_threads, cert_path=args.cert_path,
        )
    else:
        r = certify_d128(
            t_test=args.t_test, order=args.order, bandwidth=args.bandwidth,
            n_threads=args.n_threads, cert_path=args.cert_path,
        )
    print('\n=== final d=128 cert result ===')
    from dataclasses import asdict
    for k, v in asdict(r).items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
