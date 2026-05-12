"""Pod entry point: emit N=1 p=2 Hoelder certificate + verify.

Writes ``certificates/holder_N1_p2.json`` and calls certify_holder.py on it.
"""
import os
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from flint import fmpq
    from delsarte_dual.grid_bound_holder.phi_holder import PhiHolderParams
    from delsarte_dual.grid_bound_holder.bisect_holder import (
        bisect_M_cert_holder, emit_certificate_holder,
    )
    from delsarte_dual.grid_bound_holder.certify_holder import verify_certificate_holder

    N = 1
    p = fmpq(2, 1)
    tol_q = fmpq(1, 10**4)
    out_path = "delsarte_dual/grid_bound_holder/certificates/holder_N1_p2.json"

    print(f"Emitting Hoelder cert at N={N}, p={p}, tol={tol_q}")
    params = PhiHolderParams.from_mv(
        N_max=N, p=p, n_cells_min_G=4096, J_tail=1024, prec_bits=256,
    )
    bound = bisect_M_cert_holder(
        params, N=N,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(1276, 1000),
        tol_q=tol_q,
        max_cells_per_M=2_000_000,
        filter_kwargs=dict(
            enable_F4_MO217=False,  # no-op at N=1
            enable_F7=True, enable_F8=True,
        ),
        prec_bits=256,
        verbose=True,
    )
    print(f"M_cert = {bound.M_cert_q} "
          f"(~{float(bound.M_cert_q.p)/float(bound.M_cert_q.q):.6f})")
    digest = emit_certificate_holder(bound, out_path)
    print(f"Cert written: {out_path}")
    print(f"SHA-256: {digest}")

    print()
    print("=" * 70)
    print("Independent verification")
    print("=" * 70)
    res = verify_certificate_holder(out_path, prec_bits=256)
    print(f"Verifier: {'ACCEPTED' if res.accepted else 'REJECTED'}")


if __name__ == "__main__":
    main()
