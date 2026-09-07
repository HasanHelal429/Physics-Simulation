"""Regenerate the H2 (and optionally other) LDA Born-Oppenheimer curve used by
validate.py. Runs the Diatomic_HF_solver SCF on a coarse prolate grid at a
spread of R and writes Diatomic_HF_solver/media/pes_Z1_Z1_lda.csv.

    python tools/make_pes.py            # H2, ~5 min
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIA = os.path.join(HERE, "..", "..", "Diatomic_HF_solver")
sys.path.insert(0, DIA)

import diatomic_driver as dd
import prolate_coords as pc


def main(Z_A=1, Z_B=1):
    mu = pc.mu_grid(mu_max=20, N=120, s_min=1e-3)
    nu = pc.nu_grid(N=66)
    Rs = np.concatenate([np.arange(0.85, 2.4, 0.13),
                         np.arange(2.4, 3.7, 0.25),
                         np.array([3.7, 4.0])])
    E = []
    t0 = time.time()
    for R in Rs:
        r = dd.run_scf(Z_A, Z_B, float(R), method="lda", mu=mu, nu=nu,
                       max_iter=70, verbose=False)
        E.append(r["E_total"])
        print(f"  R={R:.2f}  E={r['E_total']:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    media = os.path.join(DIA, "media")
    os.makedirs(media, exist_ok=True)
    path = os.path.join(media, f"pes_Z{Z_A}_Z{Z_B}_lda.csv")
    np.savetxt(path, np.column_stack([Rs, E]), delimiter=",",
               header=f"R_bohr,E_total_ha   (Z_A={Z_A}, Z_B={Z_B}, method=lda, coarse prolate grid)",
               comments="# ")
    print("wrote", path)


if __name__ == "__main__":
    main()
