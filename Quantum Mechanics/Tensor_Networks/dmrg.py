"""
Two-site Density Matrix Renormalization Group: variationally minimize
``<psi|H|psi>`` over MPS of bounded bond dimension by sweeping back and forth,
at each bond solving the effective two-site eigenproblem with Lanczos
(``scipy.sparse.linalg.eigsh``) warm-started from the current state.

Environments are cached and updated incrementally, so a half-sweep costs
``O(N chi^3 d^2 Dw)`` -- dominated by the local eigensolver, not re-contraction.

Excited states are handled by *deflation*: pass previously converged MPS in
``ortho`` and every local eigenproblem is solved in the subspace orthogonal to
them (a penalty ``w |phi><phi|`` folded into the effective Hamiltonian), which
is the standard "DMRG in the orthogonal sector" the plan calls for.
"""

import time

import numpy as np
import scipy.sparse.linalg as spla

from mps import MPS, _svd


# ---------------------------------------------------------------- environments
def _grow_left(L, A, W):
    """Absorb one site into the left MPO environment. ``L``: (braR, wL, ketR)."""
    T = np.tensordot(L, A, axes=(2, 0))            # (braL, wL, d, ketR)
    T = np.tensordot(T, W, axes=([1, 2], [0, 3]))  # (braL, ketR, wR, out)
    T = np.tensordot(A.conj(), T, axes=([0, 1], [0, 3]))  # (braR, ketR, wR)
    return T.transpose(0, 2, 1)                    # (braR, wR, ketR)


def _grow_right(R, B, W):
    """Absorb one site into the right MPO environment. ``R``: (braL, wR, ketL)."""
    T = np.tensordot(B, R, axes=(2, 2))            # (ketL, d_ket, braR, wR)
    T = np.tensordot(W, T, axes=([1, 3], [3, 1]))  # (wL, out, ketL, braR)
    T = np.tensordot(T, B.conj(), axes=([1, 3], [1, 2]))  # (wL, ketL, braL)
    return T.transpose(2, 0, 1)                    # (braL, wL, ketL)


def _ovl_grow_left(Lo, A_psi, A_phi):
    """Left overlap environment ``<phi|psi>``. ``Lo``: (psi_bond, phi_bond)."""
    T = np.tensordot(A_psi.conj(), Lo, axes=(0, 0))       # (d, psiR, phiL)
    T = np.tensordot(T, A_phi, axes=([0, 2], [1, 0]))     # (psiR, phiR)
    return T


def _ovl_grow_right(Ro, B_psi, B_phi):
    """Right overlap environment. ``Ro``: (psi_bond, phi_bond)."""
    T = np.tensordot(B_psi.conj(), Ro, axes=(2, 0))       # (psiL, d, phiR)
    T = np.tensordot(T, B_phi, axes=([1, 2], [1, 2]))     # (psiL, phiL)
    return T


class _EffTwoSite(spla.LinearOperator):
    """Effective two-site Hamiltonian (matrix-free) on ``theta`` of shape
    ``(chi_l, d, d, chi_r)``, optionally deflated against local vectors."""

    def __init__(self, L, W1, W2, R, deflate=None, penalty=0.0):
        self.L, self.W1, self.W2, self.R = L, W1, W2, R
        self.chi_l = L.shape[0]
        self.d1, self.d2 = W1.shape[2], W2.shape[2]
        self.chi_r = R.shape[0]
        self.deflate = deflate or []              # list of flattened unit vectors
        self.penalty = penalty
        dim = self.chi_l * self.d1 * self.d2 * self.chi_r
        super().__init__(dtype=complex, shape=(dim, dim))

    def _matvec(self, x):
        th = x.reshape(self.chi_l, self.d1, self.d2, self.chi_r)
        t = np.tensordot(self.L, th, axes=(2, 0))
        t = np.tensordot(t, self.W1, axes=([1, 2], [0, 3]))
        t = np.tensordot(t, self.W2, axes=([1, 3], [3, 0]))
        t = np.tensordot(t, self.R, axes=([1, 3], [2, 1]))
        out = t.reshape(-1)
        for phi in self.deflate:
            out = out + self.penalty * phi * np.vdot(phi, x)
        return out


class DMRGResult:
    def __init__(self, mps, energy, history, discarded, entropy_mid=None):
        self.mps = mps
        self.energy = energy
        self.history = history
        self.max_discarded = discarded
        self.entropy_mid = entropy_mid

    def __repr__(self):
        return f"DMRGResult(E={self.energy:.10f}, chi={self.mps.max_bond()}, disc={self.max_discarded:.2e})"


def run_dmrg(mpo, n_sites, d, chi_max, n_sweeps=24, tol=1e-10,
             chi_schedule=None, init=None, seed=0, verbose=True,
             eig_tol=0.0, time_budget=None, ortho=None, penalty=None):
    """Ground-state (or deflated excited-state) search.

    ``chi_schedule``  per-sweep bond-dimension ramp (falls back to ``chi_max``).
    ``ortho``         list of MPS to project out (deflation).
    ``penalty``       deflation weight; defaults to ``10*(|E_est|+1)``.
    ``time_budget``   soft wall-clock cap in seconds -- finishes the current
                      sweep, then returns what it has.
    """
    ortho = ortho or []
    psi = init.copy() if init is not None else MPS.random(n_sites, d, min(chi_max, 10), seed=seed)
    psi.canonicalize()
    psi.normalize()
    psi.move_center_to(0)
    if penalty is None:
        penalty = 10.0 * (abs(mpo_energy_estimate(mpo, psi)) + 1.0)

    # MPO environments
    R_env = [None] * (n_sites + 1)
    R_env[n_sites] = np.ones((1, 1, 1), dtype=complex)
    for i in range(n_sites - 1, 0, -1):
        R_env[i] = _grow_right(R_env[i + 1], psi.M[i], mpo[i])
    L_env = [None] * (n_sites + 1)
    L_env[0] = np.ones((1, 1, 1), dtype=complex)

    # overlap environments, one pair per ortho state
    Lo = [[np.ones((1, 1), dtype=complex)] + [None] * n_sites for _ in ortho]
    Ro = [[None] * n_sites + [np.ones((1, 1), dtype=complex)] for _ in ortho]
    for p, phi in enumerate(ortho):
        for i in range(n_sites - 1, 0, -1):
            Ro[p][i] = _ovl_grow_right(Ro[p][i + 1], psi.M[i], phi.M[i])

    history, max_disc = [], 0.0
    t0 = time.time()
    energy = np.inf

    for sweep in range(n_sweeps):
        chi = chi_max if not chi_schedule else chi_schedule[min(sweep, len(chi_schedule) - 1)]
        sweep_disc = 0.0
        for direction in ("right", "left"):
            bonds = range(n_sites - 1) if direction == "right" else range(n_sites - 2, -1, -1)
            for i in bonds:
                L, R = L_env[i], R_env[i + 2]
                W1, W2 = mpo[i], mpo[i + 1]
                theta0 = np.tensordot(psi.M[i], psi.M[i + 1], axes=(2, 0))

                deflate = []
                for p, phi in enumerate(ortho):
                    ploc = np.tensordot(Lo[p][i], phi.M[i], axes=(1, 0))      # (psiL, d, phiMid)
                    ploc = np.tensordot(ploc, phi.M[i + 1], axes=(2, 0))      # (psiL, d, d, phiR)
                    ploc = np.tensordot(ploc, Ro[p][i + 2], axes=(3, 1))      # (psiL, d, d, psiR)
                    v = ploc.reshape(-1)
                    nv = np.linalg.norm(v)
                    if nv > 1e-12:
                        deflate.append(v / nv)

                Heff = _EffTwoSite(L, W1, W2, R, deflate=deflate, penalty=penalty)
                if Heff.shape[0] <= 2:
                    dense = np.array([Heff.matvec(np.eye(Heff.shape[0])[:, j])
                                      for j in range(Heff.shape[0])]).T
                    w, v = np.linalg.eigh(0.5 * (dense + dense.conj().T))
                    energy, theta = float(w[0]), v[:, 0].reshape(theta0.shape)
                else:
                    # loosen the local solve on early sweeps, tighten near convergence
                    loc_tol = eig_tol or (1e-5 if sweep < 2 else 1e-9 if sweep < 5 else 0.0)
                    w, v = spla.eigsh(Heff, k=1, which="SA", v0=theta0.reshape(-1),
                                      tol=loc_tol, maxiter=3000)
                    energy, theta = float(w[0]), v[:, 0].reshape(theta0.shape)

                cl, d1, d2, cr = theta.shape
                U, s, Vh, disc = _svd(theta.reshape(cl * d1, d2 * cr), chi_max=chi)
                sweep_disc = max(sweep_disc, disc)
                if direction == "right":
                    psi.M[i] = U.reshape(cl, d1, -1)
                    psi.M[i + 1] = (np.diag(s) @ Vh).reshape(-1, d2, cr)
                    psi.center = i + 1
                    L_env[i + 1] = _grow_left(L_env[i], psi.M[i], mpo[i])
                    for p, phi in enumerate(ortho):
                        Lo[p][i + 1] = _ovl_grow_left(Lo[p][i], psi.M[i], phi.M[i])
                else:
                    psi.M[i] = (U @ np.diag(s)).reshape(cl, d1, -1)
                    psi.M[i + 1] = Vh.reshape(-1, d2, cr)
                    psi.center = i
                    R_env[i + 1] = _grow_right(R_env[i + 2], psi.M[i + 1], mpo[i + 1])
                    for p, phi in enumerate(ortho):
                        Ro[p][i + 1] = _ovl_grow_right(Ro[p][i + 2], psi.M[i + 1], phi.M[i + 1])

        history.append(energy)
        max_disc = max(max_disc, sweep_disc)
        dE = abs(history[-1] - history[-2]) if len(history) > 1 else np.inf
        if verbose:
            print(f"  sweep {sweep + 1:2d}  chi={chi:3d}  E={energy:.10f}  dE={dE:.2e}  "
                  f"disc={sweep_disc:.2e}  ({time.time() - t0:.1f}s)")
        past_ramp = (not chi_schedule) or sweep >= len(chi_schedule) - 1
        if dE < tol and sweep >= 2 and past_ramp:
            break
        if time_budget is not None and time.time() - t0 > time_budget:
            if verbose:
                print(f"  [time budget {time_budget}s reached -- returning early]")
            break

    psi.normalize()
    mid = psi.entanglement_entropy(n_sites // 2 - 1) if n_sites > 2 else 0.0
    return DMRGResult(psi, energy, history, max_disc, entropy_mid=mid)


def mpo_energy_estimate(mpo, psi):
    try:
        return psi.expectation_mpo(mpo).real
    except Exception:
        return 0.0


def excited_states(mpo, n_sites, d, chi_max, n_levels=2, **kw):
    """Lowest ``n_levels`` eigenstates by successive deflated DMRG runs.
    Returns a list of ``DMRGResult`` in ascending energy."""
    kw.setdefault("verbose", False)
    base_seed = kw.pop("seed", 0)
    results = []
    for level in range(n_levels):
        res = run_dmrg(mpo, n_sites, d, chi_max, ortho=[r.mps for r in results],
                       seed=base_seed + 7 * level, **kw)
        results.append(res)
    return results
