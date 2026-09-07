"""
Time-Evolving Block Decimation: evolve an MPS under a nearest-neighbour
Hamiltonian by Trotter-splitting ``e^{-i H dt}`` (or ``e^{-H dtau}`` for
imaginary time) into even- and odd-bond two-site gates, applying each gate
followed by an SVD truncation back to bond dimension ``chi``.

Second-order (Strang) Trotter:  odd(dt/2) . even(dt) . odd(dt/2).
"""

import numpy as np
from scipy.linalg import expm

from mps import MPS, _svd
import models


def make_gates(bond_hamiltonians, dt, imaginary=False):
    """``U[i] = exp(-i h_i dt)`` (or ``exp(-h_i dtau)``) as ``(d,d,d,d)`` arrays
    with index order ``(out_i, out_{i+1}, in_i, in_{i+1})``."""
    coeff = -dt if imaginary else -1j * dt
    gates = []
    for h in bond_hamiltonians:
        d = h.shape[0]
        U = expm(coeff * h.reshape(d * d, d * d)).reshape(d, d, d, d)
        gates.append(U)
    return gates


def apply_gate(psi, i, U, chi_max, cutoff=1e-12):
    """Apply the two-site gate ``U`` to sites ``(i, i+1)`` and re-split."""
    psi.move_center_to(i, chi_max=chi_max)
    theta = np.tensordot(psi.M[i], psi.M[i + 1], axes=(2, 0))     # (a, s, s', b)
    theta = np.tensordot(U, theta, axes=([2, 3], [1, 2]))         # (t, t', a, b)
    theta = theta.transpose(2, 0, 1, 3)                           # (a, t, t', b)
    a, d1, d2, b = theta.shape
    A, s, Vh, disc = _svd(theta.reshape(a * d1, d2 * b), chi_max=chi_max, cutoff=cutoff)
    psi.M[i] = A.reshape(a, d1, -1)
    psi.M[i + 1] = (np.diag(s) @ Vh).reshape(-1, d2, b)
    nrm = np.linalg.norm(s)
    if nrm > 0:
        psi.M[i + 1] /= nrm
    psi.center = i + 1
    return disc


def sweep(psi, gates, chi_max, order="strang", cutoff=1e-12):
    """One time step. ``gates`` are the full-``dt`` gates; the Strang scheme
    internally uses half-``dt`` on the odd sublattice (pass ``gates_half`` via
    a 2-tuple ``(gates_full, gates_half)``)."""
    n = psi.n
    odd = list(range(1, n - 1, 2))
    even = list(range(0, n - 1, 2))
    disc = 0.0
    if order == "strang":
        gates_full, gates_half = gates
        for i in odd:
            disc = max(disc, apply_gate(psi, i, gates_half[i], chi_max, cutoff))
        for i in even:
            disc = max(disc, apply_gate(psi, i, gates_full[i], chi_max, cutoff))
        for i in odd:
            disc = max(disc, apply_gate(psi, i, gates_half[i], chi_max, cutoff))
    else:  # first order
        for i in even + odd:
            disc = max(disc, apply_gate(psi, i, gates[i], chi_max, cutoff))
    return disc


def evolve(psi, bond_hamiltonians, dt, n_steps, chi_max, imaginary=False,
           observables=None, record_every=1, cutoff=1e-12, verbose=False,
           renormalize_imag=True):
    """Propagate ``psi`` in place for ``n_steps`` of size ``dt``.

    ``observables``: dict ``name -> callable(psi) -> value`` sampled every
    ``record_every`` steps. Returns ``(times, records, max_discarded)``.
    """
    gates_full = make_gates(bond_hamiltonians, dt, imaginary)
    gates_half = make_gates(bond_hamiltonians, dt / 2, imaginary)
    times, records = [], {k: [] for k in (observables or {})}
    max_disc = 0.0
    for step in range(n_steps + 1):
        if step > 0:
            d = sweep(psi, (gates_full, gates_half), chi_max, cutoff=cutoff)
            max_disc = max(max_disc, d)
            if imaginary and renormalize_imag:
                psi.normalize()
        if step % record_every == 0:
            times.append(step * dt)
            for k, fn in (observables or {}).items():
                records[k].append(fn(psi))
            if verbose:
                extra = "  ".join(f"{k}={records[k][-1]:.6g}" for k in records)
                print(f"  t={step * dt:.3f}  chi={psi.max_bond():3d}  disc={max_disc:.1e}  {extra}")
    return np.array(times), {k: np.array(v) for k, v in records.items()}, max_disc


def ground_state_imag(model, n_sites, d, chi_max, dt_schedule=(0.1, 0.02, 0.005),
                      steps_per=200, seed=0, **model_kw):
    """Imaginary-time TEBD ground state -- an independent cross-check of DMRG."""
    if model == "tfim":
        cfg = [0] * n_sites
    elif model == "heisenberg":
        cfg = [i % d for i in range(n_sites)]        # Neel-ish start
    else:
        cfg = [1 if i % 2 == 0 else 2 for i in range(n_sites)]
    psi = MPS.product_state(cfg, d)
    psi.canonicalize(); psi.normalize()
    bond_h = models.two_site_hamiltonian(model, n_sites, **model_kw)
    for dt in dt_schedule:
        evolve(psi, bond_h, dt, steps_per, chi_max, imaginary=True)
    mpo = getattr(models, f"{model}_mpo")(n_sites, **_mpo_kw(model, model_kw))
    return psi, psi.expectation_mpo(mpo).real


def _mpo_kw(model, kw):
    if model == "tfim":
        return {"g": kw.get("g", 1.0), "J": kw.get("J", 1.0)}
    if model == "heisenberg":
        return {"S": kw.get("S", 0.5), "J": kw.get("J", 1.0),
                "Jz": kw.get("Jz", kw.get("J", 1.0)), "hz": kw.get("hz", 0.0)}
    return {"t": kw.get("t", 1.0), "U": kw.get("U", 1.0), "mu": kw.get("mu", 0.0)}
