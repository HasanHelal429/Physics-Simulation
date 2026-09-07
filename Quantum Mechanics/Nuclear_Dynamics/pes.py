"""
Potential-energy curves for nuclear motion: load a tabulated Born-Oppenheimer
scan E(R) (as written by Diatomic_HF_solver / Molecular_DFT `scan_pes`), turn
it into a continuous potential via `TDSE_Solver.potentials.potential_from_samples`
(cubic spline inside the sampled range, Morse continuation outside), and pull
spectroscopic constants out of a computed vibrational ladder.

All energies in Hartree, all lengths in Bohr, all masses in electron masses,
hbar = m_e = 1.  Frequencies are reported both in Hartree and in cm^-1.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "TDSE_Solver"))

import potentials as tdse_pot  # noqa: E402

# --------------------------------------------------------------------------
HA_TO_CM = 219474.6313632          # 1 Hartree in cm^-1
HA_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903

# nuclear (not atomic) masses in electron masses -- the plan's convention
M_PROTON = 1836.15267343
M_DEUTERON = 3670.482967
M_TRITON = 5496.921535
M_ELECTRON = 1.0
# a few heavier nuclei for completeness (most-abundant isotope, nuclear mass)
NUCLEAR_MASS = {
    "H": M_PROTON, "D": M_DEUTERON, "T": M_TRITON,
    "He": 7294.2996, "Li": 12786.4, "C": 21868.7, "N": 25520.4,
    "O": 29148.9, "F": 34622.9, "Cl": 63512.0,
}


def reduced_mass(a, b):
    """mu = M_a M_b / (M_a + M_b). `a`, `b` are element/isotope keys
    (``"H"``, ``"D"``, ...) or numeric masses in electron masses."""
    Ma = a if np.isscalar(a) and not isinstance(a, str) else NUCLEAR_MASS[a]
    Mb = b if np.isscalar(b) and not isinstance(b, str) else NUCLEAR_MASS[b]
    return Ma * Mb / (Ma + Mb)


# --------------------------------------------------------------------------
def load_pes_csv(path):
    """Read a two-column ``R_bohr, E_ha`` CSV (``#`` comments) -> (R, E)."""
    data = np.loadtxt(path, delimiter=",", comments="#")
    R, E = data[:, 0], data[:, 1]
    order = np.argsort(R)
    return R[order], E[order]


def make_potential(grid, R_samples, E_samples, fill="morse", return_fit=True):
    """Spline+Morse a tabulated PES onto a 1D `grid` (see
    TDSE_Solver.potentials.potential_from_samples). Returns (V, fit_dict)."""
    return tdse_pot.potential_from_samples(grid, R_samples, E_samples,
                                           fill=fill, axis=0, return_fit=return_fit)


# --------------------------------------------------------------------------
# analytic Morse spectrum -- the Phase-1 closed-form reference
# --------------------------------------------------------------------------
def morse_omega(D_e, a, mu):
    """Harmonic frequency of a Morse well: omega_e = a sqrt(2 D_e / mu)."""
    return a * np.sqrt(2.0 * D_e / mu)


def morse_anharmonicity(a, mu):
    """omega_e x_e = a^2 / (2 mu) (independent of D_e)."""
    return a ** 2 / (2.0 * mu)


def morse_levels(D_e, a, mu, v_max=None, E_min=0.0):
    """Exact Morse vibrational energies (above the potential minimum E_min):

        E_v = omega_e (v+1/2) - omega_e x_e (v+1/2)^2 ,   0 <= v <= v_bound

    Returns the array of bound-level energies E_min + E_v."""
    we = morse_omega(D_e, a, mu)
    wexe = morse_anharmonicity(a, mu)
    v_bound = int(np.floor((we / wexe - 1.0) / 2.0))
    if v_max is not None:
        v_bound = min(v_bound, v_max)
    v = np.arange(0, v_bound + 1)
    return E_min + we * (v + 0.5) - wexe * (v + 0.5) ** 2


def morse_bound_count(D_e, a, mu):
    """Number of bound vibrational states of a Morse well:
    floor(lambda - 1/2) + 1 with lambda = sqrt(2 mu D_e)/a."""
    lam = np.sqrt(2.0 * mu * D_e) / a
    return int(np.floor(lam - 0.5)) + 1


# --------------------------------------------------------------------------
# spectroscopic constants from a computed vibrational ladder
# --------------------------------------------------------------------------
def spectroscopic_constants(levels, mu=None, r_expect_inv2=None, E_dissoc=None):
    """Extract ``omega_e``, ``omega_e x_e`` (and, if given, ``B_e``, ``D_0``,
    ``omega_e y_e``) from an ascending array of vibrational energies by fitting

        E_v = T_e + omega_e (v+1/2) - omega_e x_e (v+1/2)^2 + omega_e y_e (v+1/2)^3

    to the lowest ~6 levels (a Dunham expansion truncated at third order).

    ``r_expect_inv2``  <1/R^2> in the v=0 state -> B_e = <1/R^2> / (2 mu).
    ``E_dissoc``       energy of the dissociation asymptote -> D_0 = E_dissoc - E_0.
    Returns a dict; frequencies come with ``*_cm`` cm^-1 partners.
    """
    levels = np.asarray(levels, float)
    n = min(len(levels), 6)
    v = np.arange(n)
    x = v + 0.5
    A = np.vstack([np.ones_like(x), x, -x ** 2, x ** 3]).T
    coef, *_ = np.linalg.lstsq(A, levels[:n], rcond=None)
    T_e, we, wexe, weye = coef

    out = {
        "T_e": T_e, "omega_e": we, "omega_e_xe": wexe, "omega_e_ye": weye,
        "omega_e_cm": we * HA_TO_CM, "omega_e_xe_cm": wexe * HA_TO_CM,
        "E_0": float(levels[0]),
        "zero_point_energy": float(levels[0] - T_e),
        "n_bound": int(len(levels)),
    }
    if r_expect_inv2 is not None and mu is not None:
        B_e = r_expect_inv2 / (2.0 * mu)
        out["B_e"] = B_e
        out["B_e_cm"] = B_e * HA_TO_CM
    if E_dissoc is not None:
        out["D_0"] = float(E_dissoc - levels[0])
        out["D_0_eV"] = float((E_dissoc - levels[0]) * HA_TO_EV)
    return out


def level_spacings(levels):
    """Successive spacings ``Delta E_v = E_{v+1} - E_v`` -- their linear
    decrease with v is ``2 omega_e x_e``."""
    return np.diff(np.asarray(levels, float))
