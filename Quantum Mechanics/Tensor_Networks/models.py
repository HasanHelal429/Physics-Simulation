"""
Lattice-model definitions for the MPS/DMRG/TEBD stack: local operators, the
matrix-product-operator (MPO) Hamiltonians as finite-state machines, the
nearest-neighbour two-site gates TEBD needs, and the analytic reference
energies each model is checked against.

Conventions
-----------
* Open boundary conditions throughout.
* ``TFIM`` is written in the **Pauli** convention
  ``H = -J sum Z_i Z_{i+1} - g J sum X_i`` so it matches the textbook
  free-fermion energy per site ``e0 = -(1/pi) int_0^pi sqrt(1+g^2-2 g cos k) dk``
  and the Lieb-Robinson velocity ``v = 2 min(g, 1) J``.
* ``Heisenberg`` (spin-1/2 and spin-1) uses **spin** operators, eigenvalues of
  ``Sz`` at ``+-1/2`` (or ``0, +-1``), ``H = J sum S_i . S_{i+1}``; the spin-1/2
  chain has the Bethe-ansatz energy per site ``J (1/4 - ln 2)``.
* ``Hubbard`` uses a Jordan-Wigner-transformed local Hilbert space of dimension
  4 (``|0>, |up>, |dn>, |up dn>``); the JW sign strings live inside the MPO and
  the gates, so the MPS never sees an explicit anticommutator.

An MPO is a list of rank-4 arrays ``W[i]`` with index order
``(left_bond, right_bond, phys_out, phys_in)``. The first tensor is a row
vector ``(1, Dw, d, d)``, the last a column vector ``(Dw, 1, d, d)``.
"""

import numpy as np

# --------------------------------------------------------------------------
# local operators
# --------------------------------------------------------------------------

# Pauli matrices (TFIM)
PAULI_I = np.eye(2)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]])
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]])


def spin_operators(S):
    """Spin-S operators ``(Sx, Sy, Sz, Sp, Sm, I)`` on the ``d = 2S+1``
    dimensional local space, ordered by descending ``m`` (``m = S, S-1, ..., -S``).
    ``Sp``/``Sm`` are the raising/lowering operators."""
    d = int(round(2 * S + 1))
    m = S - np.arange(d)                        # m_0 = +S, ... m_{d-1} = -S
    Sz = np.diag(m).astype(complex)
    # <m+1| S+ |m> = sqrt(S(S+1) - m(m+1))
    off = np.sqrt(S * (S + 1) - m[1:] * (m[1:] + 1))
    Sp = np.diag(off, k=1).astype(complex)
    Sm = Sp.conj().T
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)
    return Sx, Sy, Sz, Sp, Sm, np.eye(d, dtype=complex)


# Fermion operators in the JW local basis |0>, |up>, |dn>, |up dn> (Hubbard).
# c_up, c_dn already include the on-site JW factor so that {c_up, c_dn^dag} = 0.
def _hubbard_operators():
    I4 = np.eye(4)
    # ordering: 0=|0>, 1=|up>, 2=|dn>, 3=|up dn>
    c_up = np.zeros((4, 4)); c_up[0, 1] = 1.0; c_up[2, 3] = 1.0
    c_dn = np.zeros((4, 4)); c_dn[0, 2] = 1.0; c_dn[1, 3] = -1.0   # sign: dn sits "after" up
    n_up = c_up.T @ c_up
    n_dn = c_dn.T @ c_dn
    F = np.diag([1.0, -1.0, -1.0, 1.0])          # JW parity string (-1)^(n_up+n_dn)
    return I4, c_up, c_dn, n_up, n_dn, F


# --------------------------------------------------------------------------
# MPO builders (finite-state-machine construction)
# --------------------------------------------------------------------------

def _fsm_to_mpo(bulk, d, n_sites, left_row, right_col):
    """Turn one bulk operator-valued ``Dw x Dw`` matrix ``bulk[a][b] -> d x d``
    (Python nested list, ``None`` = zero block) into an ``n_sites``-long MPO,
    projecting the first/last tensors onto ``left_row`` / ``right_col``."""
    Dw = len(bulk)
    W_bulk = np.zeros((Dw, Dw, d, d), dtype=complex)
    for a in range(Dw):
        for b in range(Dw):
            if bulk[a][b] is not None:
                W_bulk[a, b] = bulk[a][b]
    mpo = []
    for i in range(n_sites):
        if i == 0:
            mpo.append(W_bulk[left_row:left_row + 1, :, :, :].copy())
        elif i == n_sites - 1:
            mpo.append(W_bulk[:, right_col:right_col + 1, :, :].copy())
        else:
            mpo.append(W_bulk.copy())
    return mpo


def tfim_mpo(n_sites, g, J=1.0):
    """Transverse-field Ising, Pauli convention ``H = -J sum Z Z - g J sum X``.

    FSM rows/cols: 0 = "done", 1 = "carrying a Z", 2 = "not started".
        [ I        0     0 ]
        [ Z        0     0 ]
        [ -gJ X   -J Z    I ]
    """
    I, X, Z = PAULI_I.astype(complex), PAULI_X.astype(complex), PAULI_Z.astype(complex)
    bulk = [
        [I,            None,      None],
        [Z,            None,      None],
        [-g * J * X,   -J * Z,    I],
    ]
    return _fsm_to_mpo(bulk, 2, n_sites, left_row=2, right_col=0)


def heisenberg_mpo(n_sites, S=0.5, J=1.0, Jz=None, hz=0.0):
    """XXZ Heisenberg ``H = J sum (Sx Sx + Sy Sy) + Jz sum Sz Sz - hz sum Sz``.
    ``Jz`` defaults to ``J`` (isotropic). Dw = 5.

        [ I                                        ]
        [ Sp                                       ]
        [ Sm                                       ]
        [ Sz                                       ]
        [ -hz Sz   (J/2) Sm   (J/2) Sp   Jz Sz   I ]
    """
    if Jz is None:
        Jz = J
    Sx, Sy, Sz, Sp, Sm, I = spin_operators(S)
    d = I.shape[0]
    Z = None
    bulk = [
        [I,          Z,           Z,           Z,        Z],
        [Sp,         Z,           Z,           Z,        Z],
        [Sm,         Z,           Z,           Z,        Z],
        [Sz,         Z,           Z,           Z,        Z],
        [-hz * Sz,   0.5 * J * Sm, 0.5 * J * Sp, Jz * Sz, I],
    ]
    return _fsm_to_mpo(bulk, d, n_sites, left_row=4, right_col=0)


def heisenberg_chain_mpo(spins, J=1.0, Jz=None):
    """Isotropic (or XXZ) Heisenberg MPO on a chain whose sites may carry
    *different* spins -- ``spins`` is a per-site list (e.g.
    ``[0.5, 1, 1, ..., 1, 0.5]`` for a spin-1 chain with spin-1/2 caps, the
    standard trick to bind the edge modes so the bulk Haldane gap shows up
    as a clean singlet-triplet splitting). FSM identical to
    ``heisenberg_mpo`` (Dw = 5); only the local operators vary per site."""
    if Jz is None:
        Jz = J
    n = len(spins)
    ops = [spin_operators(S) for S in spins]
    mpo = []
    for i, (Sx, Sy, Sz, Sp, Sm, I) in enumerate(ops):
        d = I.shape[0]
        W = np.zeros((5, 5, d, d), dtype=complex)
        W[0, 0] = I
        W[1, 0] = Sp
        W[2, 0] = Sm
        W[3, 0] = Sz
        W[4, 1] = 0.5 * J * Sm
        W[4, 2] = 0.5 * J * Sp
        W[4, 3] = Jz * Sz
        W[4, 4] = I
        if i == 0:
            mpo.append(W[4:5, :, :, :].copy())
        elif i == n - 1:
            mpo.append(W[:, 0:1, :, :].copy())
        else:
            mpo.append(W.copy())
    return mpo


def hubbard_mpo(n_sites, t=1.0, U=1.0, mu=0.0):
    """1D Hubbard, ``H = -t sum_sigma (c^dag_i c_{i+1} + h.c.)
    + U sum n_up n_dn - mu sum (n_up + n_dn)``, Jordan-Wigner local basis d = 4.
    Hopping terms carry the JW string ``F`` on the "in transit" bond. Dw = 6.

    rows/cols: 0 done | 1 c_up^dag in transit | 2 c_up in transit
               3 c_dn^dag in transit | 4 c_dn in transit | 5 not started
    """
    I4, c_up, c_dn, n_up, n_dn, F = _hubbard_operators()
    I4 = I4.astype(complex)
    cu, cd, F = c_up.astype(complex), c_dn.astype(complex), F.astype(complex)
    cud, cdd = cu.conj().T, cd.conj().T
    onsite = (U * (n_up @ n_dn) - mu * (n_up + n_dn)).astype(complex)
    Z = None
    # JW: c^dag_i c_{i+1} = (a^dag_i F_i)(a_{i+1}); h.c. = (F_i a_i)(a^dag_{i+1}).
    # left-site operator carries the string, right-site operator is bare.
    bulk = [
        [I4,        Z,          Z,          Z,          Z,          Z],
        [cu,        Z,          Z,          Z,          Z,          Z],   # finisher: bare a_up
        [cud,       Z,          Z,          Z,          Z,          Z],   # finisher: bare a_up^dag
        [cd,        Z,          Z,          Z,          Z,          Z],
        [cdd,       Z,          Z,          Z,          Z,          Z],
        [onsite,   -t * (cud @ F), -t * (F @ cu),
                   -t * (cdd @ F), -t * (F @ cd),  I4],
    ]
    return _fsm_to_mpo(bulk, 4, n_sites, left_row=5, right_col=0)


# --------------------------------------------------------------------------
# two-site gates for TEBD
# --------------------------------------------------------------------------

def two_site_hamiltonian(model, n_sites, **kw):
    """Return a list of ``(d^2 x d^2)`` bond Hamiltonians ``h_bond[i]`` acting on
    sites ``(i, i+1)``, such that ``sum_i h_bond[i] == H``. On-site terms are
    split half onto the left bond and half onto the right (the two end sites get
    their full on-site term on their single bond). Index order of the reshaped
    ``(d, d, d, d)`` gate is ``(out_i, out_{i+1}, in_i, in_{i+1})``."""
    if model == "tfim":
        g = kw.get("g", 1.0); J = kw.get("J", 1.0)
        d = 2
        X, Z, I = PAULI_X, PAULI_Z, PAULI_I
        two = -J * np.kron(Z, Z)
        onsite = -g * J * X
    elif model == "heisenberg":
        S = kw.get("S", 0.5); J = kw.get("J", 1.0)
        Jz = kw.get("Jz", J); hz = kw.get("hz", 0.0)
        Sx, Sy, Sz, Sp, Sm, I = spin_operators(S)
        d = I.shape[0]
        two = J * (np.kron(Sx, Sx).real + np.kron(Sy, Sy).real) + Jz * np.kron(Sz, Sz).real
        two = two.astype(complex)
        onsite = (-hz * Sz).astype(complex)
    elif model == "hubbard":
        t = kw.get("t", 1.0); U = kw.get("U", 1.0); mu = kw.get("mu", 0.0)
        I4, c_up, c_dn, n_up, n_dn, F = _hubbard_operators()
        d = 4
        hop = (-t * (np.kron(c_up.conj().T @ F, c_up) + np.kron(F @ c_up, c_up.conj().T)
                     + np.kron(c_dn.conj().T @ F, c_dn) + np.kron(F @ c_dn, c_dn.conj().T)))
        two = hop.astype(complex)
        onsite = (U * (n_up @ n_dn) - mu * (n_up + n_dn)).astype(complex)
        I = I4
    else:
        raise ValueError(f"unknown model {model!r}")

    gates = []
    for i in range(n_sites - 1):
        left_w = 1.0 if i == 0 else 0.5
        right_w = 1.0 if i == n_sites - 2 else 0.5
        h = two + left_w * np.kron(onsite, I) + right_w * np.kron(I, onsite)
        gates.append(h.reshape(d, d, d, d))
    return gates


# --------------------------------------------------------------------------
# analytic references
# --------------------------------------------------------------------------

def tfim_energy_per_site(g, n_points=200000):
    """Thermodynamic-limit ground-state energy per site of
    ``H = -sum Z Z - g sum X`` (J = 1), the free-fermion result
    ``e0 = -(1/pi) int_0^pi sqrt(1 + g^2 - 2 g cos k) dk``."""
    k = np.linspace(0.0, np.pi, n_points)
    eps = np.sqrt(1.0 + g ** 2 - 2.0 * g * np.cos(k))
    return -np.trapezoid(eps, k) / np.pi


def tfim_gap(g):
    """Exact single-particle gap of the TFIM: ``2|1 - g|`` (closes at g = 1)."""
    return 2.0 * abs(1.0 - g)


HEISENBERG_HALF_E0 = 0.25 - np.log(2.0)     # Bethe ansatz, spin-1/2, per site (J=1)
HALDANE_GAP = 0.41048                        # spin-1 Heisenberg chain, accepted value


def lieb_robinson_velocity(model, **kw):
    """Maximum group velocity of the relevant quasiparticle dispersion."""
    if model == "tfim":
        g = kw.get("g", 1.0); J = kw.get("J", 1.0)
        return 2.0 * J * min(g, 1.0)
    raise ValueError(f"no LR velocity tabulated for {model!r}")
