"""
Exact-diagonalization oracle for small chains (``N <= ~14``). Builds the full
sparse Hamiltonian as a Kronecker sum of local terms and diagonalizes it with
``scipy.sparse.linalg.eigsh`` -- the ground-truth every DMRG / TEBD result in
this project is checked against, and a useful standalone tool in its own right.

The Hamiltonians here are assembled directly from the same local operators
``models.py`` uses, *not* from the MPOs, so agreement between ED and DMRG is a
genuine cross-check of two independent construction paths.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import models


def _kron_list(ops):
    """Sparse Kronecker product of a list of dense/sparse 2D operators."""
    out = sp.csr_matrix(np.array([[1.0 + 0j]]))
    for op in ops:
        out = sp.kron(out, sp.csr_matrix(op), format="csr")
    return out


def _two_site_term(A, B, i, j, d, n):
    ops = [sp.identity(d, format="csr", dtype=complex)] * n
    ops[i] = sp.csr_matrix(A)
    ops[j] = sp.csr_matrix(B)
    return _kron_list(ops)


def _one_site_term(A, i, d, n):
    ops = [sp.identity(d, format="csr", dtype=complex)] * n
    ops[i] = sp.csr_matrix(A)
    return _kron_list(ops)


def build_hamiltonian(model, n_sites, **kw):
    """Full sparse ``H`` (dimension ``d**n_sites``) for an open chain."""
    if model == "tfim":
        g = kw.get("g", 1.0); J = kw.get("J", 1.0)
        d = 2
        X, Z = models.PAULI_X.astype(complex), models.PAULI_Z.astype(complex)
        H = sp.csr_matrix((d ** n_sites, d ** n_sites), dtype=complex)
        for i in range(n_sites - 1):
            H -= J * _two_site_term(Z, Z, i, i + 1, d, n_sites)
        for i in range(n_sites):
            H -= g * J * _one_site_term(X, i, d, n_sites)
        return H

    if model == "heisenberg":
        S = kw.get("S", 0.5); J = kw.get("J", 1.0)
        Jz = kw.get("Jz", J); hz = kw.get("hz", 0.0)
        Sx, Sy, Sz, Sp, Sm, I = models.spin_operators(S)
        d = I.shape[0]
        H = sp.csr_matrix((d ** n_sites, d ** n_sites), dtype=complex)
        for i in range(n_sites - 1):
            H += 0.5 * J * (_two_site_term(Sp, Sm, i, i + 1, d, n_sites)
                            + _two_site_term(Sm, Sp, i, i + 1, d, n_sites))
            H += Jz * _two_site_term(Sz, Sz, i, i + 1, d, n_sites)
        if hz:
            for i in range(n_sites):
                H -= hz * _one_site_term(Sz, i, d, n_sites)
        return H

    if model == "hubbard":
        return _hubbard_fock_hamiltonian(n_sites, **kw)

    raise ValueError(f"unknown model {model!r}")


def _hubbard_fock_hamiltonian(n_sites, t=1.0, U=1.0, mu=0.0):
    """1D Hubbard on the full ``2**(2*n_sites)`` Fock space, spin-orbital index
    ``p = 2*site + spin`` (spin 0 = up, 1 = down), Jordan-Wigner signs applied
    explicitly. Unambiguous reference for the MPO / gate construction."""
    n_orb = 2 * n_sites
    dim = 1 << n_orb
    rows, cols, data = [], [], []

    def apply_c(state, p):                # returns (new_state, sign) or (None, 0)
        if not (state >> p) & 1:
            return None, 0
        sign = -1 if bin(state & ((1 << p) - 1)).count("1") & 1 else 1
        return state ^ (1 << p), sign

    def apply_cdag(state, p):
        if (state >> p) & 1:
            return None, 0
        sign = -1 if bin(state & ((1 << p) - 1)).count("1") & 1 else 1
        return state ^ (1 << p), sign

    for state in range(dim):
        # diagonal: U n_up n_dn - mu (n_up + n_dn)
        diag = 0.0
        for site in range(n_sites):
            nu = (state >> (2 * site)) & 1
            nd = (state >> (2 * site + 1)) & 1
            diag += U * nu * nd - mu * (nu + nd)
        if diag:
            rows.append(state); cols.append(state); data.append(diag)
        # hopping -t (c^dag_{i,s} c_{i+1,s} + h.c.)
        for site in range(n_sites - 1):
            for s in range(2):
                p, q = 2 * site + s, 2 * (site + 1) + s
                for (a, b) in ((p, q), (q, p)):
                    s1, sg1 = apply_c(state, b)
                    if s1 is None:
                        continue
                    s2, sg2 = apply_cdag(s1, a)
                    if s2 is None:
                        continue
                    rows.append(s2); cols.append(state); data.append(-t * sg1 * sg2)
    H = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=complex)
    return H


def build_heisenberg_chain(spins, J=1.0, Jz=None):
    """Sparse Heisenberg Hamiltonian for a chain of mixed spins (per-site
    ``spins`` list). Reference for the spin-1/2-capped Haldane-gap check."""
    if Jz is None:
        Jz = J
    n = len(spins)
    ops = [models.spin_operators(S) for S in spins]
    dims = [o[5].shape[0] for o in ops]
    dim = int(np.prod(dims))
    H = sp.csr_matrix((dim, dim), dtype=complex)

    def site_op(A, i):
        mats = [sp.identity(dims[j], format="csr", dtype=complex) for j in range(n)]
        mats[i] = sp.csr_matrix(A)
        out = sp.csr_matrix(np.array([[1.0 + 0j]]))
        for m in mats:
            out = sp.kron(out, m, format="csr")
        return out

    def bond_op(A, B, i):
        mats = [sp.identity(dims[j], format="csr", dtype=complex) for j in range(n)]
        mats[i] = sp.csr_matrix(A)
        mats[i + 1] = sp.csr_matrix(B)
        out = sp.csr_matrix(np.array([[1.0 + 0j]]))
        for m in mats:
            out = sp.kron(out, m, format="csr")
        return out

    for i in range(n - 1):
        Sx_i, Sy_i, Sz_i, Sp_i, Sm_i, _ = ops[i]
        Sx_j, Sy_j, Sz_j, Sp_j, Sm_j, _ = ops[i + 1]
        H += 0.5 * J * (bond_op(Sp_i, Sm_j, i) + bond_op(Sm_i, Sp_j, i))
        H += Jz * bond_op(Sz_i, Sz_j, i)
    return H


def ground_state(model, n_sites, k=1, **kw):
    """Lowest ``k`` eigenpairs ``(energies, vectors)`` of the model Hamiltonian.
    ``vectors[:, i]`` is the i-th eigenvector (dimension ``d**n_sites``)."""
    H = build_hamiltonian(model, n_sites, **kw)
    H = 0.5 * (H + H.conj().T)                   # symmetrize away round-off
    dim = H.shape[0]
    if dim <= 256:
        w, v = np.linalg.eigh(H.toarray())
        return w[:k].real, v[:, :k]
    w, v = spla.eigsh(H, k=min(k, dim - 2), which="SA")
    order = np.argsort(w)
    return w[order].real, v[:, order]


def entanglement_entropy(state, cut, d, n_sites):
    """Von Neumann entropy (nats) of the reduced density matrix on sites
    ``0..cut-1`` for a full state vector."""
    psi = state.reshape([d] * n_sites)
    psi = psi.reshape(d ** cut, d ** (n_sites - cut))
    s = np.linalg.svd(psi, compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    p /= p.sum()
    return float(-np.sum(p * np.log(p)))


def local_expectation(state, op, site, d, n_sites):
    """``<state| op_site |state>`` for a single-site operator."""
    psi = state.reshape([d] * n_sites)
    moved = np.moveaxis(psi, site, 0).reshape(d, -1)
    return complex(np.tensordot(moved.conj(), op @ moved, axes=([0, 1], [0, 1])))
