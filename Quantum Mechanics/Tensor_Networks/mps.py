"""
Matrix Product State: a chain of rank-3 tensors ``M[i]`` with index order
``(left_bond, physical, right_bond)``, kept in *mixed canonical form* about a
movable orthogonality centre.

Everything downstream -- observables, correlators, entanglement entropy, DMRG,
TEBD -- rests on this class. The design goals are (1) numerically stable
canonicalization via SVD, (2) O(1) local expectation values when the centre is
already at the site of interest, and (3) truncation with an honest
discarded-weight report.

Sign/normalization conventions
------------------------------
* A tensor left of the centre is **left-canonical**: reshaped to
  ``(Dl*d, Dr)`` its columns are orthonormal (``sum_s A_s^dag A_s = I``).
* A tensor right of the centre is **right-canonical**: reshaped to
  ``(Dl, d*Dr)`` its rows are orthonormal (``sum_s B_s B_s^dag = I``).
* The centre tensor carries the norm; ``<psi|psi> = || centre ||_F^2``.
"""

import numpy as np
from scipy.linalg import svd


def _svd(mat, chi_max=None, cutoff=1e-14):
    """Economy SVD with optional truncation. Returns ``U, s, Vh, discarded``
    where ``discarded`` is the summed squared weight of the dropped singular
    values (relative to the total)."""
    try:
        U, s, Vh = svd(mat, full_matrices=False, lapack_driver="gesdd")
    except np.linalg.LinAlgError:
        U, s, Vh = svd(mat, full_matrices=False, lapack_driver="gesvd")
    total = np.sum(s ** 2)
    keep = np.sum(s > cutoff * (s[0] if s.size else 1.0))
    keep = max(keep, 1)
    if chi_max is not None:
        keep = min(keep, chi_max)
    discarded = float(np.sum(s[keep:] ** 2) / total) if total > 0 else 0.0
    return U[:, :keep], s[:keep], Vh[:keep, :], discarded


class MPS:
    def __init__(self, tensors, center=None):
        self.M = [np.asarray(t, dtype=complex) for t in tensors]
        self.n = len(self.M)
        self.d = self.M[0].shape[1]
        self.center = center

    # ------------------------------------------------------------------ builders
    @classmethod
    def product_state(cls, configs, d):
        """Product state; ``configs[i]`` is the occupied basis index at site i."""
        tensors = []
        for c in configs:
            t = np.zeros((1, d, 1), dtype=complex)
            t[0, c, 0] = 1.0
            tensors.append(t)
        return cls(tensors, center=0)

    @classmethod
    def random(cls, n, d, chi, seed=None):
        rng = np.random.default_rng(seed)
        tensors = []
        for i in range(n):
            Dl = 1 if i == 0 else min(chi, d ** i, d ** (n - i))
            Dr = 1 if i == n - 1 else min(chi, d ** (i + 1), d ** (n - i - 1))
            tensors.append(rng.standard_normal((Dl, d, Dr))
                           + 1j * rng.standard_normal((Dl, d, Dr)))
        m = cls(tensors)
        m.canonicalize()
        m.normalize()
        return m

    @classmethod
    def random_mixed(cls, dims, chi, seed=None):
        """Random MPS on a chain of per-site physical dimensions ``dims``."""
        rng = np.random.default_rng(seed)
        n = len(dims)

        def _cap(seq):
            out, acc = [], 1
            for x in seq:
                acc = min(acc * x, 1 << 40)
                out.append(acc)
            return out
        cum = [1] + _cap(dims)
        rev = ([1] + _cap(list(dims[::-1])))[::-1]
        tensors = []
        for i in range(n):
            Dl = 1 if i == 0 else int(min(chi, cum[i], rev[i]))
            Dr = 1 if i == n - 1 else int(min(chi, cum[i + 1], rev[i + 1]))
            tensors.append(rng.standard_normal((Dl, dims[i], Dr))
                           + 1j * rng.standard_normal((Dl, dims[i], Dr)))
        m = cls(tensors)
        m.canonicalize()
        m.normalize()
        return m

    @classmethod
    def from_statevector(cls, psi, d, n, chi_max=None):
        """Exact MPS of a full state vector by successive SVD (left to right)."""
        psi = np.asarray(psi, dtype=complex).reshape([d] * n)
        tensors = []
        rest = psi.reshape(1, -1)
        bond = 1
        for i in range(n - 1):
            rest = rest.reshape(bond * d, d ** (n - i - 1))
            U, s, Vh, _ = _svd(rest, chi_max=chi_max)
            newbond = U.shape[1]
            tensors.append(U.reshape(bond, d, newbond))
            rest = (np.diag(s) @ Vh)
            bond = newbond
        tensors.append(rest.reshape(bond, d, 1))
        m = cls(tensors, center=n - 1)
        return m

    # ------------------------------------------------------------------ canonical
    def _left_sweep_step(self, i, chi_max=None):
        """SVD-split site ``i`` into a left-canonical tensor, push ``s Vh`` right."""
        Dl, d, Dr = self.M[i].shape
        U, s, Vh, disc = _svd(self.M[i].reshape(Dl * d, Dr), chi_max=chi_max)
        self.M[i] = U.reshape(Dl, d, -1)
        SV = np.diag(s) @ Vh
        self.M[i + 1] = np.tensordot(SV, self.M[i + 1], axes=(1, 0))
        return disc

    def _right_sweep_step(self, i, chi_max=None):
        """SVD-split site ``i`` into a right-canonical tensor, push ``U s`` left."""
        Dl, d, Dr = self.M[i].shape
        U, s, Vh, disc = _svd(self.M[i].reshape(Dl, d * Dr), chi_max=chi_max)
        self.M[i] = Vh.reshape(-1, d, Dr)
        US = U @ np.diag(s)
        self.M[i - 1] = np.tensordot(self.M[i - 1], US, axes=(2, 0))
        return disc

    def canonicalize(self, chi_max=None):
        """Full sweep: right-canonicalize everything, centre lands on site 0."""
        for i in range(self.n - 1):
            self._left_sweep_step(i, chi_max=None)
        for i in range(self.n - 1, 0, -1):
            self._right_sweep_step(i, chi_max=chi_max)
        self.center = 0
        return self

    def move_center_to(self, target, chi_max=None):
        if self.center is None:
            self.canonicalize(chi_max=chi_max)
        while self.center < target:
            self._left_sweep_step(self.center, chi_max=chi_max)
            self.center += 1
        while self.center > target:
            self._right_sweep_step(self.center, chi_max=chi_max)
            self.center -= 1
        return self

    def normalize(self):
        i = self.center if self.center is not None else 0
        self.move_center_to(i)
        nrm = np.linalg.norm(self.M[i])
        if nrm > 0:
            self.M[i] = self.M[i] / nrm
        return self

    # ------------------------------------------------------------------ observ.
    def bond_dimensions(self):
        return [self.M[i].shape[2] for i in range(self.n - 1)]

    def max_bond(self):
        return max(self.bond_dimensions()) if self.n > 1 else 1

    def schmidt_values(self, bond):
        """Singular values on ``bond`` (between site ``bond`` and ``bond+1``)."""
        self.move_center_to(bond)
        Dl, d, Dr = self.M[bond].shape
        _, s, _, _ = _svd(self.M[bond].reshape(Dl * d, Dr))
        return s / np.linalg.norm(s)

    def entanglement_entropy(self, bond):
        s = self.schmidt_values(bond)
        p = s ** 2
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log(p)))

    def entanglement_profile(self):
        return np.array([self.entanglement_entropy(b) for b in range(self.n - 1)])

    def expectation_1site(self, op, site):
        self.move_center_to(site)
        A = self.M[site]
        opA = np.tensordot(op, A, axes=(1, 1))          # (d_out, Dl, Dr)
        val = np.tensordot(A.conj(), opA, axes=([0, 1, 2], [1, 0, 2]))
        return complex(val)

    def density_profile(self, op):
        return np.array([self.expectation_1site(op, i).real for i in range(self.n)])

    def correlator(self, opA, i, opB, j):
        """``<opA_i opB_j>`` for ``i < j`` via transfer-matrix contraction."""
        if i == j:
            return self.expectation_1site(opA @ opB, i)
        if i > j:
            i, j, opA, opB = j, i, opB, opA
        self.move_center_to(i)
        A = self.M[i]
        left = np.tensordot(A.conj(), np.tensordot(opA, A, axes=(1, 1)),
                            axes=([0, 1], [1, 0]))       # (Dr_bra, Dr_ket)
        for k in range(i + 1, j):
            Ak = self.M[k]
            left = np.tensordot(left, Ak, axes=(1, 0))    # (Dr_bra, d, Dr_ket)
            left = np.tensordot(Ak.conj(), left, axes=([0, 1], [0, 1]))
        Aj = self.M[j]
        right = np.tensordot(opB, Aj, axes=(1, 1))        # (d_out, Dl, Dr)
        right = np.tensordot(Aj.conj(), right, axes=([1, 2], [0, 2]))  # (Dl_bra, Dl_ket)
        return complex(np.tensordot(left, right, axes=([0, 1], [0, 1])))

    def correlators_from(self, opA, i, opB, j_list):
        """``[<opA_i opB_j> for j in j_list]`` (all ``j > i``) in a single
        left-to-right transfer-matrix sweep -- far cheaper than calling
        ``correlator`` once per ``j`` when many are needed. Requires the sites
        right of ``i`` to be right-canonical, which ``move_center_to(i)``
        ensures."""
        j_list = sorted(int(j) for j in j_list)
        self.move_center_to(i)
        A = self.M[i]
        # left(b', b): opA applied at site i, over the right bond of i
        left = np.tensordot(A.conj(), np.tensordot(opA, A, axes=(1, 1)),
                            axes=([0, 1], [1, 0]))
        out, jset = {}, set(j_list)
        for k in range(i + 1, j_list[-1] + 1):
            B = self.M[k]                                  # (Dl, d, Dr), right-canonical
            if k in jset:
                tmp = np.tensordot(left, B, axes=(1, 0))              # (b', s, c)
                tmp = np.tensordot(opB, tmp, axes=(1, 1))             # (s', b', c)
                clsd = np.tensordot(B.conj(), tmp, axes=([0, 1], [1, 0]))  # (c', c)
                out[k] = complex(np.trace(clsd))
            # advance past site k with the identity
            tmp = np.tensordot(left, B, axes=(1, 0))       # (b', s, c)
            left = np.tensordot(B.conj(), tmp, axes=([0, 1], [0, 1]))  # (c', c)
        return [out[j] for j in j_list]

    def overlap(self, other):
        """``<self|other>``."""
        E = np.ones((1, 1), dtype=complex)
        for k in range(self.n):
            E = np.tensordot(E, other.M[k], axes=(1, 0))          # (Db, d, Dk)
            E = np.tensordot(self.M[k].conj(), E, axes=([0, 1], [0, 1]))
        return complex(E[0, 0])

    def norm(self):
        return np.sqrt(self.overlap(self).real)

    def expectation_mpo(self, mpo):
        """``<psi| H |psi>`` for an MPO list (index order lbond,rbond,out,in)."""
        E = np.ones((1, 1, 1), dtype=complex)                     # (bra, w, ket)
        for k in range(self.n):
            W = mpo[k]
            E = np.tensordot(E, self.M[k], axes=(2, 0))           # (bra, w, d_ket, ketR)
            E = np.tensordot(E, W, axes=([1, 2], [0, 3]))         # (bra, ketR, rb, out)
            E = np.tensordot(self.M[k].conj(), E, axes=([0, 1], [0, 3]))  # (braR, ketR, rb)
            E = E.transpose(0, 2, 1)                              # (braR, rb, ketR)
        return complex(E[0, 0, 0])

    def copy(self):
        m = MPS([t.copy() for t in self.M], center=self.center)
        return m

    def to_statevector(self):
        psi = self.M[0]
        for k in range(1, self.n):
            psi = np.tensordot(psi, self.M[k], axes=(psi.ndim - 1, 0))
        return psi.reshape(-1)
