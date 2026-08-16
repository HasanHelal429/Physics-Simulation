"""Madelung-rule shell filling with a hardcoded exceptions table (Phase 4)."""


def madelung_order(n_max=8, l_max=3):
    """(n, l) shells in Madelung fill order: increasing (n+l), then increasing n."""
    shells = [(n, l) for n in range(1, n_max + 1) for l in range(0, min(n, l_max + 1))]
    shells.sort(key=lambda nl: (nl[0] + nl[1], nl[0]))
    return shells


def shell_capacity(l):
    """Max electron occupancy 2*(2l+1) for a shell of angular momentum l."""
    return 2 * (2 * l + 1)


def _fill_madelung(Z, n_max=8, l_max=3):
    """Fill shells strictly along Madelung order up to Z electrons total."""
    config = {}
    remaining = Z
    for n, l in madelung_order(n_max, l_max):
        if remaining <= 0:
            break
        occ = min(shell_capacity(l), remaining)
        config[(n, l)] = occ
        remaining -= occ
    return config


# Real-world ground-state configurations deviate from naive Madelung filling for
# these elements (subtle exchange/correlation effects a configuration-averaged,
# non-relativistic Xa mean field is not expected to derive from first principles --
# this table is the source of truth, not a derived result). Each entry gives only
# the *valence* shells that differ from the Madelung-predicted pattern; everything
# below is still the ordinary noble-gas core. Values checked against NIST ASD.
# Lr (Z=103) is flagged lower-confidence: its actual configuration is sensitive to
# relativistic effects that are outside the scope of this solver.
EXCEPTIONS = {
    24: {(4, 0): 1, (3, 2): 5},  # Cr: [Ar] 3d5 4s1
    29: {(4, 0): 1, (3, 2): 10},  # Cu: [Ar] 3d10 4s1
    41: {(5, 0): 1, (4, 2): 4},  # Nb: [Kr] 4d4 5s1
    42: {(5, 0): 1, (4, 2): 5},  # Mo: [Kr] 4d5 5s1
    44: {(5, 0): 1, (4, 2): 7},  # Ru: [Kr] 4d7 5s1
    45: {(5, 0): 1, (4, 2): 8},  # Rh: [Kr] 4d8 5s1
    46: {(4, 2): 10},  # Pd: [Kr] 4d10 5s0
    47: {(5, 0): 1, (4, 2): 10},  # Ag: [Kr] 4d10 5s1
    57: {(6, 0): 2, (5, 2): 1},  # La: [Xe] 5d1 6s2
    58: {(6, 0): 2, (4, 3): 1, (5, 2): 1},  # Ce: [Xe] 4f1 5d1 6s2
    64: {(6, 0): 2, (4, 3): 7, (5, 2): 1},  # Gd: [Xe] 4f7 5d1 6s2
    78: {(6, 0): 1, (4, 3): 14, (5, 2): 9},  # Pt: [Xe] 4f14 5d9 6s1
    79: {(6, 0): 1, (4, 3): 14, (5, 2): 10},  # Au: [Xe] 4f14 5d10 6s1
    89: {(7, 0): 2, (6, 2): 1},  # Ac: [Rn] 6d1 7s2
    90: {(7, 0): 2, (6, 2): 2},  # Th: [Rn] 6d2 7s2
    91: {(7, 0): 2, (5, 3): 2, (6, 2): 1},  # Pa: [Rn] 5f2 6d1 7s2
    92: {(7, 0): 2, (5, 3): 3, (6, 2): 1},  # U: [Rn] 5f3 6d1 7s2
    93: {(7, 0): 2, (5, 3): 4, (6, 2): 1},  # Np: [Rn] 5f4 6d1 7s2
    96: {(7, 0): 2, (5, 3): 7, (6, 2): 1},  # Cm: [Rn] 5f7 6d1 7s2
    103: {(7, 0): 2, (7, 1): 1, (5, 3): 14},  # Lr: [Rn] 5f14 7s2 7p1 (low confidence)
}


def ground_state_configuration(Z, exceptions=EXCEPTIONS, n_max=8, l_max=3):
    """Fixed (n, l) -> occupation dict for the ground-state configuration of atomic number Z.

    Occupations are held fixed for a given Z throughout an SCF run (not re-sorted
    by instantaneous orbital energy each iteration), which is more stable for
    automated convergence across arbitrary Z.
    """
    if Z in exceptions:
        override = {nl: occ for nl, occ in exceptions[Z].items() if occ > 0}
        core_Z = Z - sum(override.values())
        config = _fill_madelung(core_Z, n_max, l_max)
        for nl in override:
            config.pop(nl, None)
        config.update(override)
        return config
    return _fill_madelung(Z, n_max, l_max)


_L_LABELS = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}


def format_configuration(config):
    """Human-readable string like '1s2 2s2 2p6 3s2 3p6 3d5 4s1' from a config dict."""
    items = sorted(config.items())
    return " ".join(f"{n}{_L_LABELS[l]}{occ}" for (n, l), occ in items if occ > 0)
