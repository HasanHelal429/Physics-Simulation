"""Molecular orbital filling (Phase 3): straightforward aufbau on whatever
orbital energies the eigensolver returns per lambda-channel -- unlike the
atomic solver's shells.py, no Madelung-style hardcoded ordering table is
needed here, since there's no analogous well-known systematic exception
pattern for diatomics to encode; real MO level orderings (including
famous cases like the pi_u/sigma_g crossing between N2 and O2) fall out of
whatever the SCF potential actually computes, not a fixed rule.

lambda (the azimuthal/|m| quantum number, this project's diatomic analog
of the atomic solver's l) labels orbitals sigma/pi/delta/phi for
lambda=0/1/2/3, by direct analogy to atomic s/p/d/f. Spatial degeneracy is
1 for sigma (only m=0) and 2 for lambda>0 (m=+lambda and m=-lambda are
degenerate for any axially-symmetric potential); with spin, that's 2
electrons per sigma level and 4 per pi/delta/phi level.
"""

LAMBDA_LABELS = {0: "sigma", 1: "pi", 2: "delta", 3: "phi"}


def spatial_degeneracy(lam):
    """1 for sigma (lambda=0), 2 for pi/delta/phi (lambda>0, from +-lambda)."""
    return 1 if lam == 0 else 2


def orbital_capacity(lam):
    """Electron capacity of one (lambda, energy) level, spin included."""
    return 2 * spatial_degeneracy(lam)


def aufbau_fill(candidates, N_electrons):
    """candidates: list of (lambda, index, energy) tuples, index just a
    tie-breaking label (e.g. the state's position within its lambda
    -channel) distinguishing multiple levels of the same lambda. Fills
    electrons into levels in increasing energy order until N_electrons are
    placed.

    Returns a list of (lambda, index, energy, occupation) tuples for every
    level that received at least one electron -- occupation equals
    orbital_capacity(lambda) for every fully-filled level except possibly
    the last, which may be partially filled (an open-shell configuration,
    e.g. O2's degenerate pi_g pair each holding 1 electron by Hund's rule
    in reality -- this simple aufbau doesn't apply Hund's rule, and
    reports the partial filling as-is, undistributed across a degenerate
    pair, a known limitation noted alongside the rest of this project's
    single-configuration-averaged treatment).
    """
    ordered = sorted(candidates, key=lambda c: c[2])
    filled = []
    remaining = N_electrons
    for lam, idx, energy in ordered:
        if remaining <= 0:
            break
        cap = orbital_capacity(lam)
        occ = min(cap, remaining)
        filled.append((lam, idx, energy, occ))
        remaining -= occ
    if remaining > 0:
        raise ValueError(
            f"aufbau_fill: ran out of candidate levels with {remaining} electrons left to place "
            f"-- pass more states per lambda-channel"
        )
    return filled


def format_configuration(filled, parities=None):
    """Render a filled-level list as e.g. '1sigma_g^2 1sigma_u^2 1pi_u^4'.

    parities, if given, is a dict {(lambda, index): 'g'|'u'} for the
    homonuclear gerade/ungerade label (see diatomic_driver.assign_parity);
    omitted entirely for heteronuclear molecules, which have no such
    symmetry.
    """
    counts = {}
    parts = []
    for lam, idx, energy, occ in filled:
        counts[lam] = counts.get(lam, 0) + 1
        label = f"{counts[lam]}{LAMBDA_LABELS.get(lam, f'l={lam}')}"
        if parities is not None:
            label += f"_{parities[(lam, idx)]}"
        parts.append(f"{label}^{occ}")
    return " ".join(parts)
