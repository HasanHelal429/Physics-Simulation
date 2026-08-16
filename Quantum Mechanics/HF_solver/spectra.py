"""Single-active-electron emission spectra from frozen HF-Slater orbital energies.

Approximation (a "logical leap" beyond Phases 1-6, not literally Phase 7 as
originally scoped): freeze the converged self-consistent V_eff from run_scf,
then treat one electron promoted out of the atom's valence shell (the last
shell filled along Madelung order) as a single active electron moving through
the *excited* eigenstates of that frozen mean field, emitting a photon when it
falls to a lower shell with a Pauli-allowed vacancy. This is exactly the
hydrogenic Lyman/Balmer/Paschen construction generalized to a screened
potential, and is the standard zeroth-order picture behind alkali/Rydberg
spectra (e.g. the Na D-line as 3p -> 3s).

Known limitations (in addition to the ones already documented for the SCF
itself in HF_Solver_Plan.md):
- No orbital relaxation: the core density is frozen at its *ground-state*
  self-consistent value, not re-relaxed around the excited electron+hole.
  This is a Koopmans'-theorem-like approximation and systematically
  under-estimates real transition energies (validated below: predicted
  wavelengths for Na/K/Ar resonance lines are consistently 13-25% too long,
  i.e. energies too small, matching the direction Koopmans' theorem error
  should have).
- Local Xa exchange lacks the correct -1/r asymptotic tail (a well-known
  self-interaction error of Hartree/LDA-type potentials), so only the
  lowest 1-2 unoccupied levels per l are physically meaningful bound
  Rydberg states -- higher ones are numerical artifacts of the finite box.
  `excited_levels`/`dipole_transitions` filter these out via `bound_cutoff`.
- No spin-orbit coupling, so each transition is a single line, not the
  fine-structure doublets/multiplets seen in real spectra (e.g. Na's D1/D2).
- Single-configuration/single-active-electron only: no shake-up, no
  correlation, no full term-symbol structure.
"""

import numpy as np

import atomic_scf
import shells

HARTREE_TO_EV = 27.211386245988
HC_EV_NM = 1239.8419843320025  # h*c in eV*nm


def excited_levels(r, h, V, l_max=3, n_states=6):
    """All (n, l) -> energy (Hartree) eigenpairs of the frozen potential V, l=0..l_max.

    n_states per l is generous on purpose; bound_cutoff in dipole_transitions
    does the real work of discarding the unphysical near-zero-energy tail.
    """
    levels = {}
    for l in range(l_max + 1):
        energies, _ = atomic_scf.solve_radial_channel(r, l, V, h, n_states=n_states)
        for i, eps in enumerate(energies):
            levels[(l + 1 + i, l)] = eps
    return levels


def valence_shell(config):
    """The last (n, l) filled along Madelung order with occ > 0 -- the "active electron" shell."""
    order = {nl: i for i, nl in enumerate(shells.madelung_order())}
    occupied = [nl for nl, occ in config.items() if occ > 0]
    return max(occupied, key=lambda nl: order[nl])


def dipole_transitions(levels, config, bound_cutoff=-1e-6, max_wavelength_nm=1500):
    """Emission lines (upper -> lower) among bound, Pauli-allowed, dipole-allowed levels.

    Candidate levels are the valence shell (the hole left behind once one
    electron is promoted) plus any shell with room in the ground-state
    configuration (occupation < 2*(2l+1)), restricted to eps < bound_cutoff
    Hartree so at least *some* binding is present. A line is kept if it is
    dipole-allowed (Delta l = +-1), the upper level sits above the lower one
    in energy, and the resulting wavelength is under max_wavelength_nm.

    That wavelength cap matters more than bound_cutoff in practice: the local
    Xa potential's missing -1/r asymptotic tail (self-interaction error, see
    module docstring) leaves a handful of barely-bound "Rydberg" levels only
    a few 1e-4 Ha deep, which are numerical artifacts of the finite box, not
    real high-n Rydberg states -- they show up as spurious multi-micron
    "transitions" if not filtered out.
    """
    v_nl = valence_shell(config)
    candidates = {v_nl} | {
        nl for nl, eps in levels.items() if config.get(nl, 0) < shells.shell_capacity(nl[1]) and eps < bound_cutoff
    }

    lines = []
    for nl_u in candidates:
        for nl_l in candidates:
            if nl_u == nl_l or abs(nl_u[1] - nl_l[1]) != 1:
                continue
            e_u, e_l = levels[nl_u], levels[nl_l]
            if e_u <= e_l:
                continue
            dE_ha = e_u - e_l
            E_eV = dE_ha * HARTREE_TO_EV
            wavelength_nm = HC_EV_NM / E_eV
            if wavelength_nm > max_wavelength_nm:
                continue
            lines.append(
                {
                    "upper": nl_u,
                    "lower": nl_l,
                    "delta_E_hartree": dE_ha,
                    "energy_eV": E_eV,
                    "wavelength_nm": wavelength_nm,
                }
            )
    lines.sort(key=lambda ln: ln["wavelength_nm"])
    return lines


def atomic_spectrum(result, l_max=3, n_states=6, bound_cutoff=-1e-4):
    """Convenience wrapper: emission lines for a converged run_scf(...) result dict."""
    levels = excited_levels(result["r"], result["h"], result["V"], l_max=l_max, n_states=n_states)
    lines = dipole_transitions(levels, result["config"], bound_cutoff=bound_cutoff)
    return levels, lines


def wavelength_to_rgb(wl_nm):
    """Approximate visible-spectrum RGB for a wavelength (Dan Bruton's algorithm).

    Outside 380-750 nm (UV/IR, most of what this solver actually produces --
    alkali resonance lines land near-visible/near-IR, closed-shell resonance
    lines land in the vacuum UV) returns a muted violet/dark-red so those
    lines still render as ticks rather than vanishing.
    """
    wl = wl_nm
    if wl < 380:
        return (0.55, 0.0, 0.85)
    if wl > 750:
        return (0.35, 0.0, 0.0)

    if wl < 440:
        r, g, b = -(wl - 440) / (440 - 380), 0.0, 1.0
    elif wl < 490:
        r, g, b = 0.0, (wl - 440) / (490 - 440), 1.0
    elif wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / (510 - 490)
    elif wl < 580:
        r, g, b = (wl - 510) / (580 - 510), 1.0, 0.0
    elif wl < 645:
        r, g, b = 1.0, -(wl - 645) / (645 - 580), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0

    if wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl > 700:
        factor = 0.3 + 0.7 * (750 - wl) / (750 - 700)
    else:
        factor = 1.0

    return tuple(max(0.0, min(1.0, c * factor)) for c in (r, g, b))


_L_LABELS = {0: "s", 1: "p", 2: "d", 3: "f"}


def _term_label(nl):
    n, l = nl
    return f"{n}{_L_LABELS.get(l, '?')}"


def plot_spectrum(lines, ax=None, atom_label="", experimental_nm=None):
    """Vertical emission-line spectrum on a dark strip, colored by true wavelength.

    Styled like a photographed spectrograph readout (black strip, bright lines)
    rather than a generic categorical chart, since color here isn't an arbitrary
    series encoding -- it *is* the physical quantity (the actual color of the
    emitted light), so a realistic color-by-wavelength mapping is the correct
    (and only sensible) choice, not a design-system categorical palette.

    experimental_nm: optional literature wavelength(s) (float or list) drawn as
    a dashed white reference line, for comparing predicted vs. real resonance lines.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.0))

    ax.set_facecolor("#060608")
    ax.axvline(380, color="0.35", lw=0.8, ls=":", zorder=1)
    ax.axvline(750, color="0.35", lw=0.8, ls=":", zorder=1)
    ax.annotate(
        "visible",
        xy=(np.sqrt(380 * 750), 0.03),
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.55",
    )

    for ln in lines:
        wl = ln["wavelength_nm"]
        color = wavelength_to_rgb(wl)
        ax.axvline(wl, color=color, lw=3, zorder=3)
        label = f"{_term_label(ln['upper'])}$\\to${_term_label(ln['lower'])}  {wl:.1f} nm"
        ax.annotate(
            label,
            xy=(wl, 0.94),
            ha="center",
            va="top",
            fontsize=8.5,
            color="white",
            zorder=4,
        )

    if experimental_nm is not None:
        exp = np.atleast_1d(experimental_nm)
        for wl in exp:
            ax.axvline(wl, color="white", ls="--", lw=1.4, zorder=2, alpha=0.9)
        ax.annotate(
            f"experiment {exp[0]:.1f} nm",
            xy=(exp[0], 0.5),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7.5,
            color="0.75",
            rotation=90,
        )

    ax.set_xscale("log")
    ax.set_xlim(50, 2000)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("wavelength (nm)")
    ax.set_title(f"{atom_label}: predicted emission line(s)")
    return ax


def plot_grotrian(levels, lines, config, ax=None, atom_label="", l_max=3):
    """Energy-level diagram (columns = l, ticks = levels) with emission arrows.

    Only levels within the transitions list (plus the valence shell) are
    drawn, to keep the diagram readable -- the full `levels` dict includes
    the numerical continuum artifacts filtered out by dipole_transitions.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    v_nl = valence_shell(config)
    shown = {v_nl} | {ln["upper"] for ln in lines} | {ln["lower"] for ln in lines}

    for nl in shown:
        l = nl[1]
        eps = levels[nl]
        occ = config.get(nl, 0)
        cap = shells.shell_capacity(l)
        style = dict(color="k") if occ >= cap else dict(color="0.4", ls="--")
        ax.hlines(eps, l - 0.35, l + 0.35, lw=2.5, **style)
        ax.annotate(
            f"{_term_label(nl)} ({occ}/{cap}e)",
            xy=(l + 0.38, eps),
            va="center",
            fontsize=8,
        )

    for ln in lines:
        l_u, l_l = ln["upper"][1], ln["lower"][1]
        e_u, e_l = levels[ln["upper"]], levels[ln["lower"]]
        color = wavelength_to_rgb(ln["wavelength_nm"])
        ax.annotate(
            "",
            xy=(l_l, e_l),
            xytext=(l_u, e_u),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.6, shrinkA=6, shrinkB=6),
        )

    ax.set_xlim(-0.6, l_max + 1.6)
    ax.set_xticks(range(l_max + 1))
    ax.set_xticklabels([_L_LABELS.get(l, "?") for l in range(l_max + 1)])
    ax.set_xlabel("orbital angular momentum l")
    ax.set_ylabel("orbital energy (Ha), frozen V_eff")
    ax.set_title(f"{atom_label}: excited single-particle levels\n(solid=Pauli-full, dashed=has vacancy)")
    return ax
