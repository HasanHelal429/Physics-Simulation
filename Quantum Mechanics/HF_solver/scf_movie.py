"""Animate a run_scf(..., record_history=True) trajectory (Phase 7 visualization).

Four synchronized panels per frame, one frame per SCF iteration:
  - radial density 4*pi*r**2*rho(r): shell peaks sharpening into place
  - effective potential r*V_eff(r): relaxing from the crude seed to self-consistency
  - orbital energy levels per l-channel: eigenvalues settling as the density updates
  - E_total vs iteration: the convergence trace, with a marker for "you are here"

Saved as .mp4 via matplotlib's ffmpeg writer if available, otherwise falls
back to an animated .gif (Pillow writer, always available).
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

_L_LABELS = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
_L_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red", 4: "tab:purple"}


def make_scf_animation(result, atom_label, out_path, r_max=6.0, r_min_visual=0.02, fps=8, dpi=130):
    """Build and save the 4-panel SCF animation for a record_history=True result dict.

    Density/potential panels use a log r-axis (r_min_visual..r_max): core
    shells sit at r ~ 1/Z while valence shells sit at r ~ few Bohr, so a
    linear axis squashes the core peak into invisible width -- log-r is the
    standard way atomic-structure plots show all shells at comparable scale.
    """
    snapshots = result["snapshots"]
    if not snapshots:
        raise ValueError("result has no snapshots -- rerun run_scf(..., record_history=True)")

    r = result["r"]
    mask = (r > r_min_visual) & (r < r_max)
    r_plot = r[mask]

    dens_curves = [4 * np.pi * r_plot**2 * snap["rho"][mask] for snap in snapshots]
    pot_curves = [r_plot * snap["V"][mask] for snap in snapshots]
    energies = [snap["E_total"] for snap in snapshots]
    iters = [snap["iteration"] for snap in snapshots]

    all_nl = sorted({nl for snap in snapshots for nl in snap["orbital_energies"]}, key=lambda nl: (nl[1], nl[0]))
    eps_min = min(min(snap["orbital_energies"].values()) for snap in snapshots)
    eps_max = max(max(snap["orbital_energies"].values()) for snap in snapshots)
    pad = 0.08 * (eps_max - eps_min + 1e-9)

    dens_ymax = max(c.max() for c in dens_curves) * 1.1
    pot_ymin = min(c.min() for c in pot_curves) * 1.1
    pot_ymax = max(c.max() for c in pot_curves) * 1.1 + 1e-9

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    ax_dens, ax_pot, ax_levels, ax_energy = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    fig.suptitle(f"Hartree-Fock-Slater SCF loop: {atom_label}", fontsize=14, y=0.995)

    (line_dens,) = ax_dens.plot([], [], color="tab:blue", lw=2)
    ax_dens.set_xscale("log")
    ax_dens.set_xlim(r_min_visual, r_max)
    ax_dens.set_ylim(0, dens_ymax)
    ax_dens.set_xlabel("r (Bohr, log scale)")
    ax_dens.set_ylabel(r"$4\pi r^2 \rho(r)$")
    ax_dens.set_title("Electron density (shell peaks emerging)")

    (line_pot,) = ax_pot.plot([], [], color="tab:red", lw=2)
    ax_pot.set_xscale("log")
    ax_pot.set_xlim(r_min_visual, r_max)
    ax_pot.set_ylim(pot_ymin, pot_ymax)
    ax_pot.set_xlabel("r (Bohr, log scale)")
    ax_pot.set_ylabel(r"$r \cdot V_{eff}(r)$")
    ax_pot.set_title("Effective potential (screening building up)")
    ax_pot.axhline(-result["Z"], color="0.7", ls=":", lw=1, label=r"bare nucleus $-Z$")
    ax_pot.legend(loc="lower right", fontsize=8)

    level_artists = {}
    for nl in all_nl:
        l = nl[1]
        (ln,) = ax_levels.plot([], [], marker="_", markersize=28, mew=3, color=_L_COLORS.get(l, "k"), linestyle="none")
        level_artists[nl] = ln
    ax_levels.set_xlim(-0.5, max(nl[1] for nl in all_nl) + 0.5)
    ax_levels.set_yscale("symlog", linthresh=0.5)
    ax_levels.set_ylim(eps_min * 1.15 - pad, eps_max + pad + 0.1)
    ax_levels.set_xticks(sorted({nl[1] for nl in all_nl}))
    ax_levels.set_xticklabels([_L_LABELS[l] for l in sorted({nl[1] for nl in all_nl})])
    ax_levels.set_xlabel("shell (l)")
    ax_levels.set_ylabel(r"orbital energy $\epsilon_{nl}$ (Ha, symlog)")
    ax_levels.set_title("Occupied orbital energies converging")

    ax_energy.plot(iters, energies, color="0.75", lw=1)
    (marker_energy,) = ax_energy.plot([], [], marker="o", color="k", markersize=7)
    ax_energy.set_xlim(1, max(iters))
    e_pad = 0.05 * (max(energies) - min(energies) + 1e-9)
    ax_energy.set_ylim(min(energies) - e_pad, max(energies) + e_pad)
    ax_energy.set_xlabel("SCF iteration")
    ax_energy.set_ylabel(r"$E_{total}$ (Ha)")
    ax_energy.set_title("Total energy convergence")

    iter_text = fig.text(0.5, 0.955, "", ha="center", fontsize=11, color="0.2")
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    def update(frame):
        line_dens.set_data(r_plot, dens_curves[frame])
        line_pot.set_data(r_plot, pot_curves[frame])
        snap = snapshots[frame]
        for nl, ln in level_artists.items():
            eps = snap["orbital_energies"].get(nl)
            if eps is not None:
                ln.set_data([nl[1]], [eps])
            else:
                ln.set_data([], [])
        marker_energy.set_data([iters[frame]], [energies[frame]])
        iter_text.set_text(f"iteration {snap['iteration']} / {iters[-1]}   " f"dE={snap['dE']:.1e}   dn={snap['dn']:.1e}")
        return (line_dens, line_pot, marker_energy, iter_text, *level_artists.values())

    anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000 / fps, blit=False)

    writer = "ffmpeg" if animation.FFMpegWriter.isAvailable() else "pillow"
    if writer == "pillow" and out_path.endswith(".mp4"):
        out_path = out_path[: -len(".mp4")] + ".gif"
    anim.save(out_path, writer=writer, fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path
