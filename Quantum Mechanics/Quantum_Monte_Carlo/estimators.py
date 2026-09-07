"""
Estimators computed from a walker population: energy (with reblocked error),
pair-correlation function g(r), and the one-body density. Also the small
cross-solver accuracy-ledger table builder.
"""

import numpy as np

from vmc import reblock_error


def energy_estimate(local_energies, weights=None):
    """(mean, reblocked stderr) from a ``(n_samp, W)`` array of local energies
    (optionally weighted per sample/walker)."""
    E = np.asarray(local_energies, float)
    if weights is None:
        per_sample = E.mean(axis=1)
        return float(E.mean()), reblock_error(per_sample)
    w = np.asarray(weights, float)
    per_sample = np.sum(w * E, axis=1) / np.sum(w, axis=1)
    return float(np.sum(w * E) / np.sum(w)), reblock_error(per_sample)


def pair_correlation(walkers, L=None, n_bins=60, r_max=None):
    """Spherically averaged electron-electron g(r). For a finite molecule
    ``L`` is None and g(r) is un-normalized-by-density (just the histogram of
    inter-electron distances, scaled by 1/(4 pi r^2 dr)). For the periodic
    electron gas pass the box side ``L`` (minimum image) and it is normalized
    to the ideal-gas density so g(r) -> 1 at large r."""
    R = np.asarray(walkers, float)
    Wn, n_e, _ = R.shape
    diff = R[:, :, None, :] - R[:, None, :, :]
    if L is not None:
        diff -= L * np.round(diff / L)
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    iu = np.triu_indices(n_e, k=1)
    d = dist[:, iu[0], iu[1]].ravel()
    if r_max is None:
        r_max = (L / 2) if L is not None else np.percentile(d, 99)
    hist, edges = np.histogram(d, bins=n_bins, range=(0, r_max))
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 * np.pi * r ** 2 * (edges[1] - edges[0])
    if L is not None:
        # normalize so g(r) -> 1 for an ideal gas: expected pair count in a
        # shell = (n_pairs) * shell_volume / box_volume
        n_pairs = n_e * (n_e - 1) / 2
        expected = n_pairs * shell / L ** 3
        g = hist / (Wn * np.where(expected > 0, expected, 1.0))
    else:
        g = hist / (shell * Wn)
    return r, g


def density_radial(walkers, center=(0, 0, 0), n_bins=60, r_max=None):
    """Radial electron density n(r) about `center`."""
    R = np.asarray(walkers, float)
    c = np.asarray(center, float)
    d = np.sqrt(np.sum((R - c) ** 2, axis=-1)).ravel()
    if r_max is None:
        r_max = np.percentile(d, 99)
    hist, edges = np.histogram(d, bins=n_bins, range=(0, r_max))
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 * np.pi * r ** 2 * (edges[1] - edges[0])
    return r, hist / (shell * R.shape[0])


def accuracy_ledger(rows):
    """`rows`: list of dicts with keys system, method, E, (optional) err.
    Returns a formatted text table plus, per system, the correlation energy
    each method captures relative to the row tagged method='exact'."""
    systems = []
    for r in rows:
        if r["system"] not in systems:
            systems.append(r["system"])
    by = {(r["system"], r["method"]): r for r in rows}
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    lines = []
    head = f"{'system':<6} " + " ".join(f"{m:>12}" for m in methods)
    lines.append(head)
    lines.append("-" * len(head))
    for s in systems:
        cells = []
        for m in methods:
            r = by.get((s, m))
            cells.append(f"{r['E']:>12.4f}" if r else f"{'--':>12}")
        lines.append(f"{s:<6} " + " ".join(cells))

    lines.append("")
    lines.append("correlation energy captured (E_method - E_HF), Ha:")
    for s in systems:
        hf = by.get((s, "HF")) or by.get((s, "hydrogenic"))
        if hf is None:
            continue
        cells = []
        for m in methods:
            r = by.get((s, m))
            if r and m not in ("HF", "hydrogenic"):
                cells.append(f"{m}:{r['E'] - hf['E']:+.4f}")
        lines.append(f"  {s:<5} " + "  ".join(cells))
    return "\n".join(lines)
