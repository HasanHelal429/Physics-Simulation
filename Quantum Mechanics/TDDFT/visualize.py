"""Phase 6 -- visualization.

Builds the density-difference movie delta-rho(r, t) = n(r, t) - n(r, 0) for a
Helium delta-kick: the electron cloud sloshing back and forth after the
impulse, shown as a mid-plane slice next to the induced dipole trace. This is
the real-space picture behind the Phase 3 absorption spectrum -- every line in
S(omega) is one normal mode of this sloshing.

    python visualize.py [--quick] [--out FILE.mp4]

Writes media/deltarho_he_kick.mp4 (falls back to a PNG strip if ffmpeg is
missing) and a static media/deltarho_he_kick.png hero frame.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "Molecular_DFT"))

import grid3d          # noqa: E402
import potentials3d as pot  # noqa: E402
import propagate as prop    # noqa: E402
import perturb             # noqa: E402

MEDIA = os.path.join(HERE, "media")
os.makedirs(MEDIA, exist_ok=True)


def build(quick):
    L, N = 16.0, (32 if quick else 40)
    x = (np.arange(N) - N // 2) * (L / N)
    X, Y, Z_ = np.meshgrid(x, x, x, indexing="ij")
    dx = L / N
    _, _, _, G2 = grid3d.g_vectors(L, N)
    grid = (x, X, Y, Z_, dx, G2)
    coords = (X, Y, Z_)
    soft = 0.5 * dx
    nuclei = [(2, 0.0, 0.0, 0.0, soft)]

    V_nuc = pot.nuclear_potential(X, Y, Z_, nuclei)
    psi0, occ, _ = prop.imaginary_time_ground_state(grid, V_nuc, 2, method="lda",
                                                    max_iter=400, tol=1e-9)

    n0 = prop.density(psi0, occ)
    mid = N // 2
    dt = 0.05
    T = 80.0 if quick else 140.0
    n_steps = int(round(T / dt))

    psi = perturb.dipole_kick(psi0, coords, 0.12, axis=0)      # large kick -> visible delta-rho
    frames_drho, ts, dxs = [], [], []

    def obs(p, t):
        n = prop.density(p, occ)
        frames_drho.append((n - n0)[:, mid, :].copy())          # (x, z) slice through y=0
        ts.append(t)
        dxs.append(float(np.sum(X * n) * dx ** 3))

    prop.propagate(psi, grid, n_steps, dt, occ=occ, V_nuc=V_nuc, method="lda",
                   record_every=max(1, n_steps // 120), observers={"f": obs})

    return x, np.array(ts), np.array(dxs), frames_drho


def render(out, quick):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, t, dxt, frames = build(quick)
    vmax = np.percentile(np.abs(np.stack(frames)), 99.5)
    ext = [x[0], x[-1], x[0], x[-1]]
    zoom = 5.0

    fig, (ax, axd) = plt.subplots(2, 1, figsize=(5.2, 6.4), dpi=120,
                                  gridspec_kw={"height_ratios": [3, 1]})
    im = ax.imshow(frames[0].T, origin="lower", extent=ext, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    ax.set_xlabel("x  (Bohr)"); ax.set_ylabel("z  (Bohr)")
    ax.set_title(r"He  $\delta\rho(\mathbf{r},t) = n(t) - n(0)$   (kick $\parallel x$)")
    fig.colorbar(im, ax=ax, fraction=0.046, label=r"$\delta\rho$")
    (line,) = axd.plot([], [], color="#333")
    (dot,) = axd.plot([], [], "o", ms=6, mfc="#dd8452", mec="k")
    axd.set_xlim(t[0], t[-1]); axd.set_ylim(dxt.min() * 1.2 - 1e-6, dxt.max() * 1.2 + 1e-6)
    axd.set_xlabel("t  (a.u.)"); axd.set_ylabel(r"$\langle x\rangle(t)$")
    fig.tight_layout()

    def frame(i):
        im.set_data(frames[i].T)
        line.set_data(t[:i + 1], dxt[:i + 1])
        dot.set_data([t[i]], [dxt[i]])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[:, :, :3].copy()

    hero = max(range(len(frames)), key=lambda i: np.abs(frames[i]).max())
    plt.imsave(os.path.join(MEDIA, "deltarho_he_kick.png"), frame(hero))

    try:
        import imageio
        with imageio.get_writer(out, fps=20, codec="libx264", quality=8, macro_block_size=8) as wtr:
            for i in range(len(frames)):
                wtr.append_data(frame(i))
        print(f"wrote {out}")
    except Exception as e:
        strip = os.path.join(MEDIA, "deltarho_he_kick_frames")
        os.makedirs(strip, exist_ok=True)
        print(f"imageio/ffmpeg unavailable ({e}); PNG strip -> {strip}")
        for i in range(0, len(frames), max(1, len(frames) // 12)):
            plt.imsave(os.path.join(strip, f"f_{i:03d}.png"), frame(i))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(MEDIA, "deltarho_he_kick.mp4"))
    a = ap.parse_args()
    render(a.out, a.quick)
