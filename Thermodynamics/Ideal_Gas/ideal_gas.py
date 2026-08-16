"""
Shared 2D hard-sphere gas kinetics: spatial-hash collision detection,
elastic pairwise collisions, optional box-wall reflection with pressure
tracking, and an optional external acceleration field (gravity).

Consolidated from three near-duplicate notebooks (plain box, box + uniform
gravity, open domain + central-point gravity for an orbiting cloud) that
had each copy-pasted and locally tweaked the same collision code.
"""

import shutil

import numpy as np
import torch


def configure_ffmpeg():
    """
    Point matplotlib at a working ffmpeg binary for animation.save(writer="ffmpeg"),
    preferring a system install and falling back to the bundled imageio-ffmpeg
    binary. MP4/ffmpeg encodes dramatically faster than the Pillow gif writer
    (which re-quantizes a palette per frame) -- this matters a lot once an
    animation has more than a couple hundred frames.
    """
    import matplotlib
    if shutil.which("ffmpeg") is None:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def frame_stride(n_steps, target_frames=200):
    """Step size so an animation over n_steps samples down to about target_frames frames."""
    return max(1, n_steps // target_frames)


def find_close_pairs_xy(xy, cutoff):
    """
    Efficiently find all pairs of points within 'cutoff' distance using spatial hashing.
    xy: numpy array of shape (2, N) where xy[0] = x, xy[1] = y
    cutoff: distance threshold
    Returns: numpy array of shape (M, 2) with (i, j) index pairs where distance < cutoff
    """
    x, y = xy[0], xy[1]
    cell_size = cutoff
    ix = np.floor(x / cell_size).astype(int)
    iy = np.floor(y / cell_size).astype(int)
    grid = {}
    for idx, (gx, gy) in enumerate(zip(ix, iy)):
        key = (gx, gy)
        if key not in grid:
            grid[key] = []
        grid[key].append(idx)
    neighbor_offsets = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
    pairs = set()
    for key, indices in grid.items():
        for dx, dy in neighbor_offsets:
            neighbor_key = (key[0] + dx, key[1] + dy)
            if neighbor_key in grid:
                for i in indices:
                    for j in grid[neighbor_key]:
                        if i < j:
                            dist = np.hypot(x[i] - x[j], y[i] - y[j])
                            if dist < cutoff:
                                pairs.add((i, j))
    if pairs:
        return np.array(list(pairs), dtype=int)
    else:
        return np.empty((0, 2), dtype=int)


def get_deltad2_pairs(r, ids_pairs):
    """Squared distance for each (i, j) pair in ids_pairs; always shape (M,), including M=1."""
    dx = r[0][ids_pairs[:, 0]] - r[0][ids_pairs[:, 1]]
    dy = r[1][ids_pairs[:, 0]] - r[1][ids_pairs[:, 1]]
    return dx**2 + dy**2


def compute_new_v(v1, v2, r1, r2):
    """Elastic collision: exchange velocity components along the line of centers."""
    v1new = v1 - torch.sum((v1 - v2) * (r1 - r2), axis=0) / torch.sum((r1 - r2)**2, axis=0) * (r1 - r2)
    v2new = v2 - torch.sum((v1 - v2) * (r1 - r2), axis=0) / torch.sum((r2 - r1)**2, axis=0) * (r2 - r1)
    return v1new, v2new


def uniform_gravity(g=-900000.0, axis=1):
    """External acceleration field: constant `g` along `axis` (1=y, 0=x)."""
    def accel(r):
        a = [torch.zeros_like(r[0]), torch.zeros_like(r[1])]
        a[axis] = torch.full_like(r[axis], g)
        return a[0], a[1]
    return accel


def central_gravity(center=(1.0, 1.0), gc=0.5e5, eps=1e-6):
    """External acceleration field: attractive inverse-square pull toward `center`."""
    def accel(r):
        dx = r[0] - center[0]
        dy = r[1] - center[1]
        dist3 = torch.sqrt(dx**2 + dy**2)**3 + eps
        return -gc * dx / dist3, -gc * dy / dist3
    return accel


def _reflect_wall(r_axis, v_axis, L, hi, lo, p, track_pressure):
    """Reflect particles off whichever of the two walls on this axis are active."""
    if hi:
        hit = r_axis > L
        v_axis[hit] = -torch.abs(v_axis[hit])
        if track_pressure:
            p[hit] += 2 * torch.abs(v_axis[hit])
    if lo:
        hit = r_axis < 0
        v_axis[hit] = torch.abs(v_axis[hit])
        if track_pressure:
            p[hit] += 2 * torch.abs(v_axis[hit])


def motion(r, v, ts, dt, d_cutoff, L=1.0, accel_fn=None,
           reflect_x_lo=True, reflect_x_hi=True, reflect_y_lo=True, reflect_y_hi=True,
           track_pressure=True):
    """
    Advance an N-particle 2D hard-sphere gas for `ts` steps of size `dt`.

    r, v: (2, N) torch tensors (position, velocity), on whichever device
        they already live on.
    d_cutoff: collision distance between particle centers.
    L: box size; walls (where reflecting) sit at coordinate 0 and L.
    accel_fn: optional callable r -> (ax, ay) giving the true physical
        acceleration at each particle's position (see `uniform_gravity`,
        `central_gravity`). None = no external force.
    reflect_x_lo/x_hi/y_lo/y_hi: whether particles elastically bounce off
        each of the four individual box walls (x=0, x=L, y=0, y=L); a wall
        set to False lets particles pass straight through it (e.g. an open
        top for gas piling up under gravity, or no walls at all for an
        open orbital domain).
    track_pressure: accumulate wall momentum-transfer per step (only
        meaningful when at least one wall reflects).

    Returns (rs, vs, ps): trajectories of shape (ts, 2, N), and either a
    (ts,) pressure-proxy tensor or None if track_pressure is False.
    """
    device = r.device
    rs = torch.zeros((ts, *r.shape), device=device)
    vs = torch.zeros((ts, *v.shape), device=device)
    ps = torch.zeros(ts, device=device) if track_pressure else None
    rs[0] = r
    vs[0] = v

    for i in range(1, ts):
        close_pairs = find_close_pairs_xy(r.cpu().numpy(), cutoff=2 * d_cutoff)
        if close_pairs.shape[0] > 0:
            d2 = get_deltad2_pairs(r, close_pairs)
            mask = d2 < d_cutoff**2
            ic = close_pairs[mask.cpu().numpy()]
            if ic.shape[0] > 0:
                v[:, ic[:, 0]], v[:, ic[:, 1]] = compute_new_v(v[:, ic[:, 0]], v[:, ic[:, 1]], r[:, ic[:, 0]], r[:, ic[:, 1]])

        p = torch.zeros(r.shape[1], device=device) if track_pressure else None
        _reflect_wall(r[0], v[0], L, reflect_x_hi, reflect_x_lo, p, track_pressure)
        _reflect_wall(r[1], v[1], L, reflect_y_hi, reflect_y_lo, p, track_pressure)
        if track_pressure:
            ps[i] = torch.sum(p)

        if accel_fn is not None:
            ax, ay = accel_fn(r)
            r[0] = r[0] + v[0] * dt + 0.5 * ax * dt**2
            v[0] = v[0] + ax * dt
            r[1] = r[1] + v[1] * dt + 0.5 * ay * dt**2
            v[1] = v[1] + ay * dt
        else:
            r = r + v * dt

        rs[i] = r
        vs[i] = v

    return rs, vs, ps
