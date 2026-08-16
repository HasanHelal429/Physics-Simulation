"""Single-cluster Wolff algorithm for the 2D periodic Ising model (B=0,
J=1), batched over independent replicas like the rest of ising.py.

Growing a cluster is inherently a variable-size graph search, which doesn't
vectorize across space the way a Metropolis sweep does. The standard trick
used here: decide every bond's "active" status (same-sign neighbors, kept
with probability p=1-exp(-2*beta)) once per step for the whole lattice --
this fixes an ordinary bond-percolation graph independent of growth order --
then find the connected component containing a seed site by iterated
dilation (flood fill via repeated shift-and-OR), which *is* vectorizable
and runs identically across the whole batch of replicas at once. Once the
cluster is found, it is flipped unconditionally; no separate accept/reject
step is needed since the bond probabilities already embed the correct
Metropolis-consistent acceptance (this is the standard Wolff result).
"""
import torch

import ising


def wolff_step(lattice, beta, max_iters=None):
    """One single-cluster Wolff update, B=0. `beta` is a per-replica tensor
    of shape (batch,); `lattice` is (batch, 1, L, L)."""
    batch, _, L, _ = lattice.shape
    if max_iters is None:
        max_iters = 2 * L + 4
    p = 1.0 - torch.exp(-2.0 * beta).view(-1, 1, 1, 1)

    same_right = (lattice == torch.roll(lattice, shifts=-1, dims=2))
    same_down = (lattice == torch.roll(lattice, shifts=-1, dims=3))
    bond_right = same_right & (torch.rand_like(lattice) < p)
    bond_down = same_down & (torch.rand_like(lattice) < p)

    cluster = torch.zeros_like(lattice, dtype=torch.bool)
    i0 = torch.randint(0, L, (batch,))
    j0 = torch.randint(0, L, (batch,))
    cluster[torch.arange(batch), 0, i0, j0] = True

    for _ in range(max_iters):
        grow = (cluster
                | torch.roll(cluster & bond_right, shifts=1, dims=2)
                | (torch.roll(cluster, shifts=-1, dims=2) & bond_right)
                | torch.roll(cluster & bond_down, shifts=1, dims=3)
                | (torch.roll(cluster, shifts=-1, dims=3) & bond_down))
        if torch.equal(grow, cluster):
            break
        cluster = grow

    new_lattice = torch.where(cluster, -lattice, lattice)
    return new_lattice


def run_wolff(lattice, beta, n_steps=1, callback=None):
    """Run n_steps Wolff cluster updates. No anti-trap safety net is needed
    here -- the absorbing-orbit pathology found in ising.run_sweeps is
    specific to synchronous *local* spin-flip updates; a single-cluster
    flip is a large, globally-coupled move that doesn't share that failure
    mode (a fixed dE=0-everywhere configuration has no special significance
    to Wolff's bond-percolation construction)."""
    for s in range(n_steps):
        lattice = wolff_step(lattice, beta)
        if callback is not None:
            callback(s, lattice)
    return lattice
