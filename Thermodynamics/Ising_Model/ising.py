"""2D Ising model on a periodic square lattice, GPU-accelerated (torch).

Lattices are stored as (batch, 1, L, L) tensors of +/-1 spins, so many
independent (temperature, field) points can be simulated in parallel.
Hamiltonian convention (J=1 throughout):

    H = -sum_<ij> S_i*S_j - B*sum_i S_i

used consistently for both the Metropolis accept/reject decision and the
total-energy measurement, so results compare directly to textbook results
(T_c, critical exponents) with no hidden normalization factors.
"""
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_NN_KERNEL = torch.tensor([[0., 1., 0.],
                            [1., 0., 1.],
                            [0., 1., 0.]])


def init_lattice(batch, L, p_up=0.5):
    """Random +/-1 lattice, shape (batch, 1, L, L); p_up is the initial
    fraction of up spins."""
    r = torch.rand(batch, 1, L, L, device=device)
    return torch.where(r < p_up, 1.0, -1.0)


def neighbor_sum(lattice):
    """Sum of the 4 nearest neighbors at each site, periodic BC."""
    kern = _NN_KERNEL.to(lattice.device).view(1, 1, 3, 3)
    padded = F.pad(lattice, (1, 1, 1, 1), mode='circular')
    return F.conv2d(padded, kern)


def local_field_energy(lattice, B=0.0):
    """h_i = S_i*(sum_nn(S_j) + B) -- the part of H that depends on site i.

    Flipping site i changes the energy by dE_i = 2*h_i (independent of any
    global double-counting convention, since only bonds touching i change).
    """
    return lattice * (neighbor_sum(lattice) + B)


def total_energy(lattice, B=0.0):
    """Total energy per lattice in the batch, H = -sum_<ij> S_i*S_j - B*sum_i S_i."""
    nn = neighbor_sum(lattice)
    interaction = -0.5 * torch.sum(lattice * nn, dim=(1, 2, 3))
    field = -B * torch.sum(lattice, dim=(1, 2, 3))
    return interaction + field


def magnetization(lattice):
    """Total magnetization per lattice in the batch (sum of spins, not per-site)."""
    return torch.sum(lattice, dim=(1, 2, 3))


def _checkerboard_mask(L, color, device):
    ii, jj = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing='ij')
    return ((ii + jj) % 2 == color).view(1, 1, L, L)


def metropolis_substep(lattice, beta, B, color):
    """Propose flipping every site of one checkerboard color simultaneously.

    Sites of a single color are mutually non-adjacent (neighbors always
    differ in (i+j) parity), so their pre-flip local fields are unaffected
    by each other and a simultaneous update is exact, not an approximation.
    `beta` is a per-batch-element tensor of shape (batch,).
    """
    L = lattice.shape[-1]
    mask = _checkerboard_mask(L, color, lattice.device)
    dE = 2.0 * local_field_energy(lattice, B)
    beta_b = beta.view(-1, 1, 1, 1)
    accept_prob = torch.exp(-beta_b * dE)
    rand = torch.rand_like(lattice)
    flip = mask & ((dE <= 0) | (rand < accept_prob))
    return torch.where(flip, -lattice, lattice)


def sweep(lattice, beta, B=0.0):
    """One full lattice sweep = both checkerboard colors, so every spin gets
    exactly one flip opportunity."""
    lattice = metropolis_substep(lattice, beta, B, color=0)
    lattice = metropolis_substep(lattice, beta, B, color=1)
    return lattice


def run_sweeps(lattice, beta, B=0.0, n_sweeps=1, stuck_window=8, callback=None):
    """Run n_sweeps checkerboard sweeps, with a safety net against a real
    (if rare) pathology of *any* fully synchronous parallel spin update:
    a configuration where every site's local field is exactly zero has
    dE=0 everywhere, so the exact Metropolis rule force-accepts every
    proposed flip with probability 1, independent of temperature -- no
    randomness is ever invoked, so the chain cycles in a temperature-
    independent absorbing orbit forever (confirmed empirically: this
    survives reversing the checkerboard color order and switching to a
    4-way random-offset sublattice split, since neither changes the fact
    that dE=0 for every site given the current configuration).

    The trap's specific signature is that the *configuration* keeps
    changing every sweep (every site is force-flipped) while the *energy*
    stays bit-identical -- unlike a genuinely frozen cold configuration
    (e.g. sitting at the ground state at low T), where both the energy
    and the configuration itself stop changing because moves are being
    correctly rejected, not force-accepted. Checking energy alone would
    misfire on ordinary low-T equilibrium; requiring the configuration to
    also be actively changing makes the detector specific to the actual
    pathology. On detection, that replica gets one random single-spin
    flip (always accepted) to break the exact symmetry, after which
    normal Metropolis dynamics resumes -- the standard trick for a
    detectable, measure-small absorbing set: the kick negligibly perturbs
    the stationary measure while restoring ergodicity.

    `callback(step, lattice)`, if given, is invoked once per sweep for
    recording observables.
    """
    batch, L = lattice.shape[0], lattice.shape[-1]
    prev_E = total_energy(lattice, B)
    prev_lattice = lattice.clone()
    stuck_count = torch.zeros(batch, dtype=torch.long, device=lattice.device)
    for s in range(n_sweeps):
        lattice = sweep(lattice, beta, B)
        E = total_energy(lattice, B)
        energy_same = (E == prev_E)
        config_changed = (lattice != prev_lattice).any(dim=(1, 2, 3))
        trap_signal = energy_same & config_changed
        stuck_count = torch.where(trap_signal, stuck_count + 1, torch.zeros_like(stuck_count))
        prev_E = E
        prev_lattice = lattice.clone()
        trapped = torch.nonzero(stuck_count >= stuck_window, as_tuple=True)[0]
        if len(trapped) > 0:
            ii = torch.randint(0, L, (len(trapped),))
            jj = torch.randint(0, L, (len(trapped),))
            lattice[trapped, 0, ii, jj] *= -1
            stuck_count[trapped] = 0
            prev_E = total_energy(lattice, B)
            prev_lattice = lattice.clone()
        if callback is not None:
            callback(s, lattice)
    return lattice
