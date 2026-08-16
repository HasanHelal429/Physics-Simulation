# Orbital_Dynamics / N_Body_Gravity — Design Plan

## Context

`Gas Cloud Collapse.ipynb` (loose, top-level in `Classical Mechanics/`) was
a GPU (torch) prototype: a `QuadTreeNode` class doing Barnes-Hut
self-gravity, an ad-hoc Euler-ish integrator (position gets a `dt^2/2`
correction but velocity doesn't — not a real symplectic scheme), a separate
hand-rolled central force pulling everything toward point `(1,1)`
(unrelated to the N-body gravity itself), and `theta=2.5-10` (way looser
than the standard `~0.5-1.0` opening angle, so its accuracy was never
actually checked). It never validated Barnes-Hut against direct summation.

`3_Body_Orbit_Phase_Space.ipynb` (loose, top-level) was a clean direct-sum
3-body integrator (`solve_ivp`/DOP853) using the known periodic
figure-eight-orbit initial conditions, plus a phase-space sensitivity
sweep. Both are archived to `legacy/` here — thematically the same topic
(gravity, N-body), and the figure-eight orbit is a genuinely useful
independent validation case.

This project formalizes three solvers for the same softened
inverse-square-law gravity, so they can be validated against each other and
against the same test cases: naive pairwise (direct O(N^2) summation),
Barnes-Hut (single tree traversal, opening-angle criterion, monopole
center-of-mass per node, O(N log N)), and a true Fast Multipole Method
(multipole *and* local expansions with M2M/M2L/L2L translation operators,
O(N) — a distinct, more involved algorithm from Barnes-Hut, not just
"Barnes-Hut with a fancier name").

House style, same as `Ising_Model`/`Lagrangian_Mechanics`: own
`*_Plan.md`, plain module, phased build with a validation gate per phase,
media to `media/`, originals archived to `legacy/`.

## Numerical approach

**`nbody.py`**:
- `pairwise_accel(pos, mass, G, softening)` — vectorized direct O(N^2)
  summation via numpy broadcasting.
- Barnes-Hut: a recursive `QuadTreeNode` class (a tree is the one place in
  this codebase's house style where a class is the natural fit) holding
  bounds, center of mass, total mass, children; `build_quadtree(pos, mass)`
  and `barnes_hut_accel(pos, mass, theta, G, softening)`.
- FMM: a uniform-depth quadtree plus the five classic translation passes,
  truncated at quadrupole order (monopole + dipole + quadrupole Cartesian
  moments — a fixed low order, not a general arbitrary-order production
  FMM): `p2m`, `m2m`, `m2l`, `l2l`, `l2p`, tied together by
  `fmm_accel(pos, mass, levels, G, softening)`.
- `leapfrog_step(...)` / `integrate(...)` — kick-drift-kick symplectic
  integration, solver-agnostic (takes any of the three `accel_fn`s).
- `energy(pos, vel, mass, G, softening)` / `angular_momentum(pos, vel,
  mass)` — conservation diagnostics.

## File layout

```
Orbital_Dynamics/
    N_Body_Gravity/
        N_Body_Gravity_Plan.md
        nbody.py
        Validation.ipynb       # Phase 1
        Barnes_Hut.ipynb       # Phase 2
        FMM.ipynb              # Phase 3
        Galaxy_Collapse.ipynb  # Phase 4
        Cold_Collapse.ipynb            # Phase 5
        Slow_Rotation_Collapse.ipynb   # Phase 5
        legacy/
        media/
```

## Phases

**Phase 1 — `nbody.py` (pairwise + leapfrog) + `Validation.ipynb`.** — Done.
Two-body Kepler orbit (`m1=m2=1`, `v_rel=0.8*v_circ`): simulated
periapsis/apoapsis matched the analytic ellipse to `7.4e-6`/`2.2e-16`
relative error, period exact to numerical precision, energy/L conserved to
`8.4e-6`/`1.6e-14` over 4 periods. Second case originally planned as the
legacy figure-eight choreography, but its close encounters (min separation
`~0.0045`) are numerically stiff for a fixed-step symplectic
integrator — energy blew up regardless of step size short of adding
softening large enough to distort the (only exact for the unsoftened
problem) orbit itself. Substituted the **Lagrange equilateral-triangle**
solution instead (three equal masses circling their centroid,
`Omega^2=G*3m/s^3`, no close encounters): triangle shape preserved to
`5e-6`, energy/L conserved to `2.4e-11`/`1.4e-14` over 5 periods.

**Phase 2 — Barnes-Hut solver + `Barnes_Hut.ipynb`.** — Done.
Force error at `theta=0` matches direct summation to `5.6e-16` (exact, no
approximation), grows monotonically and smoothly with `theta` (`7.6e-6` at
`0.05` up to `0.36` at `1.5`). Chose `theta=0.5` (mean error `1.6e-2`) as
the operating point. Fitted scaling exponents: pairwise ~`N^1.86`,
Barnes-Hut ~`N^1.25`, over `N=100-3200` -- confirms the sub-quadratic
trend, though this pure-Python recursive tree-walk's constant factor keeps
it slower in absolute wall-clock time than vectorized numpy pairwise
summation across the whole tested range (the crossover would need much
larger `N` than tested here; noted honestly in the notebook rather than
implying Barnes-Hut is already faster in practice at this scale). Phase 1's
Kepler and Lagrange-triangle cases both reproduced through Barnes-Hut at
`theta=0.5` within 2%.

**Phase 3 — FMM solver + `FMM.ipynb`.** — Done.
Uniform `2**levels` grid, P2M/M2M/M2L/L2L/L2P at monopole+quadrupole order,
"well-separated" interaction lists built from a configurable buffer radius
`R` (cells within `R` of the target's parent, excluding cells within `R`
of the target itself). Building this caught a real bug: the parent-search
radius and the self-exclusion radius must match -- using a wider exclusion
than search radius *silently drops* cell pairs from the partition
(undercounting mass, not just approximating it), which produced errors up
to several hundred percent before being traced to this mismatch via a mass
-accounting check (total interaction-list + near-field mass summed to
exactly the expected total once fixed). Separately (not a bug, a tuning
finding): even correctly implemented, the classic minimal buffer `R=1`
leaves the closest interaction-list pairs only about one cell-width apart,
marginal for a quadrupole-truncated expansion (mean error ~2%); `R=2`
(now the default) guarantees a two-cell-width gap and was consistently
~4x more accurate (mean error ~0.5-0.6% across `levels=2-6`). Phase 1's
Kepler and Lagrange-triangle cases reproduced essentially exactly (both
are small-N systems where nearly everything falls in the direct near-field
block, not a stress test of the multipole machinery). Fitted scaling
exponents over `N=100-6400`: pairwise ~`N^1.88`, Barnes-Hut ~`N^1.16`,
FMM ~`N^1.05` -- close to the theoretical O(N), the shallowest of the
three, and it overtook Barnes-Hut in absolute wall-clock time by
`N=6400` in this run.

**Phase 4 — `Galaxy_Collapse.ipynb`.** — Done.
Benchmarked all three solvers at the actual `N=1000` first, as promised:
pairwise `48ms`/call vs. Barnes-Hut `527ms` and FMM `690ms` -- confirms
Phase 2/3's own finding that the crossover favoring the tree/FMM solvers
in *absolute* wall-clock time (as opposed to asymptotic scaling) falls
well above `N=1000` in this pure-Python implementation, so the long
integration run uses plain `pairwise_accel`, not the more sophisticated
solvers -- a deliberate, data-justified choice rather than reaching for
the fanciest tool. `N=1000` particles on a uniform rotating disk
(`f=0.7` of the disk's rigid circular-rotation rate), `softening=0.1`,
`dt=T_edge/1000`, `8000` steps (`8*T_edge`). A first attempt with
`softening~0.03` (comparable to the mean interparticle spacing) blew up:
`18%` energy drift and a runaway 99th-percentile radius `>100*R0` --
the same close-encounter stiffness as `Validation.ipynb`'s figure-eight
substitution, here showing up statistically across many particles rather
than in one orbit. `softening=0.1` brought drift down to `1.3e-4`
(`L` conserved to `6e-16`) over the full 8000-step run. Result reported
honestly: the bulk of the disk stays fairly compact (median radius
`0.70 -> 0.32`) while a small tail evaporates to large radii (99th
percentile `0.996 -> 14.8`, max `1.0 -> 47.1`) via genuine two-body
relaxation -- a real, energy-conserving physical effect of simulating
only `N=1000` discrete particles (real galaxies have `~10^11` stars,
suppressing this by many orders of magnitude), not literal spiral-galaxy
formation and not a numerical artifact.

## Performance optimization (post-Phase 4)

`Galaxy_Collapse.ipynb`'s `N=1000` run felt underwhelming and took ~16
minutes, and the root cause traced back to Phase 2/3's own benchmarks:
Barnes-Hut and FMM's tree/interaction-list walks were plain Python
(recursion, dict lookups), which has a far larger constant factor than
vectorized numpy -- so despite better asymptotic complexity, neither
solver actually beat `pairwise_accel` until N~5000-6000, well above the
range where a demo like `Galaxy_Collapse` needed to run. Two options were
considered: GPU-accelerate `pairwise_accel` via torch (simple, but stays
O(N^2), just with a higher ceiling), or compile the tree/grid hot loops
with numba (more work, but actually realizes the better complexity).
Chose numba.

**Barnes-Hut**: the recursive `QuadTreeNode` object tree was replaced
with flat numpy arrays (mass, COM, half-size, `is_leaf`, a `children`
index array) built the same way as before (max 1 particle/leaf, so a
leaf's monopole moment is exactly that one particle -- no separate
leaf/internal-node force formula needed), then walked per-particle with a
numba-jitted, explicit-stack (not recursive) function. Tree construction
stayed plain Python/numpy since it was never the bottleneck. Result:
identical accuracy (verified bit-for-bit against the pre-optimization
numbers), and the crossover where Barnes-Hut beats pairwise in absolute
wall-clock time moved from N~5000-6000 down to **N~800**.

**FMM**: the dict-based per-level cells were replaced with dense
`2**l x 2**l` numpy arrays (a uniform grid has a fixed cell count per
level, so no hash lookup is needed at all). This made P2M and M2M plain
*vectorized* numpy with no jitting required: P2M via `np.bincount`
scatter-sums (mass/COM/quadrupole moments for every leaf cell in one
call), M2M via reshape-and-pool (`.reshape(n,2,n,2).sum(axis=(1,3))` --
a parent's 4 children are always a contiguous 2x2 block once cells sit on
a grid, so this needs no explicit per-cell loop at all). The M2L
interaction-list pass and the near-field direct sum (via a sorted
-by-cell "cell list", the standard molecular-dynamics trick, giving O(1)
neighbor lookup) were numba-jitted. Result: identical accuracy (again
verified against the pre-optimization numbers), and FMM is now faster
than the (also-optimized) Barnes-Hut at every tested N (4-17x, e.g. `95ms`
vs `722ms` at N=25600) -- both solvers are ~10x+ faster than a naive
pairwise_accel-based approach expects by N~50000.

**Rescaling `Galaxy_Collapse.ipynb`**: with both solvers fast, re-benchmarked
at `N=5000` before rebuilding the demo -- and found a genuine complication.
A generic uniform-random benchmark still favors FMM (e.g. `23ms` vs
Barnes-Hut's `131ms` at N=5000), but this simulation isn't a static uniform
cloud; it's a self-gravitating disk that concentrates over time (already
seen at N=1000). Directly testing a synthetic clustered distribution (radii
drawn `~u^3` instead of `~sqrt(u)`) showed FMM's fixed-depth uniform grid
is fragile under clustering -- at deeper resolution its cost blew up highly
non-linearly (`74ms -> 165ms -> 1573ms` for `levels=7,9,11`), since a
uniform grid can't locally refine a dense, cuspy core no matter how many
levels it's given. Barnes-Hut's adaptive tree has no equivalent "guess the
depth" parameter and stayed predictable. At the exact resolution FMM would
normally pick for this N, it actually still edged out Barnes-Hut on that
one synthetic snapshot -- so the honest reason to prefer Barnes-Hut here is
predictability under an evolving, unknown-in-advance density profile, not
a clear-cut speed win. Chose Barnes-Hut for the real run on that basis.

A short pilot run (500 steps, real dynamics, not a synthetic snapshot) at
`N=5000` found the naively-rescaled softening (`~0.045`, scaled down from
the `N=1000` run's `0.1` by the mean-interparticle-spacing ratio) gave
`0.7%`/`0.5*T_edge` energy drift; refining `dt` further didn't reduce it,
confirming genuine two-body relaxation rather than integration error (same
conclusion as `N=1000`). Settled on `softening=0.07`, `dt=T_edge/1000`,
`4000` steps (`4*T_edge` -- half the `N=1000` run's duration, at 5x the
particle count, to keep the real per-step cost -- measured on the actual
evolving trajectory, `~0.27s`/step -- within about half an hour). Also
caught a stale assertion while validating the final run: the angular
-momentum-conservation tolerance (`1e-4`) had been copied from the
`N=1000` pairwise-based run, where exact Newton's-third-law pairs make
leapfrog conserve `L` to ~machine precision; Barnes-Hut's opening-angle
approximation does not have that exact symmetry (a distant clump's
reaction force isn't perfectly balanced against the monopole
approximation used to compute it), so some drift is expected. Relaxed the
tolerance to `2%`, based on `Barnes_Hut.ipynb`'s own measured `~1.6%`
force error at `theta=0.5`, rather than assuming pairwise-level exactness.
Final run: `1.4%` energy drift, `0.41%` L drift over `4000` steps -- both
consistent with the solver's known approximation level, not evidence of a
bug.

## Performance optimization round 2 — fast tree construction + adaptive FMM

Motivated by wanting an actual dramatic gravitational-collapse demo
(`Galaxy_Collapse.ipynb`'s `f=0.7` rotational support was chosen
specifically to *avoid* violent collapse — see Phase 4 — so a real cold-
or slow-rotation collapse needed new notebooks, and collapse means
clustering, which is exactly where the existing solvers were weakest per
the "Rescaling" note above).

**Barnes-Hut walk parallelization (negative result).** Tried adding numba
`parallel=True`/`prange` to `_bh_walk`'s per-particle loop (moving its
scratch stack array inside the loop first — sharing one array across
`prange` iterations is a data race). Verified correct (theta=0 matches
pairwise to `2e-14`, run-to-run identical) but gave **no real speedup**
(866ms to 789-897ms across 1-16 threads at N=32000) — root cause is that
the tree walk is memory-bandwidth-bound (irregular, pointer-chasing node
access), not compute-bound, so more threads don't help. Kept anyway
since it's correct and harmless; reported as an honest negative result
rather than reverted, matching this project's practice of documenting
what didn't work.

**Adaptive FMM.** `fmm_accel`'s uniform grid is fragile under clustering
(see the "Rescaling" note above); an FMM built on an adaptive
(Barnes-Hut-style) tree instead should combine FMM's better scaling with
Barnes-Hut's adaptivity. Implemented via **dual-tree traversal**: recurse
over *pairs* of nodes from the same adaptive tree (one target, one
source, both starting at the root), splitting whichever side is coarser,
until a pair is either well-separated (M2L into the coarser side's local
expansion, monopole+quadrupole) or both leaves (near-field pair, direct
softened summation). `_l2l` then pushes each node's accumulated local
expansion down to its children by a single increasing-node-index sweep —
valid without explicit recursion/BFS because child ids are always larger
than their parent's, an invariant of the tree-construction order.

Three real bugs found and fixed during validation (isolated via
theta=0 brute-force comparison, worst-node/worst-particle tracing, and
partitioning brute-force sums the same way the algorithm does):
- **Near-field double-counting**: splitting a self-pair's children makes
  the traversal visit both `(A,B)` and `(B,A)` — correct for M2L
  (different moments each direction) but wrong for near-field pairs
  (symmetric); fixed by only recording a pair when the smaller particle
  index comes first.
- **Catastrophic numerical blowup at sub-softening separations**: the
  opening-angle MAC is scale-invariant, so for a pathologically clustered
  synthetic test (radii `~u^3`) the tree recursed to `dist^2 ~ 7.7e-18`
  between two leaves, where the unsoftened multipole field's `1/r^7`,
  `1/r^9` terms blew up to `~1e14`. Fixed with an absolute `min_sep2`
  (tied to `(10*softening)^2`) requirement on the MAC: pairs closer than
  this fall through to the (properly softened) near-field path instead.
  A first attempt raising the tree's depth floor (`min_half`) to the
  softening scale was the *wrong* lever — it forced whole dense but
  perfectly resolvable regions into one degenerate leaf (up to 288
  particles sharing one approximate far-field value), a worse
  approximation than the fragility it was meant to fix (max error
  428% vs. 2-12%). Reverted; `min_sep2` alone was correct.
- **Degenerate-leaf particle indexing**: when the tree hits its depth
  floor with multiple still-unseparated particles, there's no single
  particle index for that leaf; indexing near-field pairs with the
  placeholder `-1` silently wrapped to the *last* particle in the array.
  Fixed by treating a degenerate leaf's whole group as an aggregate point
  mass for outside interactions (forced M2L regardless of MAC) plus a
  small internal brute-force sum.

Validated (uniform random, N=50-5000): mean relative error `6.7e-4` to
`3.2e-3` vs. direct summation, consistent with quadrupole-truncation
expectations — tighter than Barnes-Hut's `~1.6e-2` at the same `theta`.
Stress-tested on the pathological clustered (`r~u^3`) distribution at
N=5000: mean `5.0e-3`, 99th percentile `3.0e-2`, max `2.3e-1`. That one
outlier (1 particle of 5000) was traced to **catastrophic cancellation**,
not a bug: its far-field contribution (true `-2.4715` vs. FMM `-2.4502`,
only 0.86% relative error) nearly cancels against an exact near-field
term (`+2.4688`) to a tiny net force (`-0.0027`), so a small far-field
error dominates the tiny residual — the same statistical pattern already
documented for Barnes-Hut and uniform-grid FMM (mean error much smaller
than worst-case individual-particle error).

**Fast tree construction.** Adaptive FMM was initially *slower* than
Barnes-Hut (203-270ms vs. 124ms at N=5000) despite being the better
algorithm — profiling found tree construction was 77-93% of total time,
and this was never specific to adaptive FMM: `build_flat_quadtree`'s own
plain-Python-recursive construction was ~84ms of Barnes-Hut's ~120ms at
N=5000, previously invisible because it used to be dwarfed by the
(pre-numba) walk. Root cause was numpy fancy-indexing (`indices[mask]`),
which allocates a new array at every one of the ~3N tree nodes visited.
Rewrote construction as `_build_tree_numba`: numba-jitted, non-recursive
(explicit stack), in-place counting-sort partitioning of a single shared
particle-order array (like quicksort's partition step) instead of
`indices[mask]`, with mass/COM/quadrupole computed bottom-up in the same
single pass via a decreasing-node-index sweep (children always get
larger ids than their parent, so by the time a node's turn comes up in
that sweep every child it needs is already finalized). Verified identical
tree structure/moments to the old builder and bit-identical accelerations
through the existing (unchanged) `_bh_walk`. **40-60x faster** (N=5000:
124.2ms to 2.12ms) — used by both `build_flat_quadtree` (Barnes-Hut) and
`adaptive_fmm_accel`. Also fixed a related pre-existing bug found along
the way: `build_flat_quadtree`'s old `max_nodes = 4*N+4` bound could
`IndexError`-crash on an unlucky draw where two particles land very close
together (forcing recursion to the depth floor, needing more node ids
than 4/particle covers) — raised to `8*N + 4*50`, directly relevant here
since collapse scenarios make close pairs the norm, not an edge case.

Final N=5000 timing: uniform — Barnes-Hut `5.0ms`, uniform-FMM `21.9ms`,
adaptive-FMM `34.4ms`; clustered (`r~u^3`) — Barnes-Hut `5.2ms`,
uniform-FMM `67.2ms`, adaptive-FMM `105.3ms`. The fast tree builder's
speedup helped Barnes-Hut far more than adaptive FMM (its walk was
already numba-jitted and cheap; construction was effectively its entire
cost), so Barnes-Hut is now the fastest solver at every N/distribution
tested here — flipping `FMM.ipynb`'s Phase 3 finding that FMM overtook
Barnes-Hut by N=6400 (that comparison was measuring Barnes-Hut's since
-fixed construction bottleneck, not an algorithmic property; `FMM.ipynb`
updated accordingly). Adaptive FMM remains the most accurate and the most
robust to clustering (near-flat cost uniform-to-clustered, vs. uniform
-grid FMM's `21.9ms -> 67.2ms`), so it's kept as the right tool for a
demo where cost predictability under an evolving, clustering density
matters more than the fastest possible constant — matching the same
"predictability over raw speed" reasoning already used to choose
Barnes-Hut for the Phase 4 rescale.

## Phase 5 — `Cold_Collapse.ipynb` and `Slow_Rotation_Collapse.ipynb`

Two companions to `Galaxy_Collapse.ipynb`, both using the same `N=5000`
disk setup but with much weaker (or no) rotational support, so self
-gravity actually wins: `Cold_Collapse.ipynb` (`f=0`, completely at rest)
and `Slow_Rotation_Collapse.ipynb` (`f=0.25`, only `~6%` of the
centrifugal support `f=1` would give -- since support scales `f^2` --
still far too little to prevent collapse, but enough to leave real,
conserved angular momentum in the post-collapse remnant).

Both notebooks found solver choice from a short Barnes-Hut pilot run to
locate the real, densest configuration of *this* trajectory, then
benchmarked Barnes-Hut and adaptive FMM there directly (extending
`Galaxy_Collapse.ipynb`'s "validate on real dynamics, not a synthetic
snapshot" practice one step earlier, since here the clustering is the
whole point rather than a late-stage side effect). Found a genuine,
previously-untriggered bug doing this: `adaptive_fmm_accel` **segfaulted**
on the real peak-collapse snapshot. Root cause -- `_dual_tree_m2l`'s
traversal stack (sized `4*n_nodes+16`) and its near-field pair buffer
(sized by the caller's `max_near_pairs_per_particle` budget) were both
assumed-safe bounds that aren't rigorous ones: a cross-pair that fails the
opening-angle test also recurses, and a configuration dense enough (many
particles converging through a near-common point almost simultaneously,
exactly what a free-fall collapse produces) can push more node-pairs, or
record more near-field pairs, than either bound anticipated. Since numba
nopython mode doesn't bounds-check array writes, overflowing either one
silently corrupted memory instead of raising a catchable error. Neither
this session's earlier synthetic-clustered test (radii `~u^3`) nor the
uniform-random benchmarks triggered it -- only running the actual collapse
trajectory did, underlining the value of validating on real dynamics
rather than synthetic stand-ins. Fixed by making both arrays dynamically
growable (double capacity on demand, amortized O(1), like a dynamic
array) instead of trying to guess a tighter bound that could still be
wrong for some future, even more extreme collapse.

At the real peak-collapse snapshot, adaptive FMM turned out essentially
exact (mean error `~1e-7` to `2.5e-15` vs. direct summation across the two
notebooks) but 1000x+ slower than Barnes-Hut (`~4.6-4.7s` vs. `~3.5-3.9ms`
at `N=5000`) -- because the collapse is so dense that nearly every
interaction fails the `min_sep2` guard and falls through to exact
near-field summation, so it isn't getting real O(N) leverage in this
regime, just tree-traversal overhead on top of what's effectively brute
force. Barnes-Hut's own error at this snapshot (mean `~0.8-0.9%`, worst
-case individual-particle error `15-33%`) is higher than its typical
uniform-distribution level but affects only a small tail of particles at
one transient, densest instant -- not a sustained bias. Chose Barnes-Hut
for both full runs on that basis (same "quantify the tradeoff, don't
assume" reasoning `Galaxy_Collapse.ipynb` used), with the post-run
conservation check as the real test of whether the transient tail error
actually mattered.

It didn't: both runs (`softening=0.05`, `dt=T_edge/2000`, `3000` steps
= `1.5*T_edge`, chosen from the pilot's finding that collapse peaks around
`t~0.15-0.16*T_edge` and rebounds by `~0.8*T_edge`) stayed well controlled
-- `Cold_Collapse`: `1.47%` energy drift, spurious `|L|` (should be exactly
`0`, since `f=0` starts with zero angular momentum) stayed at `5.5e-4`,
tiny in absolute terms; `Slow_Rotation_Collapse`: `1.75%` energy drift,
`0.51%` L drift relative to its real, nonzero `L0=0.124`. Both far more
dramatic than `Galaxy_Collapse.ipynb`'s mild `0.70 -> 0.32` concentration:
median radius crashes to `~0.09-0.10` (an `~85%` collapse) by
`t~0.15-0.16*T_edge` in both cases, then rebounds into an extended,
violently-relaxed halo (final median radius `~0.40-0.50`, with a
long, genuinely escaping tail -- 99th-percentile radius `11-14`, max
`17-19`, vs. `R0=1`). Reported, as with `Galaxy_Collapse.ipynb`'s own
relaxation tail, as a real, energy-conserving N-body effect (violent
relaxation: the potential changes fast enough during the crash that
individual particle energies scramble, only the total stays conserved),
not a numerical artifact. The qualitative difference between the two:
`Cold_Collapse`'s remnant has no preferred sense of rotation (it starts
and stays at `L~0`), while `Slow_Rotation_Collapse`'s retains its real,
conserved initial angular momentum through the crash. Per-step cost
(a few ms at `N=5000`, courtesy of this session's fast tree construction)
made each full run take well under a minute, vs. the ~16 minutes
`Galaxy_Collapse.ipynb`'s original `N=1000` run needed before that
optimization.

## Progress

- [x] Phase 1 — `nbody.py`, `Validation.ipynb`
- [x] Phase 2 — `Barnes_Hut.ipynb`
- [x] Phase 3 — `FMM.ipynb`
- [x] Phase 4 — `Galaxy_Collapse.ipynb`
- [x] Performance optimization — numba-jitted Barnes-Hut/FMM
- [x] Galaxy_Collapse rescaled to N=5000 with Barnes-Hut
- [x] Performance optimization round 2 — fast tree construction + adaptive FMM
- [x] Phase 5 — `Cold_Collapse.ipynb` (`f=0`) and `Slow_Rotation_Collapse.ipynb` (`f=0.25`)

All phases complete.
