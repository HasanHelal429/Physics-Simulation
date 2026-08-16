# Phase 7 investigation: resolution-dependent shedding amplitude — PAUSED, unresolved

## Status

Phase 7 (`Cylinder_Vortex_Shedding.ipynb`) is **not complete**. A full run at
cells/D=12 finished and all its checks passed, but that run itself surfaced
a real problem (see below), and a follow-up investigation at the plan's
target resolution (cells/D=25) surfaced a second, more serious, unresolved
one. Paused here at the user's request rather than continuing to spend
compute chasing it.

## Timeline of what happened

1. **First full run** (cells/D=12, no cylinder offset) completed and all
   checks passed, but inspecting the actual probe signal and animation
   afterward showed the "shedding" was a linear instability still growing
   from floating-point noise (~1e-12 amplitude) — never reached a real,
   finite-amplitude vortex street in the simulated window. The vorticity
   animation showed a static symmetric wake, not a von Kármán street. The
   checks passed only because they measured *a* nonzero peak frequency and
   *a* plausible-looking Strouhal number from that infinitesimal linear
   mode, not because real shedding was observed.

2. **Standard fix applied**: offset the cylinder 0.02D off the channel
   centerline to break the exact top-bottom symmetry (a well-established
   technique — an exactly-centered cylinder with symmetric BCs can only be
   destabilized by round-off noise, taking a very long time to grow).
   `01_test_perturbation_coarse.py`, at cells/D=10: this worked great —
   amplitude grew from ~0 to a saturated ~0.41 by t~30-40, a real vortex
   street.

3. **Also, before trusting that fix**: separately found and fixed two real
   bugs in the core solver, needed regardless of the shedding question:
   - `advection.py`'s periodic-wraparound assumption, previously proven
     harmless only at *pinned* (Dirichlet-exact) boundaries, was
     contaminating real interior physics at *outflow* boundaries (a free,
     unpinned unknown). Fixed with `outflow_u_x`/`outflow_v_y` flags.
   - `operators.gradient()` wasn't mask-aware, so it computed a spurious
     pressure correction at obstacle-blocked faces (using the
     obstacle-interior's pinned p=0 as a fake neighbor), which then
     contaminated the *reported* divergence at genuinely adjacent fluid
     cells (confirmed the actual post-mask physics was fine at ~1e-9;
     only the diagnostic was lying). Fixed by threading `u_mask`/`v_mask`
     into `gradient()`.
   - Also optimized `diffusion.py`: previously reformed the sparse
     Laplacian from scratch every step (~95% of its cost, confirmed via
     profiling) despite the geometry never changing. Now precomputes CSR
     data-array positions once and does per-step matrix formation via
     plain numpy indexing. ~90x speedup on `diffuse_component` at Phase
     7's scale (250ms -> 2.75ms), verified bit-for-bit identical to the
     old approach across several BC/obstacle configurations. This is why
     re-running Phase 7 at a much finer resolution suddenly became
     affordable (~25 min instead of ~2 hours at cells/D=25).

4. **Tried to re-run properly at cells/D=25** (the plan's actual target
   resolution, now affordable) with the same 0.02D offset fix.
   `02_test_perturbation_fine.py` / `.log`: amplitude grew initially
   (0 -> ~0.021 by t~15-20) then **decayed** back down to a small, stable
   ~0.004-0.006, never reaching anything like the coarse-resolution ~0.4.

5. **Ruled out blockage and wall type** as the cause:
   `03_test_blockage_walltype.py` / `.log` — at coarse resolution
   (cells/D=10), the *same* blockage+free-slip combination, a much lower
   blockage, and no-slip walls *all* show robust growth to ~0.4. So the
   discrepancy isn't about confinement or wall type; it's specifically
   about resolution.

6. **Ruled out "perturbation too small"**: `04_test_bigger_offset_fine.py`
   / `.log` — tried offsets of 0.1D and 0.2D (5-10x larger, and 0.2D is a
   substantial fraction of the diameter) at cells/D=25. Both *still*
   settle into a small stable oscillation (~0.02), not a large saturated
   vortex street. Amplitude is essentially independent of how hard the
   flow is kicked, which rules out an under-powered seed as the
   explanation.

## Open question

Coarse resolution (cells/D=10-12) robustly produces large-amplitude
(~0.4), clearly saturated shedding regardless of offset size, blockage, or
wall type. Fine resolution (cells/D=25, the plan's own target) robustly
produces a small-amplitude (~0.02) stable oscillation regardless of how
hard it's perturbed. Since Re=100 cylinder shedding is one of the most
robustly-documented phenomena in CFD (large, unambiguous, easily visible
in any correctly-implemented incompressible solver), the small-amplitude
fine-resolution result is very unlikely to be the "more correct, less
numerically diffusive" answer just because it's the finer grid — but
which side is the artifact, and why, isn't established yet. Two live
hypotheses, not yet distinguished:

1. The coarse-resolution result is the artifact: a 10-12-cell-per-diameter
   circle is really a rough faceted polygon, and sharp/faceted bluff
   bodies shed more readily than smooth round ones — so the "shedding" at
   coarse resolution could be a staircase-geometry artifact rather than
   physically accurate for a cylinder.
2. There's a real bug or excess numerical damping specific to finer
   resolution with an obstacle in the loop, not yet located.

## Suggested next steps (not yet done)

- Compare oscillation *frequency* (not just amplitude) between the coarse
  and fine cases — same frequency at very different amplitude would point
  toward hypothesis 1 (same mode, different saturation level); a
  different frequency would point toward two qualitatively different
  phenomena.
- Check an intermediate resolution (cells/D=16-18) to see whether amplitude
  transitions smoothly or discontinuously between the two regimes.
- Look directly at the vorticity field (not just the point probe) in the
  fine-resolution case to see whether there's any asymmetric wake
  structure at all, even a weak one.

## Solver state

All the bug fixes and the diffusion speedup above are real, verified
improvements independent of the shedding-amplitude question, and remain
in place (`advection.py`, `operators.py`, `diffusion.py`). Phases 1-6 were
fully re-validated after each change with no regressions. Only Phase 7's
own notebook/result is not yet trustworthy.
