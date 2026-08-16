# Thermodynamics / Blackbody_Radiation — Design Plan

## Context

`Thermodynamics/Blackbody Radiation.py` existed as a loose top-level file
but turned out to be unrelated (a broken 2D particle-collision/billiards
script -- no blackbody physics, and it doesn't even run: `speedDist[i+1][]`
is invalid syntax) -- deleted rather than archived, since there's nothing
to build on.

This project derives Planck's law from statistical mechanics first
principles (cavity mode density x Bose-Einstein occupation) rather than
starting from the closed-form formula, then builds two demos on top of
the validated core. House style: own `*_Plan.md`, phased build with a
validation gate per phase, media to `media/`.

## Numerical approach

**`blackbody.py`**:
- `spectral_radiance(nu, T)`: Planck's law by frequency, `B(nu,T) =
  (2*h*nu^3/c^2) / (exp(h*nu/(k*T)) - 1)` -- derived from mode density
  `rho(nu) = 8*pi*nu^2/c^3` times mean photon energy per mode
  `h*nu / (exp(h*nu/(k*T)) - 1)` (Bose-Einstein occupation x quantum
  energy), not just typed in from memory.
- `planck_cdf_sample(T, n, rng)`: inverse-CDF Monte Carlo sampler for
  photon frequencies drawn from the (normalized) Planck distribution at
  temperature `T` -- the CDF has no closed form, so this numerically
  builds and inverts it once, then samples cheaply.
- `equilibrium_temperature(L_star, distance, albedo=0, greenhouse=0)`:
  planetary equilibrium temperature from radiative flux balance
  (absorbed stellar flux = emitted blackbody flux), with an optional
  albedo and a toy single-parameter greenhouse (partial IR opacity)
  correction.

## File layout

```
Thermodynamics/
    Blackbody_Radiation/
        Blackbody_Radiation_Plan.md
        blackbody.py
        Validation.ipynb                 # Phase 1
        Photon_Monte_Carlo.ipynb         # Phase 2
        Planetary_Energy_Balance.ipynb   # Phase 3
        media/
```

## Phases

**Phase 1 — `blackbody.py` + `Validation.ipynb`.**
Build `spectral_radiance` from the mode-density x Bose-Einstein
derivation above. Validate three classical results all fall out of the
one formula: (1) the Rayleigh-Jeans classical limit as `h*nu << k*T`
(and the UV-catastrophe divergence that limit has, which Planck's law
fixes); (2) numerically integrating `B(nu,T)` over all `nu` and solid
angle reproduces the Stefan-Boltzmann law `j = sigma*T^4`, with `sigma`
matching the CODATA value; (3) numerically locating the peak (root-find
on `dB/d(lambda) = 0` in wavelength form) reproduces Wien's displacement
law `lambda_peak * T = b`.

**Phase 2 — `Photon_Monte_Carlo.ipynb`.**
Use `planck_cdf_sample` to draw a growing number of photon frequencies
at a fixed `T`, animate the sampled histogram converging onto the
analytic `B(nu,T)` curve, and validate with a quantitative goodness-of-
fit check (e.g. chi-squared or KS test) between the large-`N` sample and
the analytic distribution.

**Phase 3 — `Planetary_Energy_Balance.ipynb`.**
Use `equilibrium_temperature` to compute equilibrium temperatures for
real solar-system bodies from actual stellar/orbital parameters, and
validate against measured values -- the bare-rock (no greenhouse)
prediction should undershoot Earth's actual ~288 K (undershooting to
around the textbook ~255 K figure), and adding the toy greenhouse term
should closes most of that gap; Venus's much larger, real greenhouse
effect should show an even starker bare-rock/actual mismatch.

## Progress

- [x] Phase 1 — `blackbody.py`, `Validation.ipynb`. Rayleigh-Jeans limit
  matched to `5.0e-4` relative error by `h*nu/kT=0.001`, shrinking
  monotonically. Stefan-Boltzmann integration matched CODATA `sigma` to
  `2.0e-6` - `2.7e-10` relative error across 300-10000 K. Wien's
  displacement law: `lambda_peak*T` constant to `2.7e-8` relative error
  across the same range.
- [ ] Phase 2 — `Photon_Monte_Carlo.ipynb`
- [ ] Phase 3 — `Planetary_Energy_Balance.ipynb`
