# `spectrum/spectrum.py` — line-by-line walkthrough

This document explains `spectrum/spectrum.py` in detail, with the heaviest
focus on the two things that are easiest to misread at a glance:

1. **The disorder-averaging loop** (`_disorder_average`) — how a single
   "spectrum" is actually built from many independent random draws of the
   electronic disorder.
2. **Normalization** (`_normalize_spectrum` and its callers) — what exactly
   gets divided by what, and why it happens only once, at the very end.

Line numbers refer to the current state of `spectrum/spectrum.py`.

---

## 1. Module docstring and imports (lines 1–23)

```python
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .basis import build_sector_basis
from .config import Config
from .hamiltonian import (
    build_static_hamiltonian,
    electronic_diagonal,
    electronic_masks,
)
```

- `build_sector_basis` — constructs the list of basis states for a given
  vibrational cutoff `Nv` (and the symmetry sector selected by
  `nex_target`/`jz_target`).
- `build_static_hamiltonian` — builds every part of the Hamiltonian that
  **does not depend on the electronic disorder**: vibrational + cavity
  energies, Jahn-Teller vibronic coupling, matter-cavity coupling.
- `electronic_masks` — returns two boolean-as-float arrays (`mask1`, `mask2`)
  saying which basis states have molecule 1 (resp. molecule 2) electronically
  excited.
- `electronic_diagonal` — given a disorder realization's site energies
  (`eps1_dis`, `eps2_dis`), returns the diagonal electronic-energy
  contribution for every basis state (this *is* the disorder-dependent part).

The key design idea, stated in the module docstring: the static part of `H`
is expensive to build (loops over the whole basis, operator algebra) but is
the *same* for every disorder realization and every `sigma`. So it is built
**once per `Nv`**, and only a cheap diagonal correction is added/removed for
each realization.

## 2. Lorentzian lineshape (lines 26–27)

```python
def lorentzian(x, x0, gamma):
    return (1 / np.pi) * (0.5 * gamma) / ((x - x0) ** 2 + (0.5 * gamma) ** 2)
```

Standard normalized Lorentzian centered at `x0` with full width at half
maximum `gamma`. `x` is the energy grid `E`, `x0` is an eigenvalue `En` of the
Hamiltonian, so this turns a discrete stick spectrum (energy + intensity
pairs) into a smooth curve. Because it is normalized (`∫lorentzian dx = 1`
over all real `x`), each stick contributes `intensity * 1` of total area
before the energy window is truncated to `[E_min, E_max]`.

## 3. Normalization: the valid modes (lines 30–38)

```python
VALID_NORMALIZATIONS = ("reference", "reference_area", "area", "none")

def _check_normalization(cfg: Config) -> None:
    if cfg.NORMALIZATION not in VALID_NORMALIZATIONS:
        raise ValueError(...)
```

Just a guard, called at the top of every public entry point, so a typo in
`cfg.NORMALIZATION` fails immediately instead of silently falling through to
`"none"`-like behavior deep inside `_normalize_spectrum`.

## 4. `_normalize_spectrum` — the actual division (lines 41–82)

```python
def _normalize_spectrum(
    E: np.ndarray,
    spectrum: np.ndarray,
    cfg: Config,
    reference_max: float = None,
    reference_area: float = None,
) -> np.ndarray:
    mode = cfg.NORMALIZATION
    if mode == "reference":
        if reference_max is not None and reference_max > 0:
            return spectrum / reference_max
        return spectrum
    if mode == "reference_area":
        if reference_area is not None and reference_area > 0:
            return spectrum / reference_area
        return spectrum
    if mode == "area":
        area = np.trapezoid(spectrum, E)
        if area != 0:
            return spectrum / area
        return spectrum
    if mode == "none":
        return spectrum
    raise ValueError(...)
```

This function takes a spectrum that is **already fully disorder-averaged and
broadened** — it never sees an individual realization — and divides it by one
scalar, chosen by `cfg.NORMALIZATION`:

- **`"reference"`** (the default): divide by `reference_max`, which the
  caller computes as `ref_spectrum.max()` — the **peak height** of the
  σ = 0 (zero-disorder) spectrum for this same `Nv`. This reference value is
  the *same number* for every `sigma` and every realization count in a given
  sweep, because it is computed once from the σ = 0 case and passed in. The
  effect: the σ = 0 curve itself normalizes to a peak of exactly 1, and every
  disordered curve is expressed relative to that undisturbed peak — so if
  disorder broadens/suppresses the peak, that suppression is visible (the
  disordered curve's height in the plot directly shows how much lower it is
  than the ideal case), rather than being hidden by re-normalizing every
  curve to its own peak of 1.
- **`"reference_area"`**: identical idea, but the shared reference value is
  `reference_area = trapezoid(ref_spectrum, E)` — the **integrated area**
  under the σ = 0 spectrum instead of its peak height. Same σ = 0 reference
  spectrum, different summary statistic extracted from it.
- **`"area"`**: self-normalizing — each spectrum (regardless of sigma) is
  divided by its *own* trapezoidal integral over `E`, so `∫I(E) dE = 1` for
  every curve independently. There is no shared reference here; each curve
  stands alone.
- **`"none"`**: return the raw disorder-averaged, broadened spectrum
  unchanged (whatever absolute scale falls out of the physics).

Guard clauses (`reference_max is not None and reference_max > 0`, `area !=
0`): if the reference/area couldn't be computed or is degenerate (e.g. a
totally empty spectrum), the function falls back to returning the
spectrum unmodified rather than dividing by zero or `None`.

**Why this matters physically:** in `"reference"`/`"reference_area"` mode,
comparing curves across different `sigma` at fixed `Nv` tells you how much
disorder degrades the absorption peak relative to the disorder-free case —
that comparison is only meaningful because all curves share one denominator.
If each curve were divided by its own peak/area instead, every curve would
reach the same height by construction and the disorder-induced suppression
would be normalized away — which is exactly the "corrected" behavior the
project's memory record notes was tried and then deliberately reverted.

## 5. `_prepare_Nv` — build once per `Nv` (lines 88–107)

```python
def _prepare_Nv(Nv: int, cfg: Config, show_progress: bool):
    basis_final, basis_index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
    dim = len(basis_final)
    if dim == 0:
        raise ValueError(...)
    if cfg.initial_state not in basis_index:
        raise ValueError(...)
    i0 = basis_index[cfg.initial_state]

    H = build_static_hamiltonian(Nv, basis_final, basis_index, cfg, show_progress)
    mask1, mask2 = electronic_masks(basis_final)
    return H, mask1, mask2, i0, dim
```

- `build_sector_basis(Nv, ...)`: enumerates the many-body basis states for
  vibrational cutoff `Nv`, restricted to the chosen excitation-number
  (`nex_target`) and angular-momentum (`jz_target`) sector, and gives back
  both the list of states (`basis_final`) and a state → index lookup
  (`basis_index`).
- `dim`: dimension of this sector's Hilbert space; a `dim == 0` sector is a
  configuration error, not a physical answer, so it raises rather than
  silently returning an empty spectrum.
- `i0 = basis_index[cfg.initial_state]`: the row/column index of the
  spectroscopically prepared initial state (e.g. the ground vibronic state
  with zero photons). This is the state whose transition dipole (via
  `evecs[i0, :]`) gives the absorption intensities later.
- `H = build_static_hamiltonian(...)`: builds the disorder-independent part
  of the Hamiltonian **once** for this `Nv` — this is the expensive step
  (loops over the basis assembling Jahn-Teller and matter-cavity coupling
  terms; see `hamiltonian.py`).
- `mask1, mask2 = electronic_masks(basis_final)`: precomputed boolean masks
  (as float arrays) marking which basis states have molecule 1 / molecule 2
  electronically excited. These are reused every realization to build the
  cheap disorder diagonal without re-scanning the whole basis each time.

`_prepare_Nv` is called once per `Nv`; everything it returns (`H`, `mask1`,
`mask2`, `i0`, `dim`) is then reused across every `sigma` and every disorder
realization for that `Nv`.

## 6. `_disorder_average` — the core loop (lines 110–155)

This is the heart of the disorder averaging. Full text with line-by-line
notes:

```python
def _disorder_average(
    H: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    i0: int,
    sigma: float,
    cfg: Config,
    show_progress: bool,
    desc: str,
    n_realizations: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
```

Parameters: the prebuilt static Hamiltonian `H` and masks (from
`_prepare_Nv`), the initial-state index `i0`, the disorder strength `sigma`
for *this* call (note: **not necessarily `cfg.sigma`** — the reference pass
always calls this with `sigma=0.0` regardless of what `cfg.sigma` is), and an
optional override `n_realizations` (used by the realization-count sweep to
ask for a different number of draws than `cfg.n_realizations`).

```python
    n_real = cfg.n_realizations if n_realizations is None else n_realizations
    dim = H.shape[0]
    d = np.arange(dim)
    rng = np.random.default_rng(cfg.rng_seed)
```

- `n_real`: how many disorder realizations to draw and average over.
- `dim`: Hilbert space dimension of this sector (size of `H`).
- `d = np.arange(dim)`: precomputed index array used below to address the
  diagonal of `H` as `H[d, d]` — i.e. `H[0,0], H[1,1], ..., H[dim-1,dim-1]`
  all at once, without building a full diagonal matrix.
- `rng = np.random.default_rng(cfg.rng_seed)`: a **freshly seeded** random
  generator, created fresh every time `_disorder_average` is called, always
  from the same `cfg.rng_seed`. This is important: it's why, e.g., the first
  `n` rows of a `100`-realization run are bit-for-bit identical to an
  independent `n`-realization run — the RNG always starts at the same state
  and is drawn from sequentially.

```python
    all_evals = np.empty((n_real, dim))
    all_intensity = np.empty((n_real, dim))
```

Preallocated output arrays: one row per realization, one column per
eigenstate. `all_evals[r]` will hold the `dim` eigenvalues of realization
`r`; `all_intensity[r]` will hold the corresponding transition intensities
from the initial state.

```python
    iterator = range(n_real)
    if show_progress:
        iterator = tqdm(iterator, desc=desc, leave=False, unit="real")
```

Just a progress bar wrapper when `show_progress=True`; doesn't affect the
math.

```python
    for r in iterator:
        eps1_dis = cfg.eps1 + rng.normal(0, sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, sigma)
        elec = electronic_diagonal(mask1, mask2, eps1_dis, eps2_dis)

        H[d, d] += elec
        evals, evecs = np.linalg.eigh(H)
        H[d, d] -= elec

        intensity = np.abs(evecs[i0, :]) ** 2

        all_evals[r] = evals
        all_intensity[r] = intensity
```

This is the disorder-averaging loop body, one iteration per realization
`r = 0, 1, ..., n_real-1`:

1. **Draw disordered site energies.**
   `eps1_dis = cfg.eps1 + rng.normal(0, sigma)` and similarly for
   `eps2_dis`. `cfg.eps1`/`cfg.eps2` are the *clean* (disorder-free) site
   energies (`Config.eps1`/`eps2` properties = `eps + delta1`/`delta2`).
   `rng.normal(0, sigma)` draws one Gaussian random number with mean 0 and
   standard deviation `sigma` (the disorder strength, in eV) — a fresh,
   independent draw for each molecule, each realization. When `sigma == 0`
   this draw is always exactly `0.0`, so every "realization" of the
   reference pass is identical (see §8 on why the code exploits this).

2. **Build the disorder diagonal.**
   `elec = electronic_diagonal(mask1, mask2, eps1_dis, eps2_dis)` returns
   `eps1_dis * mask1 + eps2_dis * mask2` — an array of length `dim` giving,
   for each basis state, `eps1_dis` if molecule 1 is excited in that state,
   `eps2_dis` if molecule 2 is excited, or a sum/zero depending on the masks
   (see `hamiltonian.py`, §9 below).

3. **Temporarily add the disorder to `H`'s diagonal, diagonalize, then
   remove it again.**
   `H[d, d] += elec` adds the disorder-dependent electronic energies onto
   the diagonal of the *shared* static Hamiltonian `H` — this mutates `H` in
   place rather than allocating a new full matrix each time (cheap).
   `evals, evecs = np.linalg.eigh(H)` diagonalizes the now-complete
   Hamiltonian for this realization: `evals` are the `dim` eigenvalues
   (energies of the vibronic-polaritonic eigenstates), `evecs` are the
   corresponding eigenvectors as columns (`eigh` assumes/enforces a
   Hermitian matrix, which `H` is, and returns eigenvalues in ascending
   order).
   `H[d, d] -= elec` immediately subtracts the same disorder back off,
   restoring `H` to its disorder-independent form so the *next* iteration
   (or the next `sigma`, or the next call to `_disorder_average` reusing
   this same `H` from `_prepare_Nv`) starts from a clean, unmodified matrix.
   This add/diagonalize/subtract pattern is why `H` can safely be built once
   per `Nv` and passed around by reference across the whole sigma sweep.

4. **Compute the transition intensity from the initial state.**
   `intensity = np.abs(evecs[i0, :]) ** 2` takes the `i0`-th *row* of
   `evecs` — i.e., for every eigenstate (column), the component along the
   initial basis state `i0` — and squares its magnitude. This is the
   standard absorption-intensity formula: state `i0` is the state the system
   starts in (e.g. vibronic ground state, before absorbing a photon), and
   `|⟨eigenstate|i0⟩|²` is the probability weight (oscillator strength,
   under the implicit assumption that all the transition dipole strength
   lives in this one initial-state overlap) that a transition to that
   eigenstate contributes to the spectrum.

5. **Store this realization's results.**
   `all_evals[r] = evals` and `all_intensity[r] = intensity` save the full
   set of `dim` (energy, intensity) pairs for realization `r` into the
   preallocated arrays.

```python
    return all_evals, all_intensity
```

After the loop, the function returns the full `(n_real, dim)` arrays — every
eigenvalue and every intensity from every realization, unaveraged and
unbroadened. Averaging over realizations and turning sticks into a smooth
curve happens next, in `_broaden`.

**Big picture of this loop:** disorder averaging here means literally
re-diagonalizing the full Hamiltonian `n_real` times, once per random draw of
site-energy disorder, and collecting every run's eigenvalues/intensities. The
loop does *not* average anything itself — it just accumulates raw data; the
averaging (in the statistical sense of "divide by n_real") happens in
`_broaden` below, after each realization's stick spectrum has already been
turned into a smooth curve via the Lorentzian.

## 7. `_broaden` — turn sticks into a spectrum, then average (lines 158–174)

```python
def _broaden(all_evals: np.ndarray, all_intensity: np.ndarray, cfg: Config):
    E = np.linspace(cfg.E_min, cfg.E_max, cfg.E_points)
    spectrum = np.zeros_like(E)
    for evals, intensity in zip(all_evals, all_intensity):
        for En, I in zip(evals, intensity):
            spectrum += I * lorentzian(E, En, cfg.gamma)
    n_real = len(all_evals)
    if n_real > 0:
        spectrum = spectrum / n_real
    return E, spectrum
```

- `E = np.linspace(cfg.E_min, cfg.E_max, cfg.E_points)`: the fixed energy
  grid the final spectrum will be reported on (e.g. 2000 points between 6
  and 8 eV).
- `spectrum = np.zeros_like(E)`: accumulator, starts at all zeros.
- **Outer loop** `for evals, intensity in zip(all_evals, all_intensity)`:
  iterates over realizations (each `evals`/`intensity` is one realization's
  `dim`-length arrays from `_disorder_average`).
- **Inner loop** `for En, I in zip(evals, intensity)`: iterates over every
  eigenstate within that realization, adding `I * lorentzian(E, En,
  cfg.gamma)` — a Lorentzian peak centered at that eigenvalue `En`, scaled by
  its intensity `I`, evaluated across the whole energy grid `E` — into the
  running `spectrum` total.
- After both loops, `spectrum` holds the **sum over all realizations and all
  eigenstates** of intensity-weighted Lorentzians.
- `spectrum = spectrum / n_real`: dividing by the number of realizations is
  the actual disorder *averaging* step — turning a sum over `n_real`
  independent random draws into a mean. This is the only place division by
  realization count happens; note it is completely separate from, and prior
  to, the normalization step in `_normalize_spectrum`.
- The docstring explicitly notes: **no normalization happens here** — that's
  deferred to the call site, using the σ = 0 reference.

So the two "averages" in this file are different operations, easy to
conflate:
- **Disorder average** (`_broaden`): mean over `n_real` random realizations
  of the same physical system — a genuine statistical average, dividing a
  sum by a count.
- **Normalization** (`_normalize_spectrum`): rescaling the already-averaged
  curve by one reference scalar (a peak or an area) so curves are
  comparable/interpretable — not a statistical average at all.

## 8. Public entry point: `compute_spectrum_for_Nv` (lines 293–341)

```python
def compute_spectrum_for_Nv(Nv: int, cfg: Config, show_progress: bool = True) -> Dict:
    _check_normalization(cfg)
    need_ref = cfg.NORMALIZATION in ("reference", "reference_area")
```

Validates `cfg.NORMALIZATION`, and determines whether a σ = 0 reference pass
is needed at all — `"area"` and `"none"` never need one, so that expensive
extra diagonalization pass is skipped entirely for those modes.

```python
    if _parallel_enabled(cfg):
        all_evals, all_intensity, dim = _disorder_average_parallel(...)
        if need_ref:
            ref_evals, ref_intensity, _ = _disorder_average_parallel(
                Nv, 0.0, cfg, show_progress, f"  reference(Nv={Nv})",
                n_realizations=cfg.n_realizations,
            )
    else:
        H, mask1, mask2, i0, dim = _prepare_Nv(Nv, cfg, show_progress)
        all_evals, all_intensity = _disorder_average(
            H, mask1, mask2, i0, cfg.sigma, cfg, show_progress, f"  disorder(Nv={Nv})"
        )
        if need_ref:
            ref_evals, ref_intensity = _disorder_average(
                H, mask1, mask2, i0, 0.0, cfg, show_progress,
                f"  reference(Nv={Nv})", n_realizations=cfg.n_realizations,
            )
```

Two passes through the same machinery, both reusing the one `H` built by
`_prepare_Nv`:

1. **Main pass**: `_disorder_average(..., cfg.sigma, ...)` — the actual
   disorder run at the configured `sigma`.
2. **Reference pass** (only if `need_ref`): `_disorder_average(..., 0.0,
   ...)` — the *same* function, but with `sigma` hardcoded to `0.0`
   regardless of `cfg.sigma`, and `n_realizations=cfg.n_realizations`
   explicitly passed (matching the main pass's realization count). This
   produces the "no disorder" spectrum used purely as a normalization
   reference, never plotted as the main result of this call.

```python
    E, spectrum = _broaden(all_evals, all_intensity, cfg)
    reference_max = None
    reference_area = None
    if need_ref:
        _, ref_spectrum = _broaden(ref_evals, ref_intensity, cfg)
        reference_max = ref_spectrum.max()
        reference_area = np.trapezoid(ref_spectrum, E)
    spectrum = _normalize_spectrum(E, spectrum, cfg, reference_max, reference_area)
```

- The main pass's raw sticks are broadened/averaged into `(E, spectrum)`.
- If a reference is needed, the reference pass's sticks are *separately*
  broadened/averaged into `ref_spectrum`, from which both summary statistics
  are extracted: `reference_max = ref_spectrum.max()` (peak height) and
  `reference_area = np.trapezoid(ref_spectrum, E)` (trapezoidal-rule
  numerical integral of the reference curve over the energy grid).
- Finally `_normalize_spectrum` applies the single configured division,
  exactly once, to the fully averaged `spectrum` — never to `ref_spectrum`
  itself, and never per-realization.

```python
    return {
        "Nv": Nv, "sigma": cfg.sigma, "dim": dim, "E": E, "spectrum": spectrum,
        "reference_max": reference_max, "reference_area": reference_area,
        "normalization": cfg.NORMALIZATION,
        "all_evals": all_evals, "all_intensity": all_intensity,
    }
```

The result dict carries both the final normalized `spectrum` and the raw
per-realization `all_evals`/`all_intensity` (kept around for downstream
analysis, e.g. the representative-realization workflow), plus the
provenance of exactly how normalization was done (`reference_max`,
`reference_area`, `normalization`) so it can be reproduced or audited later
without recomputation.

## 9. `compute_spectrum_sigma_sweep` (lines 344–418)

Same building blocks, restructured for efficiency across many `sigma`
values at one fixed `Nv`:

```python
    parallel = _parallel_enabled(cfg)
    if not parallel:
        H, mask1, mask2, i0, dim = _prepare_Nv(Nv, cfg, show_progress)
```

`H` (and friends) are built **once** for the whole sweep, not once per
`sigma` — the static Hamiltonian doesn't depend on `sigma`, so there is no
reason to rebuild it.

```python
    reference_max = None
    reference_area = None
    if need_ref:
        ...
        E_ref, ref_spectrum = _broaden(ref_evals, ref_intensity, cfg)
        reference_max = ref_spectrum.max()
        reference_area = np.trapezoid(ref_spectrum, E_ref)
```

The σ = 0 reference pass also happens **once**, before the sweep loop, not
once per `sigma` — every `sigma` in the sweep shares the exact same
`reference_max`/`reference_area`. This is the crux of the docstring's point:
"the peak reduction caused by disorder stays visible instead of being
normalized away per curve" — if each `sigma`'s spectrum were normalized by
its *own* peak, disorder-induced peak suppression would vanish from the plot
by construction.

```python
    for sigma in outer:
        if parallel:
            all_evals, all_intensity, dim = _disorder_average_parallel(...)
        else:
            all_evals, all_intensity = _disorder_average(
                H, mask1, mask2, i0, sigma, cfg, show_progress, f"  disorder(sigma={sigma:g})"
            )
        E, spectrum = _broaden(all_evals, all_intensity, cfg)
        spectrum = _normalize_spectrum(E, spectrum, cfg, reference_max, reference_area)
        results.append({...})
```

For each `sigma` in the sweep: run the disorder loop at that `sigma` (reusing
the shared `H`), broaden/average it, then normalize using the *shared*
reference computed once above. One result dict per `sigma` is appended to
`results` and returned as a list.

## 10. `compute_spectrum_realization_sweep` (lines 421–491)

This entry point studies convergence: how the spectrum changes as
`n_realizations` grows, at one fixed `Nv` and `sigma`.

```python
    counts = sorted(set(int(n) for n in realization_list))
    n_max = counts[-1]
```

Dedupe/sort the requested realization counts and take the largest,
`n_max`.

```python
    all_evals, all_intensity = _disorder_average(
        H, mask1, mask2, i0, cfg.sigma, cfg, show_progress,
        f"  disorder(Nv={Nv}, n={n_max})", n_realizations=n_max,
    )
```

The disorder loop runs **once**, at `n_max` realizations — not once per
requested count. This works because `_disorder_average` reseeds the RNG
identically every call (`np.random.default_rng(cfg.rng_seed)`), so the first
`n` draws of an `n_max`-realization run are bit-for-bit identical to what an
independent `n`-realization run would have produced. That means truncating
the arrays afterward is equivalent to rerunning with a smaller count, at a
fraction of the cost.

```python
    for n in counts:
        E, spectrum = _broaden(all_evals[:n], all_intensity[:n], cfg)
        spectrum = _normalize_spectrum(E, spectrum, cfg, reference_max, reference_area)
        results.append({..., "n_realizations": n, ...})
```

For each requested count `n`, `_broaden` is called on just the **first `n`
rows** (`all_evals[:n]`, `all_intensity[:n]`) of the single big run — i.e.
re-averaging over a prefix of the same realizations — and the *same* shared
reference (computed once, at `n_max`, before this loop) is used to normalize
every count's spectrum, for the same reason as the sigma sweep: so you can
see how the *averaging itself* converges, without the comparison being
distorted by re-normalizing each partial average to its own peak.

## 11. The parallel path (lines 177–288) — same math, different plumbing

`_disorder_average_parallel`, `_worker_init`, and `_worker_task` are an
opt-in (`cfg.n_workers > 1`) alternative execution path, not a different
algorithm:

- The disorder draws (`eps1_dis`, `eps2_dis` for every realization) are
  pre-generated **in the parent process**, in the exact same order the
  serial loop would generate them (same `rng = np.random.default_rng(cfg.rng_seed)`,
  same sequential `rng.normal(...)` calls) — so the *set* of realizations
  computed is identical to the serial path, just distributed across worker
  processes instead of looped over serially.
- `_worker_init` builds `H`/masks/`i0` once per worker process (mirroring
  `_prepare_Nv`).
- `_worker_task` mirrors the serial loop body exactly: add disorder to `H`'s
  diagonal, `eigh`, subtract it back off, compute intensity from `evecs[i0,
  :]`.
- Each worker is pinned to a single BLAS thread (`_THREAD_ENV_VARS`) because
  a single dense `eigh` call barely benefits from multiple BLAS threads, but
  running many *independent* `eigh` calls across worker processes scales
  almost linearly with worker count. (See project memory
  `blas-thread-perf-fix` for the related discovery that unpinned BLAS
  threading was making the serial path itself ~115× slower by
  oversubscribing cores.)

Everything downstream of getting `(all_evals, all_intensity)` — `_broaden`
and `_normalize_spectrum` — is exactly the same regardless of whether the
serial or parallel path produced those arrays.

---

## Summary: the two things this document set out to explain

**Disorder averaging is a loop, not a formula.** `_disorder_average` draws
`n_real` independent Gaussian perturbations to the two site energies, adds
each one onto the (reused) static Hamiltonian's diagonal, diagonalizes,
records eigenvalues + initial-state overlap intensities, and undoes the
diagonal change before the next draw. `_broaden` then turns every
realization's stick spectrum into a Lorentzian curve and sums them,
dividing by `n_real` at the end — that division is the actual "average" in
"disorder average."

**Normalization is one division, applied once, using a shared reference.**
`_normalize_spectrum` runs only after disorder averaging and broadening are
both complete. In the default `"reference"` mode (and its sibling
`"reference_area"`), the divisor comes from a *separate*, once-computed
σ = 0 spectrum for the same `Nv` — shared across every `sigma` and every
realization count in a sweep — specifically so that disorder's real effect
on the spectrum (peak suppression, broadening) remains visible in the
normalized curves instead of being normalized away.
