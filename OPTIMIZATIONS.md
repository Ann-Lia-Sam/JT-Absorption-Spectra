# Optimizations

A complete catalogue of every performance optimization in this project — both
**external / environment-level** (threading, process parallelism, resume) and
**within the spectra computation itself** (Hamiltonian reuse, reference reuse,
eigendata reuse), plus the ones specific to the **representative-analysis** and
**P(v) heatmap** workflows.

Each entry lists **what**, **where** (file → function), **why** (the cost it
removes), and a **correctness note** (whether results are bit-identical or agree
to floating-point round-off).

> **Reference cost anchors** (measured on this 40-core box, `Nv=12`, sector
> `dim = 6900`):
> - one dense `np.linalg.eigh`: **~63 s** pinned to 1 BLAS thread; **>180 s**
>   unpinned (40-thread oversubscription — see O1).
> - sector-basis build: **~0.43 s** (once per `Nv`).
> - static Hamiltonian build: **~0.14 s** (once per `Nv`; `H` is 0.05 % dense).
>
> So `eigh × n_realizations` is **~99.99 %** of the runtime — every optimization
> below is ultimately about doing **fewer** `eigh` calls, making each one **as
> fast as possible**, or running them **concurrently**.

---

## Summary table

| # | Optimization | Layer | Where | Net effect |
|---|--------------|-------|-------|------------|
| O1 | Pin BLAS to 1 thread/process before NumPy import | external | [spectrum/__init__.py](spectrum/__init__.py), [main.py](main.py) | ~3×+ per `eigh` (avoids catastrophic oversubscription) |
| O2 | Process-level disorder parallelism (`--workers`) | external | [spectrum/spectrum.py](spectrum/spectrum.py) `_disorder_average_parallel` | ~N× across N workers |
| O3 | Auto-parallelize the analysis workflows | external | [main.py](main.py) `run_representative*` | no-op flag needed; uses the box automatically |
| O4 | Static Hamiltonian built once per `Nv`, disorder = diagonal update | core | [spectrum/hamiltonian.py](spectrum/hamiltonian.py), [spectrum/spectrum.py](spectrum/spectrum.py) `_disorder_average` | assembly cost amortized to ~0 |
| O5 | σ=0 reference reused across a whole sweep | core | [spectrum/spectrum.py](spectrum/spectrum.py) sigma/realization sweeps | 1 reference instead of one-per-curve |
| O6 | Realization sweep: diagonalize once at max count, slice prefixes | core | [spectrum/spectrum.py](spectrum/spectrum.py) `compute_spectrum_realization_sweep` | 1 disorder pass instead of one-per-count |
| O7 | Resume / skip already-saved results | external | [spectrum/storage.py](spectrum/storage.py), [main.py](main.py) | never recompute a finished `Nv`/σ/count |
| O8 | Representative selection needs **no** diagonalization | analysis | [spectrum/representative.py](spectrum/representative.py) `collect_realization_metadata` | selection is ~free (RNG replay) |
| O9 | Representative Pass 2 reuses Pass 1 eigendata | analysis | [spectrum/representative.py](spectrum/representative.py) `assemble_representative_spectra` | 0 extra `eigh` for the case plots |
| O10 | Representative Pass 1 computes σ=0 reference **once** | analysis | [spectrum/representative.py](spectrum/representative.py) `compute_average_for_representative` | ~halves Pass 1 in `reference*` modes |
| O11 | Single-molecule vibronic basis: diagonalize once, O(1) shift | heatmap | [spectrum/vibronic.py](spectrum/vibronic.py) `diagonalize_single_molecule_reference` / `shift_reference` | no per-molecule/per-realization re-diagonalization |
| O12 | Single-molecule H block-diagonalized by `(class, v)` | heatmap | [spectrum/vibronic.py](spectrum/vibronic.py) `diagonalize_single_molecule` | many tiny `eigh` instead of one big one |
| O13 | Heatmaps only for the ~11 selected cases | heatmap | [spectrum/representative_heatmap.py](spectrum/representative_heatmap.py) | ~11 `eigh`, not 100–300 |
| O14 | Vectorized `P(v)`, `alpha/beta`, sector build | heatmap | [spectrum/participation.py](spectrum/participation.py), [spectrum/polariton.py](spectrum/polariton.py) | Python loops → BLAS / conservation-law enumeration |

---

## External / environment-level

### O1 — Pin BLAS/OpenMP to one thread per process (before NumPy import)

**Where:** [spectrum/__init__.py](spectrum/__init__.py) (top of the package,
runs before any submodule imports NumPy) and [main.py](main.py) (after
`import os`, before the `spectrum` imports).

```python
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
```

**Why:** on this shared 40-core machine, an unpinned dense `eigh` lets OpenBLAS
spawn one thread per core. For this workload that is **catastrophically
counterproductive** — a single `Nv=12` `eigh` went from **~63 s** (1 thread) to
**> 180 s** (all cores), because the tridiagonal-reduction phase is
memory-bandwidth-bound and dominated by thread-barrier synchronization, and
because the box is shared (dozens of oversubscribed threads thrash the caches
and memory bus). This is the single most important fix: without it, everything
below is 3×+ slower.

**Must be pre-import:** OpenBLAS reads these variables at load time; setting
them *after* `import numpy` has no effect. `setdefault` means a value the user
exported by hand still wins. Because the parallel path (O2) uses `spawn`, each
worker re-imports the `spectrum` package and thus re-applies this pin.

**Correctness:** thread count changes only floating-point summation order inside
BLAS, i.e. differences at the ~1e-13 level — far below `gamma = 0.044 eV` and
the energy-grid spacing; physically identical.

### O2 — Process-level disorder parallelism (`--workers` / `cfg.n_workers`)

**Where:** [spectrum/spectrum.py](spectrum/spectrum.py) →
`_disorder_average_parallel`, `_worker_init`, `_worker_task`.

Disorder realizations are independent, so they run across worker **processes**
(not threads — see O1). Design points that make it correct and efficient:

- The disorder draws (`eps1_dis`, `eps2_dis` per realization) are
  **pre-generated in the parent**, in the exact order the serial loop would
  draw them (`np.random.default_rng(cfg.rng_seed)` then sequential
  `rng.normal`), and scattered to workers. So the *set* of realizations is
  identical to serial — only the execution is distributed.
- Each worker builds the static Hamiltonian **once** in `_worker_init` (O4
  again, per process) and reuses it for every task it receives.
- Each worker is pinned to a single BLAS thread (env vars set before the pool
  is spawned), so N workers give roughly an N-fold speedup on independent
  `eigh` calls rather than N processes each fighting for all cores.
- Uses `mp.get_context("spawn")` (not `fork`) for a clean re-import in each
  worker.

**Correctness:** results match the serial path to ~1e-13–1e-15 (a summation-order
artifact of the parallel reduction, not an algorithmic difference — verified by
reproducing the same discrepancy through the serial vs. parallel path).

### O3 — Auto-parallelize the analysis workflows

**Where:** [main.py](main.py) → `run_representative` (line ~253) and
`run_representative_heatmap` (line ~316).

If the user does **not** pass `--workers`, these workflows pick a sensible
worker count automatically instead of silently running serial:

```python
if cfg.n_workers is None:
    usable = max(1, (os.cpu_count() or 1) - 2)
    cfg.n_workers = max(1, min(usable, cfg.n_realizations))   # representative
    # ... = max(1, min(usable, 9))                             # heatmap (≤ 9 cases)
```

Capped at the number of independent tasks (`n_realizations`, or 9 for the
heatmap — more workers than tasks is wasted). Prints a one-line notice so the
behavior is visible. (The plain `Nv`/sigma/realization sweeps do **not**
auto-parallelize — they run serial unless `--workers` is given.)

**Rationale:** a standalone CLI wrapping a parallel-capable function easily
leaves the whole machine idle if the parallelism knob is forgotten; auto-default
avoids that footgun on a big shared box.

---

## Within the core spectra computation

### O4 — Static Hamiltonian built once per `Nv`; disorder is an in-place diagonal update

**Where:** [spectrum/hamiltonian.py](spectrum/hamiltonian.py)
(`build_static_hamiltonian` / `electronic_masks` / `electronic_diagonal`) +
[spectrum/spectrum.py](spectrum/spectrum.py) `_prepare_Nv` / `_disorder_average`.

The Hamiltonian splits into a **disorder-independent** part (vibrational + cavity
diagonal, Jahn–Teller coupling, matter–cavity coupling) and a
**disorder-dependent electronic diagonal**. The expensive Python assembly loop
runs **once per `Nv`**; each realization only adds the electronic diagonal in
place, diagonalizes, and subtracts it back:

```python
H[d, d] += elec
evals, evecs = np.linalg.eigh(H)
H[d, d] -= elec           # restore H for the next realization / sigma
```

**Why:** assembling `H` is ~0.14 s vs. ~63 s for `eigh` at `Nv=12`; rebuilding
it every realization would add nothing but waste. This add/eigh/subtract pattern
also lets the *same* `H` object be reused across every σ in a sweep (O5) and
across the whole disorder loop, and keeps memory flat (no new dense matrix per
realization).

**Correctness:** numerically identical to rebuilding the full matrix each
realization (it *is* the same matrix), and this is part of the frozen pipeline
pinned by the regression test.

### O5 — σ=0 reference spectrum reused across a whole sweep

**Where:** [spectrum/spectrum.py](spectrum/spectrum.py)
`compute_spectrum_sigma_sweep` and `compute_spectrum_realization_sweep`.

In `"reference"` / `"reference_area"` normalization, every curve in a sweep is
divided by a single shared scalar — the raw peak/area of this `Nv`'s σ=0
spectrum. That reference is computed **once**, before the sweep loop, and reused
for every σ / realization count, rather than recomputed per curve. (This also
serves a physics purpose: keeping the disorder-induced peak reduction visible;
see the normalization discussion in [README.md](README.md).)

**Correctness:** the reference is one deterministic computation shared by all
curves — no approximation.

### O6 — Realization sweep: one disorder pass at the max count, slice prefixes

**Where:** [spectrum/spectrum.py](spectrum/spectrum.py)
`compute_spectrum_realization_sweep`.

A convergence study over realization counts `[10, 50, 100, 200]` does **not**
run four disorder loops. It runs the disorder loop **once** at
`n_max = max(counts)`, then each smaller count reuses the **leading rows** of
the already-computed `all_evals` / `all_intensity`:

```python
n_max = counts[-1]
all_evals, all_intensity = _disorder_average(..., n_realizations=n_max)
for n in counts:
    E, spectrum = _broaden(all_evals[:n], all_intensity[:n], cfg)
```

This is exact because the RNG is seeded once, so the first `n` draws of an
`n_max`-run are bit-for-bit what an independent `n`-run would produce. Turns
`sum(counts)` diagonalizations into `n_max`.

### O7 — Resume / skip already-saved results

**Where:** [spectrum/storage.py](spectrum/storage.py) (`exists`, `sigma_exists`,
`realization_exists`, `save_*` write immediately) + the sweep loops in
[main.py](main.py).

Each `Nv` / (Nv, σ) / (Nv, σ, n) result is saved (`.npz` + `.dat`) **as soon as
it is computed**, and on rerun an existing file is loaded and skipped instead of
recomputed (unless `--force`). So an interrupted run resumes where it left off,
and re-plotting never re-diagonalizes.

**Note:** the skip decision is file-existence-based. Changing a physical
parameter without `--force` (or clearing `results/`) can therefore reuse a stale
file — rerun with `--force` when parameters change.

---

## Representative-analysis workflow (`--representative`)

### O8 — Selection needs no diagonalization

**Where:** [spectrum/representative.py](spectrum/representative.py)
`collect_realization_metadata` + `select_representative_realizations`.

The physically-selected Cases A–I depend only on each realization's drawn
`(eps1, eps2)`. Those are reproduced by **replaying the exact RNG stream**
(same seed, same draw order as `_disorder_average`) — a handful of
`rng.normal` calls, **no Hamiltonian, no `eigh`**. So choosing which
realizations are "representative" is essentially free.

### O9 — Pass 2 reuses Pass 1's eigendata (0 extra `eigh`)

**Where:** [spectrum/representative.py](spectrum/representative.py)
`assemble_representative_spectra` (used by `run_representative_analysis`).

Pass 1 already diagonalizes every realization and returns per-realization
`all_evals` / `all_intensity`. A selected case is just one of those realizations
(by index), so its spectrum is obtained by **broadening the already-computed
eigenvalues/intensities** — no re-diagonalization at all. The older
`recompute_representative_spectra` (which re-runs `eigh` per case) is kept only
as an independent cross-check in the tests.

**Correctness:** bit-identical to the recompute path to ~1e-13 (float round-off
in the broadening summation; the tiny difference comes from `H`'s diagonal
accumulating floating-point residue across the Pass-1 add/subtract loop vs. a
fresh `H`).

### O10 — Pass 1 computes the σ=0 reference **once** (not `n_realizations` times)

**Where:** [spectrum/representative.py](spectrum/representative.py)
`compute_average_for_representative`.

At σ=0 the disorder draw `rng.normal(0, 0)` is exactly `0`, so **every**
"reference realization" is the *same* deterministic no-disorder Hamiltonian.
The representative Pass 1 therefore computes the reference from a **single**
realization instead of `cfg.n_realizations` of them:

```python
ref_evals, ref_intensity = _disorder_average(..., 0.0, ..., n_realizations=1)
```

In `reference` / `reference_area` modes this roughly **halves** Pass 1 (the
reference pass is otherwise as expensive as the main pass). Uses only the frozen
`spectrum.py` primitives.

**Correctness:** exact, not an approximation — averaging one copy or `N`
identical copies gives the same peak/area; matches the frozen
`compute_spectrum_for_Nv` to ~1e-16. (The frozen absorption pipeline itself is
left untouched and still computes its reference at full `n_realizations`; this
optimization lives only in the additive representative layer.)

---

## P(v) heatmap workflow (`--representative-heatmap`)

See [spectrum/HEATMAP_WALKTHROUGH.md](spectrum/HEATMAP_WALKTHROUGH.md) for the
full line-by-line explanation; the optimizations are:

### O11 — Single-molecule vibronic basis: diagonalize once, O(1) shift per molecule/realization

**Where:** [spectrum/vibronic.py](spectrum/vibronic.py)
`diagonalize_single_molecule_reference` + `shift_reference`.

The single-molecule Jahn–Teller Hamiltonian's electronic energy `eps_k` enters
the excited block only as `eps_k · I`, so its **eigenvectors are
disorder-independent** — only the excited-block eigenvalues shift rigidly by
`+eps_k`. The reference is diagonalized **once** at `eps_k = 0`; each molecule
and each realization is then obtained by an **O(1) eigenvalue shift** (copy
evals, add `eps_k` to excited rows, reuse `U`/`v`/basis unchanged) instead of a
fresh diagonalization.

**Correctness:** algebraically exact (adding `c·I` to a block shifts its
eigenvalues by `c` and leaves eigenvectors unchanged, regardless of
degeneracies).

### O12 — Single-molecule H block-diagonalized by `(class, v)`

**Where:** [spectrum/vibronic.py](spectrum/vibronic.py)
`diagonalize_single_molecule` (via `_build_block`).

`H_{m,k}` is block-diagonal in (electronic class, vibronic sector `v`). Each
`(class, v)` block is built and diagonalized separately — many **small** `eigh`
calls instead of one large one — which is both cheaper and guarantees every
eigenvector has a definite `v` (no spurious `v`-mixing across accidentally
degenerate sectors).

### O13 — Heatmaps for only the ~11 selected cases (not the full cloud)

**Where:** [spectrum/representative_heatmap.py](spectrum/representative_heatmap.py)
`run_representative_heatmaps` + `compute_representative_heatmaps` (+ its
`_worker_init`/`_worker_task` parallel path, mirroring O2).

The heatmap does **one** polaritonic `eigh` per selected case (~11), driven by
the O8 selection — never the full 100–300-realization disorder cloud. Combined
with O11 (single-molecule step done once) and O2-style process parallelism
(each worker builds the reference once, cases run concurrently, BLAS pinned),
the whole per-σ heatmap job is ~one parallel round of `eigh`s. Because the
selection is free (O8), the heatmap workflow does **not** run the absorption
disorder average at all.

**Note:** the heatmap needs the *full* polaritonic eigenvectors (for `P(v)`),
which the absorption pipeline discards — so its `eigh`s cannot be reused from
Pass 1 (unlike O9). Diagonalizing directly in the vibronic basis yields the
eigenvectors already in the basis `P(v)` needs, so no change-of-basis is
required either.

### O14 — Vectorized `P(v)`, coupling matrices, and sector enumeration

**Where:** [spectrum/participation.py](spectrum/participation.py),
[spectrum/polariton.py](spectrum/polariton.py), [spectrum/vibronic.py](spectrum/vibronic.py).

- `P(v)` is computed for **every eigenstate at once** via `np.add.at`
  (scatter-add of `|evecs|²` rows into sector groups) instead of a Python
  double loop (`participation.compute_Pv` / `_grouped_weight`).
- `alpha` / `beta` matter–cavity coupling matrices are computed as single
  matrix products `P_A.T @ P_Ep` rather than explicit double sums over `(n+, n-)`
  (`vibronic.alpha_beta`).
- The `(nex=1, j=-1)` sector basis is enumerated **directly from the
  conservation laws** (bucket eigenstates by `v`, pair by `v_a + v_b = j - p`)
  instead of brute-force generating and filtering all `(a, b, p)` triples
  (`polariton.build_sector`).

---

## Not implemented (candidate future wins)

Documented for completeness; **none are in the code**. See the "Bigger
optimizations" note in [README.md](README.md).

- **Sparse spectral method (largest potential win, ~50–500×).** The absorption
  spectrum is the projected spectral function `⟨i0|δ(E−H)|i0⟩`, and `H` is only
  ~0.05 % filled with ~120–370 of 6900 eigenstates carrying intensity. A
  **Kernel Polynomial (Chebyshev)** or **Lanczos continued-fraction** method
  computes this with sparse matrix–vector products and never forms the
  eigenvectors — sidestepping the O(dim³) `eigh` entirely. **Caveat:** it is a
  *different, approximate* numerical method, so it changes results and would
  need validation against the dense result — which is why it has been kept out
  under the "don't change the spectra" constraint.
- **Fewer realizations.** Use `--realization-sweep` to find the smallest
  converged count; dropping 200 → 50 (if converged) is a free ~4×.
- **`float32`** would roughly halve `eigh` time at some precision cost — lower
  priority, riskier.

---

## Measured net effect

- A single clean `Nv=12`, 100-realization representative run: **~105 min
  (serial, unpinned era) → ~3–5 min** with O1 (BLAS pin) + O2/O3 (process
  parallelism) + O9/O10 (Pass-2 reuse, reference-once).
- The P(v) heatmap adds **~1–2 min per σ** (one parallel round of ~11 `eigh`s,
  O11–O14) rather than the ~11 min a naïve serial re-diagonalization would cost.
- Numerically, all of O1/O2/O9/O10 agree with the original serial/frozen results
  to **1e-13–1e-16** (floating-point round-off), and O4/O5/O6/O8 are **exact**.
  The frozen absorption pipeline (`basis.py`, `operators.py`, `hamiltonian.py`,
  `spectrum.py`) is unmodified by the analysis-layer optimizations.
