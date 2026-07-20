# Disorder-Averaged Vibronic–Cavity Absorption Spectra

A modular Python project that computes disorder-averaged absorption spectra for a
two-molecule vibronic system coupled to a cavity, sweeping over the number of
vibrational levels `Nv` and overlaying the resulting spectra on a single plot.

This is a refactor of the original notebook `Spectrum_symm_avgdisorder_2.2 (3).ipynb`
into a reusable, resumable, progress-tracked command-line tool.

---

## Table of contents

- [The physics](#the-physics)
- [Project layout](#project-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line usage](#command-line-usage)
- [Representative disorder realizations (analysis)](#representative-disorder-realizations-analysis)
- [Configuration](#configuration)
- [Output files](#output-files)
- [How it works](#how-it-works)
- [Performance notes](#performance-notes)
- [Reproducing / cross-checking the notebook](#reproducing--cross-checking-the-notebook)
- [Troubleshooting](#troubleshooting)

---

## The physics

Each basis state is an 8-tuple:

```
(e1, np1, nm1, e2, np2, nm2, cp, cm)
```

| Symbol        | Meaning                                                       |
|---------------|---------------------------------------------------------------|
| `e1`, `e2`    | Electronic state of molecule 1 / 2: `'A'` (ground), `'Ep'`, `'Em'` (excited doublet) |
| `np1`, `nm1`  | b+ / b− vibrational quanta of molecule 1 (0 … `Nv-1`)         |
| `np2`, `nm2`  | b+ / b− vibrational quanta of molecule 2 (0 … `Nv-1`)         |
| `cp`, `cm`    | Cavity photon occupations of the two photon modes (0 or 1)    |

The full product basis is filtered down to a single **symmetry sector** before
diagonalization:

- **Single excitation** `Nex = 1` — total of electronic excitations plus cavity
  photons equals one.
- **Angular-momentum sector** `Jz = -1`, where `Jz = 2·Lz + Sz + lz` with
  `Lz = (np1-nm1)+(np2-nm2)`, `Sz` from the electronic labels, and `lz = cp-cm`.

The Hamiltonian is

```
H = H_electronic + H_vibrational + H_cavity + H_JahnTeller(kappa) + H_matter-cavity(g)
```

- **Electronic**: site energies `eps1`, `eps2` (only when that molecule is excited),
  with per-realization Gaussian **disorder** of strength `sigma`.
- **Vibrational**: `omega · (np1 + nm1 + np2 + nm2)`.
- **Cavity**: `omega_c · (cp + cm)`.
- **Jahn–Teller (vibronic) coupling** `kappa`: mixes `Ep ↔ Em` while raising/lowering
  a vibrational quantum.
- **Matter–cavity coupling** `g = Omega / (2·√N)`: exchanges an electronic excitation
  for a cavity photon.

For each of `n_realizations` disorder samples the Hamiltonian is diagonalized, and the
absorption **intensity** of each eigenstate is the squared overlap with the initial
state `('A',0,0,'A',0,0,0,1)`. Every eigenvalue is broadened by a **Lorentzian** of
FWHM `gamma`, summed, and averaged over realizations to produce the final spectrum.

---

## Project layout

```
Summer-Internship-2026/
├── main.py                 # CLI entry point: sweep Nv, save/resume, overlay plot
├── requirements.txt        # numpy, scipy, matplotlib, tqdm
├── README.md               # this file
├── spectrum/               # the package
│   ├── __init__.py
│   ├── config.py           # all physical + run parameters (Config dataclass)
│   ├── basis.py            # basis construction + Nex/Jz sector filtering
│   ├── operators.py        # second-quantized ladder + electronic operators
│   ├── hamiltonian.py      # static H (built once/Nv) + per-realization diagonal
│   ├── spectrum.py         # disorder averaging + Lorentzian broadening
│   ├── storage.py          # per-Nv save/load (.npz + .dat), resume logic
│   ├── plotting.py         # overlay all spectra (+ optional reference curve) + representative plots
│   └── representative.py  # representative-disorder-realization analysis (Pass 1 + Pass 2)
├── tests/
│   └── test_representative.py  # RNG-stream match, Pass-2 exactness, spectrum-unchanged pin
└── results/                # created at runtime
    ├── spectrum_Nv2.npz    # full result per Nv (arrays + provenance)
    ├── spectrum_Nv2.dat    # portable two-column text (E, intensity)
    ├── overlay_spectra.png # final overlaid figure
    ├── representative_scatter_Nv12_sigma0.24.png
    └── representative_spectrum_Nv12_sigma0.24_case{A..G}.png
```

---

## Installation

The project targets the bundled virtual environment `venv/` (Python 3.14).

```bash
cd JT-Absorption-Spectra/
venv/bin/pip install -r requirements.txt
```

Dependencies: **numpy**, **scipy**, **matplotlib**, **tqdm**.
(`qutip` is *not* required — the original notebook imported it but never used it.)

---

## Quick start

```bash
# Full sweep over Nv = 2..12 (the default), overlay at the end
venv/bin/python main.py

# Fast smoke test: two small Nv, few realizations, no GUI window
venv/bin/python main.py --nv 2 3 --realizations 3 --no-show
```

Results and the overlay figure land in `results/`.

---

## Command-line usage

```
venv/bin/python main.py [--nv N [N ...]] [--realizations K]
                        [--force] [--no-show] [--results-dir DIR]
```

| Flag              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `--nv N [N ...]`  | Explicit list of `Nv` values to compute. Default: `Config.nv_list` (`2..12`).|
| `--realizations K`| Override the number of disorder realizations for this run.                  |
| `--force`         | Recompute every `Nv` even if a saved result already exists.                 |
| `--no-show`       | Save the overlay figure without opening an interactive window (headless).   |
| `--results-dir D` | Write outputs to `D` instead of the configured `results/`.                  |
| `--workers N`     | Run disorder realizations across `N` processes (default: serial). See [performance notes](#performance-notes). |
| `--sigma-sweep`   | Sweep disorder strength `sigma` for a **single** `Nv` instead of sweeping `Nv`. |
| `--sigmas S [S ...]` | Sigma values for `--sigma-sweep` (default: `Config.sigma_list`).         |
| `--realization-sweep` | Sweep the number of disorder realizations for a **single** `Nv` (convergence study). |
| `--realization-list N [N ...]` | Realization counts for `--realization-sweep` (default: `Config.realization_list`). |

**Examples**

```bash
# Only the even Nv values
venv/bin/python main.py --nv 2 4 6 8 10 12

# Recompute Nv=12 from scratch with a heavier disorder average
venv/bin/python main.py --nv 12 --realizations 500 --force

# Headless run on a server
venv/bin/python main.py --no-show
```

### Sigma sweep (fixed Nv, varying disorder strength)

Instead of sweeping `Nv`, compute several spectra for **one** `Nv` at different
disorder strengths `sigma` and overlay them. The static Hamiltonian is built **once**
and reused for every `sigma`, so only the disorder loop reruns.

```bash
# Sweep sigma for Nv=12 using the configured sigma_list
venv/bin/python main.py --sigma-sweep --nv 12

# Explicit sigma values
venv/bin/python main.py --sigma-sweep --nv 8 --sigmas 0.01 0.03 0.05 0.08 --no-show
```

- The `Nv` for the sweep is the first value passed to `--nv`, or `Config.sigma_sweep_nv`
  (default `12`) if `--nv` is omitted.
- Sigma values come from `--sigmas`, or `Config.sigma_list`
  (default `[0.01, 0.03, 0.05, 0.08]`).
- Each `(Nv, sigma)` is saved to its own file and resumed/skipped on rerun, exactly
  like the `Nv` sweep. Use `--force` to recompute.
- **Normalization.** Unlike the other overlays (each curve scaled to its own peak of
  1), the sigma overlay scales **every** curve by a single shared factor — the raw
  broadened peak of the smallest-σ curve (ideally `σ=0`, i.e. no disorder). The `σ=0`
  curve therefore peaks at 1 and each disordered curve peaks **below** it, making the
  disorder-driven peak reduction visible; this also lines up with the external
  reference curve (which already peaks at 1). This relies on the `spectrum_max` field
  now stored in each `.npz` (see below), so **existing sigma results predating this
  field must be recomputed** — rerun with `--force` (or clear `results/`). Old files
  without it fall back to the previous per-curve normalization, with a warning.

Programmatic use:

```python
from spectrum.config import Config
from spectrum.spectrum import compute_spectrum_sigma_sweep

cfg = Config()
results = compute_spectrum_sigma_sweep(Nv=12, sigma_list=[0.01, 0.03, 0.05], cfg=cfg)
for res in results:
    print(res["sigma"], res["spectrum"].max())
```

### Realization sweep (fixed Nv & sigma, varying #realizations)

A convergence study: compute the spectrum for **one** `Nv` at several disorder
**realization counts** and overlay them, to see how the disorder average settles as
more samples are added. The disorder loop runs **once** at the largest requested
count; each smaller count reuses the leading realizations (numerically identical to an
independent shorter run, since the RNG is seeded once).

```bash
# Convergence for Nv=12 using the configured realization_list
venv/bin/python main.py --realization-sweep --nv 12

# Explicit counts, headless
venv/bin/python main.py --realization-sweep --nv 8 --realization-list 10 50 100 200 --no-show
```

- The `Nv` is the first value passed to `--nv`, or `Config.realization_sweep_nv`
  (default `12`). The disorder strength is `Config.sigma`.
- Counts come from `--realization-list`, or `Config.realization_list`
  (default `[10, 50, 100, 200]`).
- Each `(Nv, sigma, n)` is saved to its own file and resumed/skipped on rerun; use
  `--force` to recompute.

Programmatic use:

```python
from spectrum.config import Config
from spectrum.spectrum import compute_spectrum_realization_sweep

cfg = Config()
results = compute_spectrum_realization_sweep(Nv=12, realization_list=[10, 50, 100], cfg=cfg)
for res in results:
    print(res["n_realizations"], res["spectrum"].max())
```

---

## Representative disorder realizations (analysis)

A **separate, additive** analysis layer (`--representative`) examines the *individual*
disorder realizations behind the averaged spectrum, without changing that spectrum's
physics. The absorption-spectrum modules (`basis.py`, `operators.py`, `hamiltonian.py`,
`spectrum.py`) are untouched; a regression test
([`tests/test_representative.py`](tests/test_representative.py)) pins this.

Two memory-light passes:

1. **Pass 1** — call the existing disorder-averaging pipeline completely unmodified
   (same averaged spectrum as always), and separately record lightweight per-realization
   metadata (`index`, `eps1`, `eps2`, `delta1 = eps1 - eps`, `delta2 = eps2 - eps`) by
   replaying the *same* RNG stream (same seed, same draw order) the pipeline uses
   internally. No individual spectra are stored.
2. **Automatic selection** — from that `(eps1, eps2)` cloud, pick the realization
   closest to each of 7 physically meaningful targets:

   | Case | Meaning |
   |------|---------|
   | A | Nearly no disorder (`eps1 ≈ eps`, `eps2 ≈ eps`) |
   | B | Molecule 1 resonant with the cavity, molecule 2 far detuned |
   | C | Reverse of B |
   | D | Large *opposite* disorder (`eps1 = eps+δ`, `eps2 = eps-δ`, large `\|δ\|`) |
   | E | Both shifted upward |
   | F | Both shifted downward |
   | G | Large disorder — realization nearest the *edge* of the sampled cloud (farthest from center, not closest to a point) |

3. **Pass 2** — build the static Hamiltonian **once** (as the sigma-sweep code already
   does) and recompute a full spectrum only for those ~7 selected realizations, reusing
   the same Hamiltonian/diagonalization/broadening/normalization building blocks as the
   main pipeline — so Pass 2's cost and memory footprint never scale with
   `n_realizations`.

```bash
# 300 realizations at Nv=12 (defaults), scatter + 7 case plots + summary table
venv/bin/python main.py --representative

# Custom Nv / realization count / disorder strength, headless
venv/bin/python main.py --representative --nv 8 --repr-realizations 500 --repr-sigma 0.3 --no-show
```

| Flag                    | Description                                                        |
|-------------------------|----------------------------------------------------------------------|
| `--representative`      | Run this analysis instead of the Nv sweep.                          |
| `--repr-realizations N` | Realizations sampled for the cloud (default: `Config.representative_n_realizations`, 300). |
| `--repr-sigma S`        | Disorder strength for this analysis (default: `Config.sigma`).      |

Outputs (in `results/`): `representative_metadata_Nv{N}_sigma{σ}.csv` (all sampled
realizations), `representative_summary_Nv{N}_sigma{σ}.csv` (the 7-case table),
`representative_scatter_Nv{N}_sigma{σ}.png`, and one
`representative_spectrum_Nv{N}_sigma{σ}_case{X}.{png,npz,dat}` per case.

---

## Configuration

All parameters live in [`spectrum/config.py`](spectrum/config.py) as a `Config`
dataclass. Edit the defaults there, or override the common ones from the command line.

| Parameter          | Default        | Meaning                                            |
|--------------------|----------------|----------------------------------------------------|
| `omega`            | `0.08196`      | Vibrational quantum (eV)                           |
| `kappa`            | `0.180312`     | Jahn–Teller (vibronic) coupling (eV)               |
| `omega_c`          | `6.85`         | Cavity photon energy (eV)                          |
| `Omega`            | `2·6.85·0.05`  | Collective light–matter coupling scale             |
| `N`                | `2`            | Number of molecules                                |
| `Np`               | `2`            | Cavity photon-number cutoff (states 0, 1)          |
| `eps`              | `7.0`          | Base electronic site energy (eV)                   |
| `delta1`, `delta2` | `+0.2`, `-0.2` | Site detunings → `eps1 = 7.2`, `eps2 = 6.8`        |
| `sigma`            | `0.03`         | Disorder strength *W* (eV)                         |
| `n_realizations`   | `200`          | Number of disorder samples averaged                |
| `rng_seed`         | `1234`         | Seed for reproducible disorder                     |
| `E_min`, `E_max`   | `6.0`, `8.0`   | Spectrum energy window (eV)                        |
| `E_points`         | `2000`         | Grid resolution                                    |
| `gamma`            | `0.044`        | Lorentzian FWHM (eV)                               |
| `initial_state`    | `('A',0,0,'A',0,0,0,1)` | State whose overlap defines the intensity |
| `nex_target`       | `1`            | Excitation-number sector                           |
| `jz_target`        | `-1`           | `Jz` sector                                        |
| `nv_list`          | `range(2,13)`  | Default `Nv` sweep                                 |
| `sigma_sweep_nv`   | `12`           | `Nv` used by `--sigma-sweep` when `--nv` is omitted |
| `sigma_list`       | `[0,0.01,0.03,0.05,0.08]` | Default disorder strengths for the sigma sweep |
| `realization_sweep_nv` | `12`       | `Nv` used by `--realization-sweep` when `--nv` is omitted |
| `realization_list` | `[10,50,100,200]` | Default realization counts for the realization sweep |
| `results_dir`      | `"results"`    | Output directory                                   |
| `reference_file`   | `None`         | Optional "without disorder" curve (see below)      |

> **Note on `eps1`/`eps2`.** In the original notebook these were defined only in a
> commented-out cell yet used by the Hamiltonian loop. They are now derived properties
> `eps1 = eps + delta1` and `eps2 = eps + delta2`.

### Optional reference curve

To overlay a pre-computed "without disorder" spectrum, set `reference_file` in
`config.py` to the path of a two-column (`E  intensity`) text file (lines beginning
with `#` are treated as comments). If the file is missing, the overlay simply omits it.

---

## Output files

For each `Nv`, two files are written **immediately** after it is computed:

- **`results/spectrum_Nv{Nv}.npz`** — the complete result, loadable with
  `numpy.load`. Keys:
  - `E` — energy grid, shape `(E_points,)`
  - `spectrum` — broadened spectrum, self-normalized to a peak of 1, shape `(E_points,)`
  - `spectrum_max` — the raw (pre-normalization) peak that was divided out; lets the
    sigma overlay re-normalize every curve to a common σ=0 reference (scalar)
  - `all_evals` — eigenvalues, shape `(n_realizations, dim)`
  - `all_intensity` — per-eigenstate intensities, shape `(n_realizations, dim)`
  - `Nv`, `dim`, and provenance scalars (`n_realizations`, `sigma`, `gamma`,
    `rng_seed`, `eps1`, `eps2`, `omega`, `kappa`, `omega_c`, `Omega`)
- **`results/spectrum_Nv{Nv}.dat`** — a portable two-column text file (`E  intensity`)
  for import into other tools.

At the end of the sweep:

- **`results/overlay_spectra.png`** — all computed spectra overlaid (one curve per
  `Nv`), plus the optional reference curve.

The **sigma sweep** writes analogous files tagged by both `Nv` and `sigma`:

- `results/spectrum_Nv{Nv}_sigma{sigma}.npz` and `.dat` — one per `(Nv, sigma)`
  (the `.npz` also stores the `sigma` used).
- `results/overlay_sigma_Nv{Nv}.png` — the spectra for that `Nv` overlaid, one curve
  per `sigma`.

The **realization sweep** writes files tagged by `Nv`, `sigma`, and count:

- `results/spectrum_Nv{Nv}_sigma{sigma}_real{n}.npz` and `.dat` — one per count.
- `results/overlay_realizations_Nv{Nv}_sigma{sigma}.png` — spectra for that `Nv`
  overlaid, one curve per realization count.

Loading a saved result in your own script:

```python
import numpy as np
d = np.load("results/spectrum_Nv12.npz")
E, spec = d["E"], d["spectrum"]
```

---

## How it works

`main.py` orchestrates the sweep:

1. Build a `Config`, applying any CLI overrides.
2. For each `Nv` (outer progress bar):
   - If `results/spectrum_Nv{Nv}.npz` exists and `--force` was **not** given, **load
     and skip** (resume).
   - Otherwise compute the spectrum, then **save immediately** so partial progress
     survives an interruption.
3. Overlay all collected spectra and write `overlay_spectra.png`.
4. Print a summary of which `Nv` were computed vs loaded, plus their sector dimensions.

Per-`Nv` computation (`spectrum.compute_spectrum_for_Nv`):

1. `basis.build_sector_basis` builds the full product basis and filters to the
   `Nex=1`, `Jz=-1` sector.
2. `hamiltonian.build_static_hamiltonian` builds the **disorder-independent** matrix
   once (vibrational + cavity diagonal, Jahn–Teller and matter–cavity couplings).
3. For each disorder realization (inner progress bar): draw disordered site energies,
   add the electronic diagonal in place, `numpy.linalg.eigh`, record eigenvalues and
   the initial-state overlap intensities, then restore the diagonal.
4. Sum Lorentzians over all realizations and eigenvalues; normalize.

### Progress bars

Three nested `tqdm` bars are shown:
- **`Nv sweep`** — outer loop over `Nv`.
- **`H(Nv=…) diag`** — Hamiltonian assembly for the current `Nv`.
- **`disorder(Nv=…)`** — the per-realization diagonalization loop.

---

## Performance notes

- Only the **electronic diagonal** depends on disorder. The expensive Python assembly
  loop therefore runs **once per `Nv`**; each realization only updates the diagonal in
  place before diagonalizing. This is numerically identical to rebuilding the full
  matrix every realization, but far cheaper.
- The cost is dominated by `eigh` on a dense `dim × dim` matrix, repeated
  `n_realizations` times. `dim` grows with `Nv`:

  | Nv | dim  |
  |----|------|
  | 2  | 30   |
  | 3  | 105  |
  | …  | …    |
  | 12 | 6900 |

  Small `Nv` finish in well under a second; `Nv = 10, 12` at 200 realizations take
  substantially longer.
- **Everything is resumable.** Because each `Nv` is saved as soon as it finishes, you
  can stop the run (Ctrl-C) and restart later — completed `Nv` are loaded from disk and
  skipped. Use `--force` to recompute.

### Parallel realizations (`--workers`)

Disorder realizations are independent, so they can run across processes:

```bash
venv/bin/python main.py --nv 12 --workers 4        # 4 worker processes
```

- Default (no `--workers`, or `--workers 1`) runs the **unchanged serial path**.
- Workers are each pinned to a single BLAS thread, and the disorder draws are
  pre-generated in the parent in the same order, so **results are identical to the
  serial run** (to floating-point noise, ~1e-13, far below `gamma`).

**Expect only a modest speedup on a typical desktop.** The cost is dominated by dense
`numpy.linalg.eigh`, which is **memory-bandwidth-bound** — computing every eigenvector
streams large amounts of data, and the CPU cores share one memory bus. Measured on an
8-core desktop (dim≈2040):

| workers | speedup |
|---------|---------|
| serial  | 1.0×    |
| 2       | 1.1×    |
| 4       | **1.3×** (sweet spot) |
| 8       | 1.2× (no better — bandwidth saturated) |

So `--workers 2`–`4` is worthwhile; beyond that it stops helping on this class of
machine. On a workstation/cluster node with more memory channels or NUMA, it scales
considerably better, which is where large `Nv=12 × 200`-realization runs belong. For an
order-of-magnitude speedup on any hardware, the real lever is algorithmic — see below.

### Bigger optimizations (not yet implemented)

- **Sparse spectral method (largest win, ~50–500×).** The spectrum is the projected
  spectral function ⟨i0|δ(E−H)|i0⟩, and `H` is only ~0.05% filled while just ~120–370 of
  the 6900 eigenstates carry any intensity. A **Kernel Polynomial (Chebyshev)** or
  **Lanczos continued-fraction** method computes this with sparse matrix-vector products
  and never forms the eigenvectors — sidestepping both the O(dim³) cost and the memory
  wall above. Caveat: it changes the numerical method and the per-realization
  max-normalization would need a physics decision, so it warrants validation against the
  dense result.
- **Fewer realizations.** Use `--realization-sweep` to find the smallest converged
  realization count; dropping 200→50 (if converged) is a free 4×.
- **`float32`** would roughly halve `eigh` time at some precision cost (likely fine
  relative to `gamma = 0.044`), but is a lower-priority, riskier tweak.

---

## Reproducing / cross-checking the notebook

The modular Hamiltonian matches the original notebook exactly. For `Nv = 12`:

```
dim              : 6900     (notebook: 6900)
Hermitian        : True
non-zero entries : 24724    (notebook: 24724)
diagonal entries : 6900     (notebook: 6900)
initial state    : index 0
```

You can re-verify with:

```bash
venv/bin/python - <<'PY'
import numpy as np
from spectrum.config import Config
from spectrum.basis import build_sector_basis
from spectrum.hamiltonian import build_static_hamiltonian, electronic_masks, electronic_diagonal

cfg = Config(); Nv = 12
basis, index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
H = build_static_hamiltonian(Nv, basis, index, cfg, show_progress=False)
m1, m2 = electronic_masks(basis); d = np.arange(len(basis))
H[d, d] += electronic_diagonal(m1, m2, cfg.eps1, cfg.eps2)
print("dim", len(basis), "| Hermitian", np.allclose(H, H.T),
      "| nonzero", np.count_nonzero(H))
PY
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'numpy'`** — install the dependencies into
  the venv: `venv/bin/pip install -r requirements.txt`.
- **A plot window never appears / running over SSH** — pass `--no-show`; the figure is
  still written to `results/overlay_spectra.png`.
- **`Initial state (...) not in the Nv=… sector basis`** — the configured
  `initial_state` does not satisfy the `Nex`/`Jz` sector constraints; check
  `initial_state`, `nex_target`, and `jz_target` in `config.py`.
- **Want to start over** — delete `results/` (or the specific `spectrum_Nv*.npz`
  files) and rerun, or use `--force`.
