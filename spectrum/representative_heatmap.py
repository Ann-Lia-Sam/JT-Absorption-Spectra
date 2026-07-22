"""P(v) heatmaps for the selected representative disorder realizations.

This is a *separate, additive* analysis layer (Option A). It reuses:

* the physically-motivated Case A-K selection from
  :mod:`spectrum.representative` (which needs only each realization's drawn
  ``(eps1, eps2)`` -- no absorption diagonalization at all), and
* the paper's vibronic -> polaritonic machinery
  (:mod:`spectrum.vibronic` / :mod:`spectrum.polariton` /
  :mod:`spectrum.participation`) to build, for each selected realization, the
  discrete Fig. S1 heatmap of the single-molecule vibronic-sector population
  ``P(v)`` of the bright polaritonic states.

For each of the (typically 11) selected realizations we do exactly **one**
polaritonic diagonalization. The single-molecule Jahn-Teller basis is
diagonalized **once** (at ``eps_k = 0``) and reused for every molecule and every
realization via :func:`spectrum.vibronic.shift_reference` (an O(1) rigid
eigenvalue shift). The realizations can be solved across worker processes
(``cfg.n_workers``), mirroring the absorption pipeline's parallel path.

Nothing here touches the frozen absorption-spectrum pipeline
(``basis``/``operators``/``hamiltonian``/``spectrum.py``); those are only
imported read-only, never modified.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from . import participation, polariton, vibronic
from .config import Config
from .representative import (
    RealizationMetadata,
    collect_realization_metadata,
    select_representative_realizations,
)
from .storage import _case_slug, _sigma_tag, ensure_results_dir


# ---------------------------------------------------------------------------
# Single-realization solve (one polaritonic diagonalization)
# ---------------------------------------------------------------------------
def solve_representative(
    cfg: Config,
    eps1_dis: float,
    eps2_dis: float,
    reference: Optional[vibronic.VibronicSolution] = None,
) -> polariton.PolaritonSolution:
    """Vibronic -> polaritonic solve for one explicit ``(eps1, eps2)`` pair.

    ``reference`` (from :func:`spectrum.vibronic.diagonalize_single_molecule_reference`)
    lets both molecules' vibronic bases be built by an O(1) eigenvalue shift
    instead of re-diagonalizing the single-molecule JT Hamiltonian. If omitted it
    is built here (one cheap diagonalization, shared by both molecules).
    """
    if reference is None:
        reference = vibronic.diagonalize_single_molecule_reference(cfg)
    vib1 = vibronic.shift_reference(reference, eps1_dis)
    vib2 = vibronic.shift_reference(reference, eps2_dis)
    return polariton.solve(vib1, vib2, cfg)


def _heatmap_entry(sel: Dict, hm: participation.HeatmapResult) -> Dict:
    """Bundle a selected case's metadata with its computed Fig. S1 heatmap."""
    m: RealizationMetadata = sel["metadata"]
    return {
        "case": sel["case"],
        "letter": sel["letter"],
        "description": sel["description"],
        "realization_index": m.index,
        "eps1": m.eps1,
        "eps2": m.eps2,
        "delta1": m.delta1,
        "delta2": m.delta2,
        "score": sel.get("score"),
        "filter_satisfied": sel.get("filter_satisfied"),
        # Fig. S1 heatmap arrays
        "grid": hm.grid,
        "bright_idx": hm.bright_idx,
        "energies": hm.energies,
        "intensity": hm.intensity,
        "heatmap": hm.heatmap,       # (n_v, n_bright) = P(v) per bright state
        "pr": hm.pr,
    }


# ---------------------------------------------------------------------------
# Optional parallel path (opt-in via cfg.n_workers > 1), mirrors spectrum.py
# ---------------------------------------------------------------------------
_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_WORKER: Dict = {}


def _parallel_enabled(cfg: Config) -> bool:
    n = getattr(cfg, "n_workers", None)
    return bool(n) and n > 1


def _worker_init(cfg: Config) -> None:
    """Build the single-molecule vibronic reference once per worker process."""
    _WORKER["cfg"] = cfg
    _WORKER["reference"] = vibronic.diagonalize_single_molecule_reference(cfg)


def _worker_task(task):
    """Solve one representative and return its heatmap (mirrors the serial body)."""
    key, eps1, eps2 = task
    cfg = _WORKER["cfg"]
    sol = solve_representative(cfg, eps1, eps2, reference=_WORKER["reference"])
    hm = participation.bright_heatmap(sol, cfg)
    return key, hm


def _compute_parallel(cfg, tasks, show_progress, desc):
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    saved = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
    for k in _THREAD_ENV_VARS:
        os.environ[k] = "1"

    out: Dict[str, participation.HeatmapResult] = {}
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=cfg.n_workers, mp_context=ctx,
            initializer=_worker_init, initargs=(cfg,),
        ) as ex:
            it = ex.map(_worker_task, tasks, chunksize=1)
            if show_progress:
                it = tqdm(it, total=len(tasks), desc=desc, unit="case")
            for key, hm in it:
                out[key] = hm
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def compute_representative_heatmaps(
    selections: Dict[str, Dict],
    cfg: Config,
    show_progress: bool = True,
) -> Dict[str, Dict]:
    """Compute the discrete Fig. S1 heatmap for every selected representative.

    Returns an (insertion-ordered) dict keyed like ``selections`` -> an entry
    dict (see :func:`_heatmap_entry`) with the case metadata and heatmap arrays.
    Runs one polaritonic diagonalization per case, in parallel across
    ``cfg.n_workers`` processes when enabled.
    """
    tasks = [
        (key, sel["metadata"].eps1, sel["metadata"].eps2)
        for key, sel in selections.items()
    ]

    if _parallel_enabled(cfg):
        hm_by_key = _compute_parallel(
            cfg, tasks, show_progress, "representative heatmap"
        )
    else:
        reference = vibronic.diagonalize_single_molecule_reference(cfg)
        iterator = tasks
        if show_progress:
            iterator = tqdm(tasks, desc="representative heatmap", unit="case")
        hm_by_key = {}
        for key, eps1, eps2 in iterator:
            sol = solve_representative(cfg, eps1, eps2, reference=reference)
            hm_by_key[key] = participation.bright_heatmap(sol, cfg)

    # Preserve the selection order and attach metadata.
    return {key: _heatmap_entry(sel, hm_by_key[key]) for key, sel in selections.items()}


def run_representative_heatmaps(
    Nv: int, cfg: Config, show_progress: bool = True
) -> Dict:
    """Full standalone workflow: select Cases A-K, then heatmap each.

    Selection uses only the drawn ``(eps1, eps2)`` cloud (RNG replay via
    :func:`spectrum.representative.collect_realization_metadata`), so this does
    **not** run the absorption disorder average -- only the ``len(selections)``
    polaritonic diagonalizations for the heatmaps.
    """
    metadata = collect_realization_metadata(cfg)
    selections = select_representative_realizations(metadata, cfg)
    heatmaps = compute_representative_heatmaps(selections, cfg, show_progress=show_progress)
    return {"Nv": Nv, "metadata": metadata, "selections": selections, "heatmaps": heatmaps}


# ---------------------------------------------------------------------------
# Plotting (discrete Fig. S1, one per representative)
# ---------------------------------------------------------------------------
def _crop_v_range(grid: np.ndarray, heatmap: np.ndarray, floor: float = 1e-4):
    """Tightest ``(vlo, vhi)`` sector window holding all non-negligible weight."""
    rows = np.where(heatmap.max(axis=1) > floor)[0]
    if rows.size == 0:
        return int(grid[0]), int(grid[-1])
    return int(grid[rows[0]]), int(grid[rows[-1]])


def plot_representative_heatmap(
    entry: Dict, cfg: Config, out_path: str, show: bool = True
) -> str:
    """Plot one representative's Fig. S1 heatmap: P(v) vs bright polariton state.

    y-axis = vibronic sector ``v``; x-axis = bright polaritonic states ordered by
    increasing energy; color = ``P(v)`` -- exactly the paper's Fig. S1 layout.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap = entry["heatmap"]
    grid = entry["grid"]
    n_bright = heatmap.shape[1]
    x = np.arange(n_bright)

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(x, grid, heatmap, cmap="viridis", shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$P(v)$")
    ax.set_xlabel("Bright polaritonic states (energy-ordered)")
    ax.set_ylabel(r"Vibronic sector $v$")
    ax.set_title(
        f"{entry['case']}\n"
        f"realization #{entry['realization_index']}: "
        rf"$\epsilon_1$={entry['eps1']:.3f} eV, $\epsilon_2$={entry['eps2']:.3f} eV "
        f"(Nv={cfg.heatmap_nv}, σ={cfg.sigma:g})"
    )
    vlo, vhi = _crop_v_range(grid, heatmap)
    ax.set_ylim(vlo - 0.5, vhi + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Storage (self-contained; reuses storage.py path helpers, does not modify it)
# ---------------------------------------------------------------------------
import os  # noqa: E402


def representative_heatmap_path(cfg: Config, Nv: int, case_name: str, ext: str = "png") -> str:
    return os.path.join(
        cfg.results_dir,
        f"representative_heatmap_Nv{Nv}_{_sigma_tag(cfg.sigma)}_{_case_slug(case_name)}.{ext}",
    )


def save_representative_heatmap(entry: Dict, cfg: Config, Nv: int) -> str:
    """Persist one representative's heatmap arrays + provenance as a .npz."""
    ensure_results_dir(cfg)
    path = representative_heatmap_path(cfg, Nv, entry["case"], ext="npz")
    np.savez_compressed(
        path,
        case=entry["case"],
        letter=entry["letter"],
        description=entry["description"],
        realization_index=int(entry["realization_index"]),
        eps1=float(entry["eps1"]),
        eps2=float(entry["eps2"]),
        delta1=float(entry["delta1"]),
        delta2=float(entry["delta2"]),
        grid=entry["grid"],
        energies=entry["energies"],
        intensity=entry["intensity"],
        heatmap=entry["heatmap"],
        pr=entry["pr"],
        # provenance
        Nv=Nv,
        heatmap_nv=cfg.heatmap_nv,
        sigma=cfg.sigma,
        rng_seed=cfg.rng_seed,
        n_realizations=cfg.n_realizations,
        which_molecule=str(cfg.heatmap_which_molecule),
        bright_threshold=cfg.heatmap_bright_threshold,
        omega=cfg.omega, kappa=cfg.kappa, omega_c=cfg.omega_c, Omega=cfg.Omega,
        eps=cfg.eps,
    )
    return path
