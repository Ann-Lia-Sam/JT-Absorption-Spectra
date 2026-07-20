"""Heatmap workflow: single-molecule sector populations of the polaritonic states.

Ties together the vibronic-basis machinery for two use cases:

* :func:`sigma0_heatmap` -- the discrete, single-realization Fig. S1 heatmap
  ``P(v)`` vs. bright-polariton index, reproducing the paper at ``sigma = 0``.
* :func:`disorder_average_heatmap` -- the disorder-averaged, *energy-resolved*
  heatmap. Each bright polaritonic state contributes ``intensity * P(v)``,
  Lorentzian-broadened in energy, averaged over realizations that use the *same*
  disorder draws as the absorption-spectrum pipeline (:mod:`spectrum.disorder`).
  Summing this map over ``v`` returns the polariton absorption spectrum, so it is
  a sector-resolved view of the spectrum -- the object used to analyze the
  collective vibronic cascade.

Two performance shortcuts, both exact (not approximations):

* The one-molecule vibronic eigenvectors and sector labels do not depend on the
  electronic energy (see :mod:`spectrum.vibronic`), so every solve in this module
  diagonalizes a single reference once (:func:`spectrum.vibronic.
  diagonalize_single_molecule_reference`) and reuses it -- for both molecules and
  every realization -- via an O(1) eigenvalue shift instead of re-diagonalizing.
* At ``sigma = 0`` every realization draws the identical ``(eps1, eps2)`` (the
  spectrum pipeline's ``rng.normal(0, 0)`` is exactly ``0``), so
  :func:`disorder_average_heatmap` solves once and skips the disorder loop
  entirely instead of repeating an identical calculation ``n_real`` times.

This is a self-contained analysis layer: nothing here touches the
absorption-spectrum modules, and nothing in ``main.py``'s default (non
``--heatmap``) code path imports this module.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from . import polariton, vibronic
from .config import Config
from .disorder import disorder_draws
from .participation import (
    HeatmapResult,
    bright_heatmap,
    compute_Pv,
    participation_ratio,
    sector_grid,
)
from .spectrum import lorentzian


# ---------------------------------------------------------------------------
# Single realization
# ---------------------------------------------------------------------------
def solve_realization(
    cfg: Config,
    eps1_dis: float,
    eps2_dis: float,
    reference: Optional[vibronic.VibronicSolution] = None,
) -> polariton.PolaritonSolution:
    """Full vibronic->polaritonic solve for one realization's electronic energies.

    ``reference`` (from :func:`spectrum.vibronic.diagonalize_single_molecule_reference`)
    lets both molecules' vibronic bases be built by an O(1) eigenvalue shift
    (:func:`spectrum.vibronic.shift_reference`) instead of a fresh
    diagonalization. If omitted, the reference is computed here (a single
    diagonalization, still shared between the two molecules).
    """
    if reference is None:
        reference = vibronic.diagonalize_single_molecule_reference(cfg)
    vib1 = vibronic.shift_reference(reference, eps1_dis)
    vib2 = vibronic.shift_reference(reference, eps2_dis)
    return polariton.solve(vib1, vib2, cfg)


def sigma0_heatmap(cfg: Config) -> HeatmapResult:
    """Discrete Fig. S1 heatmap at zero disorder (both molecules at ``cfg.eps``)."""
    reference = vibronic.diagonalize_single_molecule_reference(cfg)
    sol = solve_realization(cfg, cfg.eps, cfg.eps, reference=reference)
    return bright_heatmap(sol, cfg)


# ---------------------------------------------------------------------------
# Energy grid shared by the disorder-averaged heatmap and the spectrum
# ---------------------------------------------------------------------------
def energy_grid(cfg: Config) -> np.ndarray:
    E_min = cfg.E_min if cfg.heatmap_E_min is None else cfg.heatmap_E_min
    E_max = cfg.E_max if cfg.heatmap_E_max is None else cfg.heatmap_E_max
    return np.linspace(E_min, E_max, cfg.heatmap_E_points)


def _energy_map(sol: polariton.PolaritonSolution, cfg: Config, E: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Intensity-weighted, Lorentzian-broadened ``P(v)`` map for one realization.

    ``M[iv, iE] = sum_n intensity_n * P_n(grid[iv]) * L(E[iE] - lambda_n)``.
    Summing over ``iv`` yields that realization's polariton absorption spectrum.
    """
    _, Pv = compute_Pv(sol, cfg, grid)                 # (n_v, n_eig)
    # Broadened, intensity-weighted line shape per eigenstate: (n_eig, n_E)
    #   L[n, iE] = intensity_n * lorentzian(E, lambda_n)
    L = sol.intensity[:, None] * lorentzian(E[None, :], sol.evals[:, None], cfg.gamma)
    return Pv @ L                                       # (n_v, n_E)


@dataclass
class DisorderHeatmap:
    """Disorder-averaged, energy-resolved sector-population heatmap."""

    grid: np.ndarray            # (n_v,) integer vibronic sectors (y-axis)
    E: np.ndarray               # (n_E,) energy grid (x-axis)
    M: np.ndarray               # (n_v, n_E) intensity-weighted <P(v)>, disorder-averaged
    spectrum: np.ndarray        # (n_E,) polariton absorption spectrum = sum_v M
    Pv_norm: np.ndarray         # (n_v, n_E) M normalized per energy column (sum_v = 1)
    PR: np.ndarray              # (n_E,) participation ratio from the averaged P(v)
    sigma: float
    n_realizations: int
    Nv: int


def _accumulate(cfg: Config, draws, E, grid, show_progress: bool, reference) -> np.ndarray:
    """Serial accumulation of the per-realization energy maps.

    ``reference`` is diagonalized once by the caller and reused for every
    realization and both molecules via :func:`spectrum.vibronic.shift_reference`.
    """
    M = np.zeros((len(grid), len(E)))
    iterator = draws
    if show_progress:
        iterator = tqdm(draws, desc=f"  heatmap disorder(Nv={cfg.heatmap_nv})",
                        leave=False, unit="real")
    for eps1_dis, eps2_dis in iterator:
        sol = solve_realization(cfg, eps1_dis, eps2_dis, reference=reference)
        M += _energy_map(sol, cfg, E, grid)
    return M


def _finalize(M_avg: np.ndarray, n_real: int, grid, E, sigma, cfg) -> DisorderHeatmap:
    """Assemble a :class:`DisorderHeatmap` from an already realization-averaged map."""
    spectrum = M_avg.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Pv_norm = np.where(spectrum > 0, M_avg / spectrum, 0.0)
    PR = participation_ratio(Pv_norm)                  # per-energy PR from averaged P(v)
    return DisorderHeatmap(
        grid=grid, E=E, M=M_avg, spectrum=spectrum, Pv_norm=Pv_norm, PR=PR,
        sigma=sigma, n_realizations=n_real, Nv=cfg.heatmap_nv,
    )


def disorder_average_heatmap(
    cfg: Config,
    sigma: Optional[float] = None,
    n_real: Optional[int] = None,
    show_progress: bool = True,
) -> DisorderHeatmap:
    """Disorder-averaged, energy-resolved sector-population heatmap.

    Uses the same disorder draws (seed, order) as the absorption-spectrum
    pipeline via :func:`spectrum.disorder.disorder_draws`. The one-molecule
    vibronic reference is diagonalized exactly once and reused for both
    molecules and every realization (see module docstring).

    At ``sigma = 0`` every realization is identical (the disorder draw is
    ``eps + rng.normal(0, 0) == eps`` exactly), so the disorder loop is skipped
    entirely: a single solve stands in for the whole ``n_real``-realization
    average, giving the same ``M``/``spectrum``/``PR`` it would produce.
    """
    sigma = cfg.heatmap_sigma if sigma is None else sigma
    n_real = cfg.heatmap_realizations if n_real is None else n_real

    reference = vibronic.diagonalize_single_molecule_reference(cfg)
    E = energy_grid(cfg)
    # Sector grid is disorder-independent -> derive it from the cheap sector
    # build, without diagonalizing the (expensive) polaritonic Hamiltonian.
    vib0 = vibronic.shift_reference(reference, cfg.eps)
    grid = sector_grid(polariton.build_sector(vib0, vib0, cfg))

    if sigma == 0.0:
        sol = solve_realization(cfg, cfg.eps1, cfg.eps2, reference=reference)
        M_avg = _energy_map(sol, cfg, E, grid)
        return _finalize(M_avg, n_real, grid, E, sigma, cfg)

    draws = disorder_draws(cfg, sigma, n_real)
    if getattr(cfg, "n_workers", None) and cfg.n_workers > 1:
        M_sum = _accumulate_parallel(cfg, draws, E, grid, show_progress)
    else:
        M_sum = _accumulate(cfg, draws, E, grid, show_progress, reference)

    return _finalize(M_sum / max(n_real, 1), n_real, grid, E, sigma, cfg)


# ---------------------------------------------------------------------------
# Optional parallel path (opt-in via cfg.n_workers > 1), mirrors spectrum.py
# ---------------------------------------------------------------------------
_WORKER: Dict = {}
_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _worker_init(cfg: Config, E: np.ndarray, grid: np.ndarray) -> None:
    _WORKER["cfg"] = cfg
    _WORKER["E"] = E
    _WORKER["grid"] = grid
    # One diagonalization per worker process, reused for every task it handles.
    _WORKER["reference"] = vibronic.diagonalize_single_molecule_reference(cfg)


def _worker_task(task):
    eps1_dis, eps2_dis = task
    cfg, E, grid = _WORKER["cfg"], _WORKER["E"], _WORKER["grid"]
    sol = solve_realization(cfg, eps1_dis, eps2_dis, reference=_WORKER["reference"])
    return _energy_map(sol, cfg, E, grid)


def _accumulate_parallel(cfg, draws, E, grid, show_progress) -> np.ndarray:
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    saved = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
    for k in _THREAD_ENV_VARS:
        os.environ[k] = "1"

    M = np.zeros((len(grid), len(E)))
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=cfg.n_workers, mp_context=ctx,
            initializer=_worker_init, initargs=(cfg, E, grid),
        ) as ex:
            it = ex.map(_worker_task, draws, chunksize=1)
            if show_progress:
                it = tqdm(it, total=len(draws),
                          desc=f"  heatmap disorder(Nv={cfg.heatmap_nv})",
                          leave=False, unit="real")
            for m in it:
                M += m
    finally:
        for k, val in saved.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
    return M
