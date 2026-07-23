"""Disorder averaging and Lorentzian broadening.

Two entry points:

* :func:`compute_spectrum_for_Nv` — one spectrum for a single ``Nv`` at the
  configured ``sigma`` (used by the Nv sweep).
* :func:`compute_spectrum_sigma_sweep` — several spectra for a single ``Nv``,
  one per disorder strength ``sigma``. The static Hamiltonian does not depend
  on ``sigma``, so it is built once and reused across the whole sweep.
"""

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


def lorentzian(x, x0, gamma):
    return (1 / np.pi) * (0.5 * gamma) / ((x - x0) ** 2 + (0.5 * gamma) ** 2)


VALID_NORMALIZATIONS = ("reference", "reference_area", "area", "peak", "none")


def _check_normalization(cfg: Config) -> None:
    if cfg.NORMALIZATION not in VALID_NORMALIZATIONS:
        raise ValueError(
            f"cfg.NORMALIZATION={cfg.NORMALIZATION!r} must be one of "
            f"{VALID_NORMALIZATIONS}"
        )


def _normalize_spectrum(
    E: np.ndarray,
    spectrum: np.ndarray,
    cfg: Config,
    reference_max: float = None,
    reference_area: float = None,
) -> np.ndarray:
    """Apply the single configured normalization to a fully averaged spectrum.

    Called exactly once, after Lorentzian broadening and disorder averaging
    are already complete -- never on an individual realization or on an
    individual disorder spectrum's own peak.

    * ``"reference"`` -- divide by ``reference_max`` (the peak of this Nv's
      σ=0 spectrum), the same value shared by every sigma/realization-count.
    * ``"reference_area"`` -- divide by ``reference_area`` (the trapezoidal
      area of this Nv's σ=0 spectrum), the same value shared by every
      sigma/realization-count -- same reference spectrum as ``"reference"``,
      a different summary statistic taken from it.
    * ``"area"`` -- divide this spectrum by its own trapezoidal integral over
      ``E``, so ``∫I(E)dE = 1``.
    * ``"peak"`` -- divide this spectrum by its own maximum, so it peaks at 1.
      Unlike ``"reference"`` (a single shared σ=0 peak), every curve is scaled
      by its *own* peak, so each one peaks at exactly 1 and only the lineshape
      -- not the disorder-driven peak reduction -- is compared.
    * ``"none"`` -- return the spectrum unchanged.
    """
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
    if mode == "peak":
        peak = spectrum.max()
        if peak > 0:
            return spectrum / peak
        return spectrum
    if mode == "none":
        return spectrum
    raise ValueError(
        f"cfg.NORMALIZATION={mode!r} must be one of {VALID_NORMALIZATIONS}"
    )


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
def _prepare_Nv(Nv: int, cfg: Config, show_progress: bool):
    """Build the sector basis and disorder-independent Hamiltonian for one Nv.

    Returns ``(H, mask1, mask2, i0, dim)``.
    """
    basis_final, basis_index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
    dim = len(basis_final)
    if dim == 0:
        raise ValueError(
            f"Empty sector for Nv={Nv} (Nex={cfg.nex_target}, Jz={cfg.jz_target})."
        )
    if cfg.initial_state not in basis_index:
        raise ValueError(
            f"Initial state {cfg.initial_state} not in the Nv={Nv} sector basis."
        )
    i0 = basis_index[cfg.initial_state]

    H = build_static_hamiltonian(Nv, basis_final, basis_index, cfg, show_progress)
    mask1, mask2 = electronic_masks(basis_final)
    return H, mask1, mask2, i0, dim


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
    """Average over disorder realizations at a given ``sigma``.

    The electronic diagonal is added in place before ``eigh`` and restored
    afterwards, so ``H`` is left unchanged for reuse across sigmas. Pass
    ``n_realizations`` to override ``cfg.n_realizations`` (used by the
    realization sweep). Because the RNG is seeded once, the first ``n`` rows of
    a longer run are identical to an independent ``n``-realization run.
    """
    n_real = cfg.n_realizations if n_realizations is None else n_realizations
    dim = H.shape[0]
    d = np.arange(dim)
    rng = np.random.default_rng(cfg.rng_seed)

    all_evals = np.empty((n_real, dim))
    all_intensity = np.empty((n_real, dim))

    iterator = range(n_real)
    if show_progress:
        iterator = tqdm(iterator, desc=desc, leave=False, unit="real")

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

    return all_evals, all_intensity


def _broaden(all_evals: np.ndarray, all_intensity: np.ndarray, cfg: Config):
    """Lorentzian-broaden each realization and average over realizations.

    Returns ``(E, spectrum)`` with ``spectrum`` the raw, disorder-averaged
    intensity (summed over realizations, divided by the realization count).
    No normalization is applied here -- that happens once, at the call site,
    using the zero-disorder (σ=0) spectrum's peak as the common reference.
    """
    E = np.linspace(cfg.E_min, cfg.E_max, cfg.E_points)
    spectrum = np.zeros_like(E)
    for evals, intensity in zip(all_evals, all_intensity):
        for En, I in zip(evals, intensity):
            spectrum += I * lorentzian(E, En, cfg.gamma)
    n_real = len(all_evals)
    if n_real > 0:
        spectrum = spectrum / n_real
    return E, spectrum


# ---------------------------------------------------------------------------
# Optional parallel path (opt-in via cfg.n_workers > 1)
#
# The serial code above is untouched and remains the default. When enabled,
# disorder realizations run across worker processes. To keep results the same
# as serial, the disorder draws are generated once in the parent (in the same
# order as the serial loop) and scattered to the workers. Each worker is pinned
# to a single BLAS thread (a single dense ``eigh`` barely benefits from more),
# so N workers give roughly an N-fold speedup on independent realizations.
# ---------------------------------------------------------------------------
_WORKER: Dict = {}

# Environment variables that control BLAS / threading backends.
_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _parallel_enabled(cfg: Config) -> bool:
    n = getattr(cfg, "n_workers", None)
    return bool(n) and n > 1


def _worker_init(Nv: int, cfg: Config) -> None:
    """Build the disorder-independent Hamiltonian once per worker process."""
    basis_final, basis_index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
    H = build_static_hamiltonian(Nv, basis_final, basis_index, cfg, show_progress=False)
    mask1, mask2 = electronic_masks(basis_final)
    _WORKER["H"] = H
    _WORKER["mask1"] = mask1
    _WORKER["mask2"] = mask2
    _WORKER["i0"] = basis_index[cfg.initial_state]
    _WORKER["d"] = np.arange(H.shape[0])


def _worker_task(task):
    """Diagonalize one realization; mirrors the serial loop body exactly."""
    r, eps1_dis, eps2_dis = task
    H = _WORKER["H"]
    d = _WORKER["d"]
    elec = electronic_diagonal(_WORKER["mask1"], _WORKER["mask2"], eps1_dis, eps2_dis)

    H[d, d] += elec
    evals, evecs = np.linalg.eigh(H)
    H[d, d] -= elec

    intensity = np.abs(evecs[_WORKER["i0"], :]) ** 2
    return r, evals, intensity


def _disorder_average_parallel(
    Nv: int,
    sigma: float,
    cfg: Config,
    show_progress: bool,
    desc: str,
    n_realizations: int = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parallel version of :func:`_disorder_average`. Returns (evals, I, dim).

    Disorder draws are generated in the parent in the same order as the serial
    loop, so the set of realizations is identical to the serial path.
    """
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    n_real = cfg.n_realizations if n_realizations is None else n_realizations

    # Same RNG stream and draw order as the serial loop.
    rng = np.random.default_rng(cfg.rng_seed)
    tasks = []
    for r in range(n_real):
        eps1_dis = cfg.eps1 + rng.normal(0, sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, sigma)
        tasks.append((r, eps1_dis, eps2_dis))

    # Pin BLAS to one thread per worker; spawned children inherit the env.
    saved = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
    for k in _THREAD_ENV_VARS:
        os.environ[k] = "1"

    all_evals = [None] * n_real
    all_intensity = [None] * n_real
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=cfg.n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(Nv, cfg),
        ) as ex:
            iterator = ex.map(_worker_task, tasks, chunksize=1)
            if show_progress:
                iterator = tqdm(iterator, total=n_real, desc=desc, leave=False, unit="real")
            for r, evals, intensity in iterator:
                all_evals[r] = evals
                all_intensity[r] = intensity
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    dim = all_evals[0].shape[0]
    return np.array(all_evals), np.array(all_intensity), dim


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def compute_spectrum_for_Nv(Nv: int, cfg: Config, show_progress: bool = True) -> Dict:
    """Disorder-averaged absorption spectrum for one ``Nv`` at ``cfg.sigma``.

    Normalized per ``cfg.NORMALIZATION`` (see :func:`_normalize_spectrum`).
    """
    _check_normalization(cfg)
    need_ref = cfg.NORMALIZATION in ("reference", "reference_area")

    if _parallel_enabled(cfg):
        all_evals, all_intensity, dim = _disorder_average_parallel(
            Nv, cfg.sigma, cfg, show_progress, f"  disorder(Nv={Nv})"
        )
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

    E, spectrum = _broaden(all_evals, all_intensity, cfg)
    reference_max = None
    reference_area = None
    if need_ref:
        _, ref_spectrum = _broaden(ref_evals, ref_intensity, cfg)
        reference_max = ref_spectrum.max()
        reference_area = np.trapezoid(ref_spectrum, E)
    spectrum = _normalize_spectrum(E, spectrum, cfg, reference_max, reference_area)

    return {
        "Nv": Nv,
        "sigma": cfg.sigma,
        "dim": dim,
        "E": E,
        "spectrum": spectrum,
        "reference_max": reference_max,
        "reference_area": reference_area,
        "normalization": cfg.NORMALIZATION,
        "all_evals": all_evals,
        "all_intensity": all_intensity,
    }


def compute_spectrum_sigma_sweep(
    Nv: int,
    sigma_list: Sequence[float],
    cfg: Config,
    show_progress: bool = True,
) -> List[Dict]:
    """Compute one spectrum per disorder strength ``sigma`` for a fixed ``Nv``.

    The static Hamiltonian is built once and reused for every ``sigma``; only
    the disorder loop reruns. If ``cfg.NORMALIZATION`` is ``"reference"`` or
    ``"reference_area"``, the zero-disorder (σ=0) spectrum for this ``Nv`` is
    also computed once (with the same ``cfg.n_realizations`` as every other
    spectrum), and its raw peak (``reference_max``) / area (``reference_area``)
    is used to normalize every sigma's spectrum -- so the peak reduction
    caused by disorder stays visible instead of being normalized away per
    curve. Returns a list of result dicts (one per sigma), each shaped like
    :func:`compute_spectrum_for_Nv`'s output plus a ``sigma`` key.
    """
    _check_normalization(cfg)
    need_ref = cfg.NORMALIZATION in ("reference", "reference_area")

    parallel = _parallel_enabled(cfg)
    if not parallel:
        H, mask1, mask2, i0, dim = _prepare_Nv(Nv, cfg, show_progress)

    reference_max = None
    reference_area = None
    if need_ref:
        if parallel:
            ref_evals, ref_intensity, _ = _disorder_average_parallel(
                Nv, 0.0, cfg, show_progress, f"  reference(Nv={Nv})",
                n_realizations=cfg.n_realizations,
            )
        else:
            ref_evals, ref_intensity = _disorder_average(
                H, mask1, mask2, i0, 0.0, cfg, show_progress,
                f"  reference(Nv={Nv})", n_realizations=cfg.n_realizations,
            )
        E_ref, ref_spectrum = _broaden(ref_evals, ref_intensity, cfg)
        reference_max = ref_spectrum.max()
        reference_area = np.trapezoid(ref_spectrum, E_ref)

    sigmas = list(sigma_list)
    outer = sigmas
    if show_progress:
        outer = tqdm(sigmas, desc=f"sigma sweep(Nv={Nv})", unit="sigma")

    results: List[Dict] = []
    for sigma in outer:
        if parallel:
            all_evals, all_intensity, dim = _disorder_average_parallel(
                Nv, sigma, cfg, show_progress, f"  disorder(sigma={sigma:g})"
            )
        else:
            all_evals, all_intensity = _disorder_average(
                H, mask1, mask2, i0, sigma, cfg, show_progress, f"  disorder(sigma={sigma:g})"
            )
        E, spectrum = _broaden(all_evals, all_intensity, cfg)
        spectrum = _normalize_spectrum(E, spectrum, cfg, reference_max, reference_area)
        results.append(
            {
                "Nv": Nv,
                "sigma": sigma,
                "dim": dim,
                "E": E,
                "spectrum": spectrum,
                "reference_max": reference_max,
                "reference_area": reference_area,
                "normalization": cfg.NORMALIZATION,
                "all_evals": all_evals,
                "all_intensity": all_intensity,
            }
        )

    return results


def compute_spectrum_realization_sweep(
    Nv: int,
    realization_list: Sequence[int],
    cfg: Config,
    show_progress: bool = True,
) -> List[Dict]:
    """Compute one spectrum per disorder-realization count for a fixed ``Nv``.

    Shows how the disorder average converges with the number of realizations.
    The disorder loop runs **once** at ``max(realization_list)`` and each smaller
    count reuses the leading rows (identical to an independent shorter run since
    the RNG is seeded once). Returns a list of result dicts, one per realization
    count, each with an ``n_realizations`` key.
    """
    _check_normalization(cfg)
    need_ref = cfg.NORMALIZATION in ("reference", "reference_area")

    counts = sorted(set(int(n) for n in realization_list))
    n_max = counts[-1]

    # The reference spectrum uses the same realization count as the main
    # disorder pass for this sweep (n_max), matching "the other spectra".
    if _parallel_enabled(cfg):
        all_evals, all_intensity, dim = _disorder_average_parallel(
            Nv, cfg.sigma, cfg, show_progress,
            f"  disorder(Nv={Nv}, n={n_max})", n_realizations=n_max,
        )
        if need_ref:
            ref_evals, ref_intensity, _ = _disorder_average_parallel(
                Nv, 0.0, cfg, show_progress, f"  reference(Nv={Nv})", n_realizations=n_max
            )
    else:
        H, mask1, mask2, i0, dim = _prepare_Nv(Nv, cfg, show_progress)
        all_evals, all_intensity = _disorder_average(
            H, mask1, mask2, i0, cfg.sigma, cfg, show_progress,
            f"  disorder(Nv={Nv}, n={n_max})", n_realizations=n_max,
        )
        if need_ref:
            ref_evals, ref_intensity = _disorder_average(
                H, mask1, mask2, i0, 0.0, cfg, show_progress,
                f"  reference(Nv={Nv})", n_realizations=n_max,
            )

    reference_max = None
    reference_area = None
    if need_ref:
        E_ref, ref_spectrum = _broaden(ref_evals, ref_intensity, cfg)
        reference_max = ref_spectrum.max()
        reference_area = np.trapezoid(ref_spectrum, E_ref)

    results: List[Dict] = []
    for n in counts:
        E, spectrum = _broaden(all_evals[:n], all_intensity[:n], cfg)
        spectrum = _normalize_spectrum(E, spectrum, cfg, reference_max, reference_area)
        results.append(
            {
                "Nv": Nv,
                "sigma": cfg.sigma,
                "n_realizations": n,
                "dim": dim,
                "E": E,
                "spectrum": spectrum,
                "reference_max": reference_max,
                "reference_area": reference_area,
                "normalization": cfg.NORMALIZATION,
                "all_evals": all_evals[:n],
                "all_intensity": all_intensity[:n],
            }
        )

    return results
