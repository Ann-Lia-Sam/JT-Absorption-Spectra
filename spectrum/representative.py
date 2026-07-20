"""Representative-disorder-realization analysis.

Analyzes the *individual* disorder realizations that contribute to the
disorder-averaged absorption spectrum, without changing that spectrum's
physics. Two passes, kept deliberately memory-light:

* **Pass 1** ( :func:`run_representative_analysis` 's first half): call the
  existing, unmodified :func:`spectrum.spectrum.compute_spectrum_for_Nv` to get
  the disorder-averaged spectrum exactly as before, and separately record
  lightweight per-realization metadata -- ``(index, eps1, eps2, delta1,
  delta2)`` -- for every realization. No per-realization spectra are stored.
* **Selection**: from that metadata cloud, automatically pick one realization
  closest to each of several physically meaningful targets (Cases A-G).
* **Pass 2** ( :func:`recompute_representative_spectra` ): rebuild the static
  Hamiltonian once (as the existing sigma-sweep code already does) and
  recompute a full spectrum only for the handful of *selected* realizations --
  reusing the exact same basis / Hamiltonian / diagonalization / broadening /
  normalization primitives as the main pipeline, unmodified.

Nothing in this module alters ``spectrum.basis``, ``spectrum.operators``,
``spectrum.hamiltonian``, or ``spectrum.spectrum``.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .basis import build_sector_basis
from .config import Config
from .hamiltonian import build_static_hamiltonian, electronic_diagonal, electronic_masks
from .spectrum import _broaden, _normalize_spectrum, _check_normalization, compute_spectrum_for_Nv


# ---------------------------------------------------------------------------
# Pass 1 (addition): lightweight per-realization metadata
# ---------------------------------------------------------------------------
@dataclass
class RealizationMetadata:
    """Lightweight record for one disorder realization -- no spectra."""

    index: int
    eps1: float
    eps2: float
    delta1: float  # eps1 - cfg.eps: this realization's deviation from the clean transition energy
    delta2: float  # eps2 - cfg.eps


def collect_realization_metadata(
    cfg: Config, sigma: Optional[float] = None, n_real: Optional[int] = None
) -> List[RealizationMetadata]:
    """Reproduce the exact per-realization ``(eps1, eps2)`` draws made internally
    by :func:`spectrum.spectrum._disorder_average`, without storing any spectra.

    Mirrors -- rather than modifies or imports internals from -- that function's
    inline draw loop::

        rng = np.random.default_rng(cfg.rng_seed)
        eps1_dis = cfg.eps1 + rng.normal(0, sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, sigma)

    in the same order (molecule 1 then molecule 2, realization by realization),
    so realization ``r`` here is exactly realization ``r`` inside
    :func:`spectrum.spectrum.compute_spectrum_for_Nv`'s internal averaging loop,
    for the same ``(cfg.rng_seed, sigma, n_real)``. Verified bit-for-bit in
    ``tests/test_representative.py``.
    """
    sigma = cfg.sigma if sigma is None else sigma
    n_real = cfg.n_realizations if n_real is None else n_real
    rng = np.random.default_rng(cfg.rng_seed)

    metadata = []
    for r in range(n_real):
        eps1_dis = cfg.eps1 + rng.normal(0, sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, sigma)
        metadata.append(RealizationMetadata(
            index=r, eps1=eps1_dis, eps2=eps2_dis,
            delta1=eps1_dis - cfg.eps, delta2=eps2_dis - cfg.eps,
        ))
    return metadata


# ---------------------------------------------------------------------------
# Automatic selection of representative realizations (Cases A-G)
# ---------------------------------------------------------------------------
@dataclass
class CaseSpec:
    letter: str
    name: str
    description: str
    mode: str                      # "nearest" (to target) or "farthest" (from cloud center)
    target: Tuple[float, float]    # (eps1, eps2) target -- or cloud center, if mode="farthest"


def _case_specs(cfg: Config, sigma: float) -> List[CaseSpec]:
    """Case targets in (eps1, eps2) space, built from cfg.eps (clean transition
    energy), cfg.omega_c (cavity resonance), and the disorder strength ``sigma``.
    """
    eps0 = cfg.eps
    res = cfg.omega_c
    s = sigma
    return [
        CaseSpec("A", "Nearly no disorder", "eps1 ~ eps, eps2 ~ eps",
                 "nearest", (eps0, eps0)),
        CaseSpec("B", "Molecule 1 resonant", "eps1 ~ cavity resonance, eps2 significantly detuned",
                 "nearest", (res, eps0 + 2 * s)),
        CaseSpec("C", "Molecule 2 resonant", "reverse of case B",
                 "nearest", (eps0 + 2 * s, res)),
        CaseSpec("D", "Large opposite disorder", "eps1=eps+delta, eps2=eps-delta, large |delta|",
                 "nearest", (eps0 + 2 * s, eps0 - 2 * s)),
        CaseSpec("E", "Both shifted upward", "eps1, eps2 both above the clean transition energy",
                 "nearest", (eps0 + s, eps0 + s)),
        CaseSpec("F", "Both shifted downward", "eps1, eps2 both below the clean transition energy",
                 "nearest", (eps0 - s, eps0 - s)),
        CaseSpec("G", "Large disorder (edge of cloud)", "realization near the edge of the Gaussian cloud",
                 "farthest", (eps0, eps0)),
    ]


def select_representative_realizations(
    metadata: List[RealizationMetadata], cfg: Config, sigma: Optional[float] = None
) -> Dict[str, Dict]:
    """Pick, for each Case A-G, the sampled realization closest to its target
    (Case G instead picks the realization *farthest* from the cloud center --
    "near the edge of the cloud" has no single target point to be near).

    Returns an (insertion-ordered) dict keyed by ``"A: <name>"`` -> a small dict
    with ``case``, ``description``, and ``metadata`` (the selected
    :class:`RealizationMetadata`).
    """
    if not metadata:
        raise ValueError("No realization metadata to select representative cases from.")
    sigma = cfg.sigma if sigma is None else sigma
    eps1 = np.array([m.eps1 for m in metadata])
    eps2 = np.array([m.eps2 for m in metadata])

    selections: Dict[str, Dict] = {}
    for spec in _case_specs(cfg, sigma):
        t1, t2 = spec.target
        dist = np.sqrt((eps1 - t1) ** 2 + (eps2 - t2) ** 2)
        idx = int(np.argmax(dist)) if spec.mode == "farthest" else int(np.argmin(dist))
        key = f"{spec.letter}: {spec.name}"
        selections[key] = {
            "case": key,
            "letter": spec.letter,
            "description": spec.description,
            "metadata": metadata[idx],
        }
    return selections


# ---------------------------------------------------------------------------
# Pass 2: recompute only the selected representative realizations
# ---------------------------------------------------------------------------
def compute_single_realization_spectrum(
    Nv: int,
    eps1_dis: float,
    eps2_dis: float,
    cfg: Config,
    H: np.ndarray = None,
    mask1: np.ndarray = None,
    mask2: np.ndarray = None,
    i0: int = None,
    reference_max: float = None,
    reference_area: float = None,
) -> Dict:
    """One-shot spectrum for an explicit ``(eps1_dis, eps2_dis)`` pair.

    Performs exactly the per-realization body of
    :func:`spectrum.spectrum._disorder_average` for a single, explicit
    ``(eps1, eps2)`` pair instead of one drawn from the RNG, then broadens and
    normalizes with the *same* (private, reused-not-reimplemented) helpers
    ``spectrum.py`` uses internally -- so a representative realization's
    spectrum sits on an identical footing to the ensemble average. Pass a
    pre-built ``(H, mask1, mask2, i0)`` (e.g. from
    :func:`recompute_representative_spectra`) to avoid rebuilding the static
    Hamiltonian for every representative case.
    """
    _check_normalization(cfg)
    if H is None:
        basis, index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
        H = build_static_hamiltonian(Nv, basis, index, cfg, show_progress=False)
        mask1, mask2 = electronic_masks(basis)
        i0 = index[cfg.initial_state]

    dim = H.shape[0]
    d = np.arange(dim)
    elec = electronic_diagonal(mask1, mask2, eps1_dis, eps2_dis)
    H[d, d] += elec
    evals, evecs = np.linalg.eigh(H)
    H[d, d] -= elec
    intensity = np.abs(evecs[i0, :]) ** 2

    E, spectrum = _broaden(evals[None, :], intensity[None, :], cfg)
    spectrum = _normalize_spectrum(E, spectrum, cfg, reference_max, reference_area)

    return {"Nv": Nv, "dim": dim, "eps1": eps1_dis, "eps2": eps2_dis, "E": E, "spectrum": spectrum}


def recompute_representative_spectra(
    Nv: int,
    selections: Dict[str, Dict],
    cfg: Config,
    reference_max: float = None,
    reference_area: float = None,
    show_progress: bool = True,
) -> Dict[str, Dict]:
    """Build the static Hamiltonian once and reuse it for every selected case.

    Only the (typically 7) *selected* representative realizations are
    recomputed here -- never the full disorder ensemble -- keeping Pass 2's
    memory and compute footprint independent of ``cfg.n_realizations``.
    """
    basis, index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
    H = build_static_hamiltonian(Nv, basis, index, cfg, show_progress=show_progress)
    mask1, mask2 = electronic_masks(basis)
    i0 = index[cfg.initial_state]

    results: Dict[str, Dict] = {}
    for name, sel in selections.items():
        m: RealizationMetadata = sel["metadata"]
        res = compute_single_realization_spectrum(
            Nv, m.eps1, m.eps2, cfg, H=H, mask1=mask1, mask2=mask2, i0=i0,
            reference_max=reference_max, reference_area=reference_area,
        )
        res.update({
            "case": name,
            "letter": sel["letter"],
            "description": sel["description"],
            "realization_index": m.index,
            "delta1": m.delta1,
            "delta2": m.delta2,
        })
        results[name] = res
    return results


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def run_representative_analysis(Nv: int, cfg: Config, show_progress: bool = True) -> Dict:
    """Full two-pass workflow.

    Pass 1 calls the existing disorder-averaging pipeline completely unmodified
    (so the averaged spectrum is byte-identical to what the default pipeline
    would produce for this ``Nv``/``cfg``) and separately collects lightweight
    metadata; Pass 2 recomputes only the automatically selected representative
    realizations.
    """
    average = compute_spectrum_for_Nv(Nv, cfg, show_progress=show_progress)
    metadata = collect_realization_metadata(cfg)
    selections = select_representative_realizations(metadata, cfg)
    representative = recompute_representative_spectra(
        Nv, selections, cfg,
        reference_max=average.get("reference_max"),
        reference_area=average.get("reference_area"),
        show_progress=show_progress,
    )
    return {"Nv": Nv, "average": average, "metadata": metadata, "representative": representative}
