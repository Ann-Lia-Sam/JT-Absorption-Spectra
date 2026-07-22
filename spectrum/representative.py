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
from .spectrum import (
    _broaden,
    _check_normalization,
    _disorder_average,
    _disorder_average_parallel,
    _normalize_spectrum,
    _parallel_enabled,
    _prepare_Nv,
    compute_spectrum_for_Nv,
)


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
# Automatic selection of representative realizations (Cases A-K)
#
# Selection is done by PHYSICAL CRITERIA on each realization's *actual* drawn
# site energies (eps1, eps2), never by "nearest to an invented target
# coordinate". Each case is either
#
#   * a pure ranking  (argmin/argmax of a physically motivated score), or
#   * a hard boolean filter  (e.g. "both above resonance") followed by a
#     ranking *within* the realizations that pass the filter.
#
# The 11 cases are processed in order A -> K. A shared ``used_indices`` set
# guarantees every case selects a *distinct* realization: each case takes the
# best-scoring realization that has not already been claimed. If a case's
# boolean filter admits no (unused) realization -- possible for a finite
# sample, e.g. "both below resonance" when disorder is weak -- we fall back to
# ranking over all still-unused realizations and flag ``filter_satisfied`` so
# the fallback is visible rather than silent.
#
# Physical constants used (all derived from cfg, none invented):
#   eps0    = cfg.eps      clean transition (bare-molecule) energy
#   wc      = cfg.omega_c  cavity photon (resonance) energy
#   g       = cfg.g        per-molecule matter-cavity coupling
#   Omega   = cfg.Omega    collective light-matter coupling scale
# ---------------------------------------------------------------------------
@dataclass
class _Ctx:
    """Vectorized realization quantities + physical constants for scoring."""
    eps1: np.ndarray      # drawn site energy, molecule 1
    eps2: np.ndarray      # drawn site energy, molecule 2
    d1: np.ndarray        # eps1 - eps0 (deviation from clean energy)
    d2: np.ndarray        # eps2 - eps0
    det1: np.ndarray      # |eps1 - wc|  (detuning of molecule 1 from the cavity)
    det2: np.ndarray      # |eps2 - wc|
    gap: np.ndarray       # |eps1 - eps2|  (inter-molecular energy mismatch)
    eps0: float
    wc: float
    g: float
    Omega: float


@dataclass
class CaseSpec:
    """One physical-criteria case.

    ``score`` maps a :class:`_Ctx` to a per-realization array; the case selects
    the ``argmin`` (``sense="min"``) or ``argmax`` (``sense="max"``) of that
    score. ``mask``, if given, is a hard boolean requirement a realization must
    satisfy to be eligible (ranking then happens only among those that pass).
    """
    letter: str
    name: str
    description: str
    sense: str                                   # "min" or "max"
    score: "callable"                            # _Ctx -> np.ndarray
    mask: Optional["callable"] = None            # _Ctx -> np.ndarray[bool]


def build_cases() -> List[CaseSpec]:
    """The 11 physical cases A-K (see module docstring for the selection rules)."""
    return [
        # A -- Nearly no disorder: both molecules sit at the clean energy.
        #      Minimize the radial deviation sqrt(d1^2 + d2^2).
        CaseSpec("A", "Nearly no disorder",
                 "eps1 ~ eps and eps2 ~ eps (smallest overall disorder)",
                 "min", lambda c: np.hypot(c.d1, c.d2)),

        # B -- Both molecules resonant: both detunings from the cavity small.
        #      Minimize distance of (eps1, eps2) to (wc, wc), which drives BOTH
        #      |eps1 - wc| and |eps2 - wc| down together.
        CaseSpec("B", "Both molecules resonant",
                 "both |eps1 - wc| and |eps2 - wc| minimized (near cavity resonance)",
                 "min", lambda c: np.hypot(c.det1, c.det2)),

        # C -- Molecule 1 resonant: |eps1 - wc| small while |eps2 - wc| large.
        #      Minimize det1 - det2 (small det1, large det2).
        CaseSpec("C", "Molecule 1 resonant",
                 "|eps1 - wc| small while |eps2 - wc| as large as possible",
                 "min", lambda c: c.det1 - c.det2),

        # D -- Molecule 2 resonant: reverse of C.
        CaseSpec("D", "Molecule 2 resonant",
                 "|eps2 - wc| small while |eps1 - wc| as large as possible",
                 "min", lambda c: c.det2 - c.det1),

        # E -- Both above resonance: hard requirement eps1 > wc AND eps2 > wc.
        #      Among those, pick the one sitting most clearly above (maximize
        #      the smaller of the two margins above wc).
        CaseSpec("E", "Both above resonance",
                 "eps1 > wc and eps2 > wc",
                 "max", lambda c: np.minimum(c.eps1 - c.wc, c.eps2 - c.wc),
                 mask=lambda c: (c.eps1 > c.wc) & (c.eps2 > c.wc)),

        # F -- Both below resonance: hard requirement eps1 < wc AND eps2 < wc.
        #      Maximize the smaller margin below wc (both comfortably below).
        CaseSpec("F", "Both below resonance",
                 "eps1 < wc and eps2 < wc",
                 "max", lambda c: np.minimum(c.wc - c.eps1, c.wc - c.eps2),
                 mask=lambda c: (c.eps1 < c.wc) & (c.eps2 < c.wc)),

        # G -- Opposite disorder: molecules deviate in opposite directions from
        #      the clean energy, (d1)(d2) < 0. Maximize |eps1 - eps2|.
        CaseSpec("G", "Opposite disorder",
                 "(eps1 - eps)(eps2 - eps) < 0, maximizing |eps1 - eps2|",
                 "max", lambda c: c.gap,
                 mask=lambda c: (c.d1 * c.d2) < 0),

        # H -- Very large disorder (bare-molecule limit): both molecules far
        #      from the cavity, |eps_i - wc| >> g. Maximize min(det1, det2) --
        #      the honest finite-sample "furthest both-detuned" realization.
        CaseSpec("H", "Very large disorder (bare-molecule limit)",
                 "both |eps_i - wc| >> g (both molecules far off-resonance)",
                 "max", lambda c: np.minimum(c.det1, c.det2)),

        # I -- Resonance mismatch: inter-molecular gap comparable to the
        #      coupling, |eps1 - eps2| ~ g -- the collective-to-localized
        #      crossover. Minimize | |eps1 - eps2| - g |.
        CaseSpec("I", "Resonance mismatch (crossover)",
                 "|eps1 - eps2| ~ g (collective-to-localized crossover)",
                 "min", lambda c: np.abs(c.gap - c.g)),

        # J -- Nearly degenerate molecules: minimize |eps1 - eps2| (they may
        #      both be shifted far from resonance, but track each other).
        CaseSpec("J", "Nearly degenerate molecules",
                 "|eps1 - eps2| minimized (molecules nearly degenerate)",
                 "min", lambda c: c.gap),

        # K -- Maximum energy mismatch: strongest symmetry breaking.
        CaseSpec("K", "Maximum energy mismatch",
                 "|eps1 - eps2| maximized (strongest symmetry breaking)",
                 "max", lambda c: c.gap),
    ]


def _build_ctx(metadata: List[RealizationMetadata], cfg: Config) -> _Ctx:
    eps1 = np.array([m.eps1 for m in metadata])
    eps2 = np.array([m.eps2 for m in metadata])
    eps0, wc = cfg.eps, cfg.omega_c
    return _Ctx(
        eps1=eps1, eps2=eps2,
        d1=eps1 - eps0, d2=eps2 - eps0,
        det1=np.abs(eps1 - wc), det2=np.abs(eps2 - wc),
        gap=np.abs(eps1 - eps2),
        eps0=eps0, wc=wc, g=cfg.g, Omega=cfg.Omega,
    )


def select_representative_realizations(
    metadata: List[RealizationMetadata], cfg: Config, sigma: Optional[float] = None
) -> Dict[str, Dict]:
    """Select one *distinct* realization for each physical case A-K.

    Cases are evaluated in order; a shared ``used_indices`` set enforces
    uniqueness (each case claims the best-scoring realization not yet taken).
    A case with a boolean filter that no unused realization satisfies falls
    back to ranking over all unused realizations, with ``filter_satisfied``
    set to ``False`` on that selection so the fallback is explicit.

    Returns an (insertion-ordered) dict keyed by ``"<letter>: <name>"`` -> a
    dict with ``case``, ``letter``, ``description``, ``metadata`` (the selected
    :class:`RealizationMetadata`), ``score`` (the winning realization's score),
    and ``filter_satisfied``.
    """
    if not metadata:
        raise ValueError("No realization metadata to select representative cases from.")

    ctx = _build_ctx(metadata, cfg)
    n = len(metadata)

    used_indices: set = set()
    selections: Dict[str, Dict] = {}

    for spec in build_cases():
        scores = np.asarray(spec.score(ctx), dtype=float)

        # Eligibility: not yet used, and (if the case has one) passing its
        # physical filter.
        available = np.ones(n, dtype=bool)
        for i in used_indices:
            available[i] = False

        filter_satisfied = True
        eligible = available
        if spec.mask is not None:
            passes = np.asarray(spec.mask(ctx), dtype=bool)
            masked = available & passes
            if masked.any():
                eligible = masked
            else:
                # No unused realization satisfies the hard requirement in this
                # finite sample -- rank over all unused ones instead, and flag.
                filter_satisfied = False
                eligible = available

        if not eligible.any():
            # Fewer realizations than cases: nothing left to assign.
            break

        # argmin/argmax restricted to eligible realizations.
        masked_scores = np.where(eligible, scores, np.inf if spec.sense == "min" else -np.inf)
        idx = int(np.argmin(masked_scores) if spec.sense == "min" else np.argmax(masked_scores))
        used_indices.add(idx)

        key = f"{spec.letter}: {spec.name}"
        selections[key] = {
            "case": key,
            "letter": spec.letter,
            "description": spec.description,
            "metadata": metadata[idx],
            "score": float(scores[idx]),
            "filter_satisfied": filter_satisfied,
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
            "score": sel.get("score"),
            "filter_satisfied": sel.get("filter_satisfied"),
        })
        results[name] = res
    return results


def assemble_representative_spectra(
    Nv: int,
    selections: Dict[str, Dict],
    average: Dict,
    cfg: Config,
) -> Dict[str, Dict]:
    """Build each selected case's spectrum by *reusing* Pass 1's eigendata.

    Pass 1 (:func:`spectrum.spectrum.compute_spectrum_for_Nv`) already
    diagonalizes every disorder realization and returns the per-realization
    ``all_evals`` / ``all_intensity`` arrays. A selected case is just one of
    those realizations (identified by ``metadata.index``), so its spectrum is
    obtained by broadening the already-computed eigenvalues/intensities and
    normalizing with the ensemble's reference constants -- **no re-diagonalization
    at all**. This is bit-identical (to float round-off in the broadening
    summation) to :func:`recompute_representative_spectra`, but skips the
    (typically 11) expensive ``eigh`` calls entirely.

    Falls back to a one-shot recompute for any case whose realization index is
    out of range for the stored arrays (should not happen when ``average`` came
    from the same ``cfg``).
    """
    all_evals = average.get("all_evals")
    all_intensity = average.get("all_intensity")
    reference_max = average.get("reference_max")
    reference_area = average.get("reference_area")
    dim = int(average.get("dim", 0))

    have_eigendata = all_evals is not None and all_intensity is not None
    n_stored = len(all_evals) if have_eigendata else 0

    results: Dict[str, Dict] = {}
    H = mask1 = mask2 = i0 = None  # built lazily only if a fallback is needed
    for name, sel in selections.items():
        m: RealizationMetadata = sel["metadata"]
        if have_eigendata and 0 <= m.index < n_stored:
            E, spectrum = _broaden(
                all_evals[m.index][None, :], all_intensity[m.index][None, :], cfg
            )
            spectrum = _normalize_spectrum(E, spectrum, cfg, reference_max, reference_area)
            res = {"Nv": Nv, "dim": dim, "eps1": m.eps1, "eps2": m.eps2,
                   "E": E, "spectrum": spectrum}
        else:
            if H is None:
                basis, index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
                H = build_static_hamiltonian(Nv, basis, index, cfg, show_progress=False)
                mask1, mask2 = electronic_masks(basis)
                i0 = index[cfg.initial_state]
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
            "score": sel.get("score"),
            "filter_satisfied": sel.get("filter_satisfied"),
        })
        results[name] = res
    return results


# ---------------------------------------------------------------------------
# Efficient Pass 1 (avoids the frozen pipeline's redundant reference pass)
# ---------------------------------------------------------------------------
def compute_average_for_representative(
    Nv: int, cfg: Config, show_progress: bool = True
) -> Dict:
    """Disorder-averaged spectrum for the representative workflow.

    Behaves like :func:`spectrum.spectrum.compute_spectrum_for_Nv` (same result
    dict, same disorder realizations, same normalization), with one efficiency
    difference in the ``"reference"`` / ``"reference_area"`` normalization
    modes: the zero-disorder reference spectrum is computed from a **single**
    realization instead of ``cfg.n_realizations`` of them.

    This is exact, not an approximation: at ``sigma = 0`` the disorder draw
    ``rng.normal(0, 0)`` is identically ``0``, so every "reference realization"
    is the *same* deterministic no-disorder Hamiltonian. Averaging one copy or
    ``N`` identical copies yields the same reference peak/area (to floating-point
    summation round-off, ~1e-14 relative -- the same order as this codebase's
    existing serial-vs-parallel differences). Doing it once avoids
    re-diagonalizing an identical matrix ``cfg.n_realizations`` times, which for
    a large ``Nv`` roughly halves Pass 1's wall time.

    Uses only the frozen ``spectrum.py`` primitives, unmodified.
    """
    _check_normalization(cfg)
    need_ref = cfg.NORMALIZATION in ("reference", "reference_area")
    parallel = _parallel_enabled(cfg)

    if parallel:
        all_evals, all_intensity, dim = _disorder_average_parallel(
            Nv, cfg.sigma, cfg, show_progress, f"  disorder(Nv={Nv})"
        )
    else:
        H, mask1, mask2, i0, dim = _prepare_Nv(Nv, cfg, show_progress)
        all_evals, all_intensity = _disorder_average(
            H, mask1, mask2, i0, cfg.sigma, cfg, show_progress, f"  disorder(Nv={Nv})"
        )

    E, spectrum = _broaden(all_evals, all_intensity, cfg)

    reference_max = None
    reference_area = None
    if need_ref:
        # sigma=0 is deterministic -> one realization is exact (see docstring).
        if parallel:
            ref_evals, ref_intensity, _ = _disorder_average_parallel(
                Nv, 0.0, cfg, show_progress, f"  reference(Nv={Nv})", n_realizations=1
            )
        else:
            ref_evals, ref_intensity = _disorder_average(
                H, mask1, mask2, i0, 0.0, cfg, show_progress,
                f"  reference(Nv={Nv})", n_realizations=1,
            )
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


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def run_representative_analysis(Nv: int, cfg: Config, show_progress: bool = True) -> Dict:
    """Full two-pass workflow.

    Pass 1 computes the disorder-averaged spectrum with
    :func:`compute_average_for_representative` (same realizations and
    normalization as the default pipeline, but without its redundant
    ``cfg.n_realizations``-fold recomputation of the deterministic sigma=0
    reference) and separately collects lightweight metadata. Pass 2 then
    *reuses* the per-realization eigendata Pass 1 already produced to assemble
    each selected case's spectrum -- so no realization is ever diagonalized
    twice. (The older :func:`recompute_representative_spectra` path, which
    re-diagonalizes each case, remains available and is exercised by the tests
    as an independent cross-check.)
    """
    average = compute_average_for_representative(Nv, cfg, show_progress=show_progress)
    metadata = collect_realization_metadata(cfg)
    selections = select_representative_realizations(metadata, cfg)
    representative = assemble_representative_spectra(Nv, selections, average, cfg)
    return {"Nv": Nv, "average": average, "metadata": metadata, "representative": representative}
