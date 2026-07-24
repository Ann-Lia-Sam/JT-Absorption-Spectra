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
  closest to each of several physically meaningful targets (Cases A-H).
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
# Automatic selection of representative realizations (Cases A-H)
#
# Selection is done by PHYSICAL CRITERIA on each realization's *actual* drawn
# site energies (eps1, eps2), never by "nearest to an invented target
# coordinate". Each case is either
#
#   * a pure ranking  (argmin/argmax of a physically motivated score), or
#   * a hard boolean filter  (e.g. "opposite-sign disorder") followed by a
#     ranking *within* the realizations that pass the filter.
#
# RESONANCE REFERENCE. A molecule is cavity-resonant when its *bright vibronic
# transition* equals omega_c. Because omega_c was tuned to the clean bright peak
# (eps0=cfg.eps -> bright ~ omega_c; verified: at eps_i=7.0 the bright line is
# 6.85=omega_c), that condition is eps_i ~= eps0, NOT eps_i ~= omega_c. Every
# case A-G below is therefore written in the signed detuning-from-resonance
#   d_i = eps_i - eps0                    (c.d1, c.d2)
# and the disorder scale is compared to the per-molecule coupling g = cfg.g.
# Case H is the deliberate exception: it targets ``omega_c`` itself (``wc``/
# ``det1``/``det2`` in _Ctx) -- see its comment in build_cases() below.
#
# The set spans two orthogonal axes: the common-mode shift s=(d1+d2)/2 (both
# molecules move together; cases A/D/E/G) and the inter-molecular mismatch
# |eps1-eps2|=|d1-d2| (cases B/C/F) -- so each case probes a distinct region.
# Case H is an interior point ON the A-C line (the s~0 anti-diagonal, C-ward
# branch eps1<eps0<eps2) closest to the omega_c dotted line -- i.e. between the
# resonant baseline A and the far opposite-disorder point C.
#
# The 8 cases are processed in order A -> H. A shared ``used_indices`` set
# guarantees every case selects a *distinct* realization: each case takes the
# best-scoring realization that has not already been claimed. If a case's
# boolean filter admits no (unused) realization -- possible for a finite
# sample when disorder is weak -- we fall back to ranking over all still-unused
# realizations and flag ``filter_satisfied`` so the fallback is visible rather
# than silent.
#
# Physical constants used (all derived from cfg, none invented):
#   eps0    = cfg.eps      clean transition energy == cavity-resonance point
#   wc      = cfg.omega_c  cavity photon energy (unused by A-G; targeted by H)
#   g       = cfg.g        per-molecule matter-cavity coupling (the disorder ruler)
#   Omega   = cfg.Omega    collective light-matter coupling scale
# ---------------------------------------------------------------------------
@dataclass
class _Ctx:
    """Vectorized realization quantities + physical constants for scoring."""
    eps1: np.ndarray      # drawn site energy, molecule 1
    eps2: np.ndarray      # drawn site energy, molecule 2
    d1: np.ndarray        # eps1 - eps0 (signed detuning from resonance)
    d2: np.ndarray        # eps2 - eps0
    det1: np.ndarray      # |eps1 - wc|  (cavity-energy detuning; used only by case H)
    det2: np.ndarray      # |eps2 - wc|  (cavity-energy detuning; used only by case H)
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
    """The 8 physical cases A-H (see module docstring for the selection rules).

    Resonance is referenced to the clean transition energy ``eps0 = cfg.eps``
    (NOT to ``cfg.omega_c``): a molecule is cavity-resonant when its bright
    vibronic transition equals ``omega_c``, which happens when ``eps_i ~= eps0``
    -- the clean value -- because ``omega_c`` was tuned to the clean bright peak.
    Every criterion A-G below is therefore written in the signed detuning-from-
    resonance ``d_i = eps_i - eps0`` (``c.d1`` / ``c.d2``), and the disorder
    scale is always compared to the per-molecule coupling ``g = cfg.g``. Case H
    is the deliberate exception, targeting ``omega_c`` directly.

    The set spans two orthogonal axes -- the common-mode shift
    ``s = (d1 + d2)/2`` (both molecules move together) and the inter-molecular
    mismatch ``|eps1 - eps2| = |d1 - d2|`` (``c.gap``) -- so each case probes a
    distinct region of the disorder plane. Case H lives on the same anti-
    diagonal (``s ~ 0``) as A and C -- it is the point on the A-C line closest
    to the scatter plot's dotted ``omega_c`` reference lines.
    """
    return [
        # A -- Resonant baseline (no disorder): both molecules at eps0, so both
        #      sit on cavity resonance. Minimize the radial deviation
        #      sqrt(d1^2 + d2^2). Reference realization; maximal collective cascade.
        CaseSpec("A", "Resonant baseline",
                 "eps1 ~ eps and eps2 ~ eps: both on cavity resonance (no disorder)",
                 "min", lambda c: np.hypot(c.d1, c.d2)),

        # B -- Large mismatch / localized: a large SAME-SIGN gap, i.e. one
        #      molecule near resonance and the other far detuned on the same side
        #      -- the excitation localizes on the less-detuned molecule.
        CaseSpec("B", "Large mismatch (localized)",
                 "|eps1 - eps2| >> g, same sign: excitation localizes on one molecule",
                 "max", lambda c: c.gap,
                 mask=lambda c: (c.d1 * c.d2) > 0),

        # C -- Opposite disorder (anti-correlated): d1 and d2 have opposite signs
        #      (net shift s ~ 0). Maximize the smaller |d_i| so BOTH molecules sit
        #      as far as possible from resonance, in opposite directions.
        CaseSpec("C", "Opposite disorder",
                 "(eps1 - eps)(eps2 - eps) < 0: molecules detuned oppositely (s ~ 0)",
                 "max", lambda c: np.minimum(np.abs(c.d1), np.abs(c.d2)),
                 mask=lambda c: (c.d1 * c.d2) < 0),

        # D -- Common blue detuning (matched): both molecules ~ +g above
        #      resonance, staying matched. Minimize distance to (d1, d2) = (g, g).
        CaseSpec("D", "Common blue detuning",
                 "eps1 ~ eps2 ~ eps + g: both blue-detuned together (symmetry intact)",
                 "min", lambda c: np.hypot(c.d1 - c.g, c.d2 - c.g)),

        # E -- Common red detuning (matched): both molecules ~ -g below
        #      resonance. Minimize distance to (d1, d2) = (-g, -g).
        CaseSpec("E", "Common red detuning",
                 "eps1 ~ eps2 ~ eps - g: both red-detuned together (symmetry intact)",
                 "min", lambda c: np.hypot(c.d1 + c.g, c.d2 + c.g)),

        # F -- Single-molecule resonant (asymmetric): one molecule on resonance
        #      (d ~ 0) while the other is detuned by at least ~ g. Minimize the
        #      smaller |d_i|, requiring the larger to exceed g.
        CaseSpec("F", "Single-molecule resonant",
                 "one molecule on resonance while the other is detuned by >~ g",
                 "min", lambda c: np.minimum(np.abs(c.d1), np.abs(c.d2)),
                 mask=lambda c: np.maximum(np.abs(c.d1), np.abs(c.d2)) > c.g),

        # G -- Bare-molecule limit: both molecules far off-resonance on the SAME
        #      side, |eps_i - eps0| >> g -- both decouple from the cavity (no
        #      polariton). Maximize the smaller |d_i| among same-sign pairs.
        CaseSpec("G", "Bare-molecule limit",
                 "both |eps_i - eps| >> g, same sign: both decouple from the cavity",
                 "max", lambda c: np.minimum(np.abs(c.d1), np.abs(c.d2)),
                 mask=lambda c: (c.d1 * c.d2) > 0),

        # H -- A-C line, closest to omega_c: a representative point *between* A
        #      and C on the line joining them. That line is the s=0 anti-diagonal
        #      eps1 + eps2 = 2*eps0 taken in the C-ward direction
        #      (eps1 < eps0 < eps2, i.e. d1 < 0 < d2 -- the SAME opposite-disorder
        #      branch C lives on, NOT its mirror). A sits at its origin (d1=d2=0)
        #      and C far out; case H is the interior point on that segment nearest
        #      the cavity. Its score combines two distances, both in eV:
        #        * |d1 + d2| = 2|s|  -- the *perpendicular distance to the A-C
        #          line* (how well the point sits ON the A-C line), and
        #        * min(|eps1 - wc|, |eps2 - wc|) -- proximity of the nearer
        #          molecule to the omega_c dotted line in the scatter plot.
        #      Minimizing their sum picks the on-line point closest to omega_c
        #      (A and C themselves are already claimed, so H is a distinct
        #      interior realization). This case is deliberately referenced to
        #      omega_c (via det1/det2), unlike the eps0-referenced cases A-G.
        CaseSpec("H", "A-C line, closest to omega_c",
                 "between A and C on the line joining them (eps1 < eps < eps2, "
                 "s ~ 0), the point nearest the omega_c dotted line: minimize "
                 "|d1+d2| + min(|eps1-wc|, |eps2-wc|)",
                 "min", lambda c: np.abs(c.d1 + c.d2) + np.minimum(c.det1, c.det2),
                 mask=lambda c: (c.d1 < 0) & (c.d2 > 0)),
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
    """Select one *distinct* realization for each physical case A-H.

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

    Only the (typically 8) *selected* representative realizations are
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
    (typically 8) expensive ``eigh`` calls entirely.

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
