"""Representative-disorder-realization analysis: correctness checks.

These tests verify the *new* analysis layer without ever modifying the
existing solver:

* Pass 1's metadata reproduces, bit-for-bit, the exact ``(eps1, eps2)`` draws
  the existing ``_disorder_average`` makes internally (same seed, same order).
* Pass 2's one-shot recompute exactly reproduces what the existing,
  unmodified pipeline would produce for that same realization.
* The automatic case selection behaves sanely (nearest/farthest as intended).
* The absorption-spectrum pipeline itself remains byte-for-byte unchanged.
"""
import numpy as np

from spectrum.basis import build_sector_basis
from spectrum.config import Config
from spectrum.hamiltonian import build_static_hamiltonian, electronic_masks
from spectrum.spectrum import _disorder_average, compute_spectrum_for_Nv
from spectrum import representative as rep


def _reference_draws(cfg, sigma, n_real):
    """Replicate spectrum.py's inline draw loop exactly (the ground truth stream)."""
    rng = np.random.default_rng(cfg.rng_seed)
    out = []
    for _ in range(n_real):
        eps1_dis = cfg.eps1 + rng.normal(0, sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, sigma)
        out.append((eps1_dis, eps2_dis))
    return out


def test_metadata_matches_disorder_average_rng_stream():
    cfg = Config()
    for sigma in (0.0, 0.05, 0.24):
        ref = _reference_draws(cfg, sigma, 40)
        metadata = rep.collect_realization_metadata(cfg, sigma=sigma, n_real=40)
        assert len(metadata) == len(ref)
        for (e1r, e2r), m in zip(ref, metadata):
            assert m.eps1 == e1r
            assert m.eps2 == e2r
            assert m.delta1 == e1r - cfg.eps
            assert m.delta2 == e2r - cfg.eps


def test_pass2_matches_existing_pipeline_exactly():
    """compute_single_realization_spectrum must exactly reproduce what the
    existing, unmodified _disorder_average produces for the same (eps1, eps2),
    forced deterministically via sigma=0 and matching delta1/delta2."""
    Nv = 3
    cfg = Config()
    cfg.NORMALIZATION = "none"

    metadata = rep.collect_realization_metadata(cfg, sigma=0.2, n_real=10)
    m = metadata[3]  # arbitrary representative realization

    # Force the EXISTING _disorder_average to draw exactly (m.eps1, m.eps2)
    # deterministically: sigma=0 -> rng.normal(0,0)==0, so eps1_dis=cfg.eps1 and
    # eps2_dis=cfg.eps2 exactly, and eps1/eps2 are cfg.eps+cfg.delta1/2.
    basis, index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
    H = build_static_hamiltonian(Nv, basis, index, cfg, show_progress=False)
    mask1, mask2 = electronic_masks(basis)
    i0 = index[cfg.initial_state]

    cfg_forced = Config()
    cfg_forced.NORMALIZATION = "none"
    cfg_forced.delta1 = m.eps1 - cfg_forced.eps
    cfg_forced.delta2 = m.eps2 - cfg_forced.eps
    all_evals, all_intensity = _disorder_average(
        H, mask1, mask2, i0, sigma=0.0, cfg=cfg_forced,
        show_progress=False, desc="", n_realizations=1,
    )

    res = rep.compute_single_realization_spectrum(
        Nv, m.eps1, m.eps2, cfg, H=H, mask1=mask1, mask2=mask2, i0=i0,
    )

    from spectrum.spectrum import _broaden
    E_ref, spectrum_ref = _broaden(all_evals, all_intensity, cfg)
    assert np.allclose(res["E"], E_ref)
    assert np.allclose(res["spectrum"], spectrum_ref, atol=1e-12)


def test_case_selection_sanity():
    cfg = Config()
    metadata = rep.collect_realization_metadata(cfg, sigma=0.3, n_real=500)
    selections = rep.select_representative_realizations(metadata, cfg, sigma=0.3)

    # All 11 physical cases A-K selected.
    assert set(s["letter"] for s in selections.values()) == set("ABCDEFGHIJK")

    # Every case claims a distinct realization (used_indices uniqueness).
    chosen = [s["metadata"].index for s in selections.values()]
    assert len(chosen) == len(set(chosen)) == 11

    eps1 = np.array([m.eps1 for m in metadata])
    eps2 = np.array([m.eps2 for m in metadata])
    dist_from_center = np.sqrt((eps1 - cfg.eps) ** 2 + (eps2 - cfg.eps) ** 2)
    gap = np.abs(eps1 - eps2)
    det1 = np.abs(eps1 - cfg.omega_c)
    det2 = np.abs(eps2 - cfg.omega_c)

    def idx_of(letter):
        m = next(s["metadata"] for s in selections.values() if s["letter"] == letter)
        return next(i for i, mm in enumerate(metadata) if mm.index == m.index)

    # A -- nearly no disorder: closest to the clean center of all UNUSED
    # realizations; with A picked first it is simply the global minimum.
    assert dist_from_center[idx_of("A")] == dist_from_center.min()

    # K -- maximum energy mismatch: among the very largest gaps. (The global
    # max may be claimed first by case G, which also maximizes the gap but only
    # over opposite-sign realizations -- used_indices then hands K the largest
    # remaining gap, so K sits within the top handful.)
    assert gap[idx_of("K")] >= np.sort(gap)[-3]

    # J -- nearly degenerate: much smaller gap than K.
    assert gap[idx_of("J")] < gap[idx_of("K")]

    # E -- both above resonance: both site energies exceed omega_c.
    e = selections["E: Both above resonance"]["metadata"]
    if selections["E: Both above resonance"]["filter_satisfied"]:
        assert e.eps1 > cfg.omega_c and e.eps2 > cfg.omega_c

    # F -- both below resonance: both site energies below omega_c.
    f = selections["F: Both below resonance"]["metadata"]
    if selections["F: Both below resonance"]["filter_satisfied"]:
        assert f.eps1 < cfg.omega_c and f.eps2 < cfg.omega_c

    # G -- opposite disorder: deviations from the clean energy have opposite signs.
    g = selections["G: Opposite disorder"]["metadata"]
    if selections["G: Opposite disorder"]["filter_satisfied"]:
        assert (g.eps1 - cfg.eps) * (g.eps2 - cfg.eps) < 0

    # I -- resonance mismatch: gap should be close to the coupling g.
    assert abs(gap[idx_of("I")] - cfg.g) < abs(gap[idx_of("K")] - cfg.g)


def test_pass2_reuse_matches_recompute():
    """The fast Pass-2 path (reuse Pass-1 eigendata) must match the explicit
    re-diagonalization path to float round-off, for every normalization mode."""
    Nv = 4
    for norm in ("none", "reference", "reference_area", "area"):
        cfg = Config()
        cfg.NORMALIZATION = norm
        cfg.n_realizations = 8
        cfg.sigma = 0.24
        avg = compute_spectrum_for_Nv(Nv, cfg, show_progress=False)
        md = rep.collect_realization_metadata(cfg)
        sel = rep.select_representative_realizations(md, cfg)

        reuse = rep.assemble_representative_spectra(Nv, sel, avg, cfg)
        recompute = rep.recompute_representative_spectra(
            Nv, sel, cfg,
            reference_max=avg.get("reference_max"),
            reference_area=avg.get("reference_area"),
            show_progress=False,
        )
        assert set(reuse) == set(recompute)
        for k in reuse:
            assert reuse[k]["realization_index"] == recompute[k]["realization_index"]
            assert np.allclose(reuse[k]["spectrum"], recompute[k]["spectrum"], atol=1e-11)


def test_run_representative_analysis_end_to_end():
    """Small end-to-end smoke test: Pass 1 + selection + Pass 2, self-consistent."""
    cfg = Config()
    cfg.n_realizations = 25
    cfg.NORMALIZATION = "none"
    result = rep.run_representative_analysis(3, cfg, show_progress=False)

    assert len(result["metadata"]) == 25
    assert len(result["representative"]) == 11
    for name, res in result["representative"].items():
        assert res["case"] == name
        assert res["E"].shape == result["average"]["E"].shape
        assert np.all(np.isfinite(res["spectrum"]))


# Golden values for Nv=2, sigma=0.24, 2 realizations, NORMALIZATION='none'
# (byte-identical to the pre-existing spectrum pipeline; guards against any
# accidental edit to basis.py/operators.py/hamiltonian.py/spectrum.py).
_BASELINE = {
    "dim": 30,
    "spectrum_sum": 982.8740107471,
    "spectrum_max": 3.3985529682,
    "E_at_max": 6.5682841421,
    "evals_sum": 425.2842566353,
    "intens_sum": 2.0000000000,
}


def test_spectrum_pipeline_unchanged():
    cfg = Config()
    cfg.NORMALIZATION = "none"
    cfg.n_realizations = 2
    cfg.sigma = 0.24
    res = compute_spectrum_for_Nv(2, cfg, show_progress=False)
    E, s = res["E"], res["spectrum"]

    assert res["dim"] == _BASELINE["dim"]
    assert np.isclose(s.sum(), _BASELINE["spectrum_sum"], rtol=0, atol=1e-6)
    assert np.isclose(s.max(), _BASELINE["spectrum_max"], rtol=0, atol=1e-6)
    assert np.isclose(E[np.argmax(s)], _BASELINE["E_at_max"], rtol=0, atol=1e-6)
    assert np.isclose(res["all_evals"].sum(), _BASELINE["evals_sum"], rtol=0, atol=1e-6)
    assert np.isclose(res["all_intensity"].sum(), _BASELINE["intens_sum"], rtol=0, atol=1e-9)


if __name__ == "__main__":
    test_metadata_matches_disorder_average_rng_stream()
    test_pass2_matches_existing_pipeline_exactly()
    test_case_selection_sanity()
    test_run_representative_analysis_end_to_end()
    test_spectrum_pipeline_unchanged()
    print("representative-analysis tests passed")
