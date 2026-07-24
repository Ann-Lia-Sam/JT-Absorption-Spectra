"""Representative P(v) heatmap analysis: correctness checks.

The heatmap layer is validated against the *frozen* absorption pipeline:

* For any ``(eps1, eps2)`` (including disordered, eps1 != eps2), the vibronic ->
  polaritonic route must reproduce the primitive-basis eigenvalues/intensities
  that the absorption pipeline would produce -- i.e. the heatmap is built from
  the same physics, just in the vibronic basis.
* ``sum_v P(v) = 1`` for every polaritonic eigenstate.
* The standalone workflow selects the same Cases A-H and produces a valid
  discrete Fig. S1 heatmap per case.

Nothing here modifies the absorption-spectrum pipeline.
"""
import numpy as np

from spectrum.basis import build_sector_basis
from spectrum.config import Config
from spectrum.hamiltonian import (
    build_static_hamiltonian,
    electronic_diagonal,
    electronic_masks,
)
from spectrum import participation, polariton, vibronic
from spectrum import representative_heatmap as rh


def _primitive_evals_intensity(Nv, cfg, eps1, eps2):
    """Eigenvalues/intensities the frozen absorption pipeline gives for (eps1,eps2)."""
    basis, index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
    H = build_static_hamiltonian(Nv, basis, index, cfg, show_progress=False)
    mask1, mask2 = electronic_masks(basis)
    d = np.arange(H.shape[0])
    H[d, d] += electronic_diagonal(mask1, mask2, eps1, eps2)
    evals, evecs = np.linalg.eigh(H)
    intensity = np.abs(evecs[index[cfg.initial_state], :]) ** 2
    return len(basis), evals, intensity


def test_vibronic_route_matches_primitive_disordered():
    """The heatmap's vibronic route reproduces the frozen primitive physics for
    a DISORDERED realization (eps1 != eps2), not just sigma=0."""
    for Nv in (3, 4):
        cfg = Config()
        cfg.heatmap_nv = Nv
        eps1 = cfg.eps + 0.13   # arbitrary asymmetric disorder
        eps2 = cfg.eps - 0.21

        dim, evp, ip = _primitive_evals_intensity(Nv, cfg, eps1, eps2)
        sol = rh.solve_representative(cfg, eps1, eps2)

        assert sol.sector.dim == dim
        assert np.max(np.abs(np.sort(evp) - np.sort(sol.evals))) < 1e-9
        # intensities agree once both are ordered by energy
        op, ov = np.argsort(evp), np.argsort(sol.evals)
        assert np.max(np.abs(ip[op] - sol.intensity[ov])) < 1e-9


def test_Pv_columns_sum_to_one():
    cfg = Config()
    cfg.heatmap_nv = 4
    sol = rh.solve_representative(cfg, cfg.eps + 0.1, cfg.eps - 0.05)
    grid, Pv = participation.compute_Pv(sol, cfg)
    assert np.allclose(Pv.sum(axis=0), 1.0, atol=1e-9)


def test_bright_heatmap_columns_sum_to_one():
    """Each bright column of the Fig. S1 heatmap is a full P(v) -> sums to 1."""
    cfg = Config()
    cfg.heatmap_nv = 4
    sol = rh.solve_representative(cfg, cfg.eps, cfg.eps)
    hm = participation.bright_heatmap(sol, cfg)
    assert hm.heatmap.shape[0] == len(hm.grid)
    assert np.allclose(hm.heatmap.sum(axis=0), 1.0, atol=1e-9)
    # PR is finite and >= 1 for every bright state.
    assert np.all(np.isfinite(hm.pr))
    assert np.all(hm.pr >= 1.0 - 1e-9)


def test_end_to_end_representative_heatmaps():
    """Selection + per-case heatmap, self-consistent, at a small Nv."""
    cfg = Config()
    cfg.heatmap_nv = 3
    cfg.n_realizations = 40
    cfg.sigma = 0.24
    result = rh.run_representative_heatmaps(3, cfg, show_progress=False)

    # 8 physical cases A-H, each with a valid heatmap.
    assert set(e["letter"] for e in result["heatmaps"].values()) == set("ABCDEFGH")
    for name, entry in result["heatmaps"].items():
        assert entry["case"] == name
        hm = entry["heatmap"]
        assert hm.shape[0] == len(entry["grid"])
        assert hm.shape[1] == len(entry["energies"]) == len(entry["pr"])
        assert np.all(np.isfinite(hm))
        if hm.shape[1] > 0:
            assert np.allclose(hm.sum(axis=0), 1.0, atol=1e-9)


def test_solve_matches_selected_representative_index():
    """The heatmap of a selected case uses exactly that realization's (eps1,eps2),
    and its evals match the frozen primitive route for that same pair."""
    cfg = Config()
    cfg.heatmap_nv = 3
    cfg.n_realizations = 40
    cfg.sigma = 0.24
    result = rh.run_representative_heatmaps(3, cfg, show_progress=False)

    entry = result["heatmaps"]["A: Resonant baseline"]
    _, evp, _ = _primitive_evals_intensity(3, cfg, entry["eps1"], entry["eps2"])
    sol = rh.solve_representative(cfg, entry["eps1"], entry["eps2"])
    assert np.max(np.abs(np.sort(evp) - np.sort(sol.evals))) < 1e-9


if __name__ == "__main__":
    test_vibronic_route_matches_primitive_disordered()
    test_Pv_columns_sum_to_one()
    test_bright_heatmap_columns_sum_to_one()
    test_end_to_end_representative_heatmaps()
    test_solve_matches_selected_representative_index()
    print("representative-heatmap tests passed")
