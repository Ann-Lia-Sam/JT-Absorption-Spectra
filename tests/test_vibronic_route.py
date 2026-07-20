"""The vibronic route must reproduce the primitive-basis physics exactly.

The absorption-spectrum pipeline builds the Hamiltonian directly in the primitive
product basis restricted to the (nex=1, j=-1) sector. The vibronic route
(SM Sec. I.A) is a per-molecule change of basis of that *same* Hamiltonian, so at
sigma=0 (identical molecules) the two must yield:

* the same sector dimension,
* identical eigenvalues,
* identical absorption intensities,

and the paper's alpha/beta-built vibronic Hamiltonian must equal the exact
change-of-basis W^T H_prim W of the primitive Hamiltonian. We also check that
sum_v P(v) = 1 for every polaritonic eigenstate.
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


def _primitive_route(Nv, cfg):
    basis, index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
    H = build_static_hamiltonian(Nv, basis, index, cfg, show_progress=False)
    mask1, mask2 = electronic_masks(basis)
    H[np.arange(H.shape[0]), np.arange(H.shape[0])] += electronic_diagonal(
        mask1, mask2, cfg.eps1, cfg.eps2
    )
    evals, evecs = np.linalg.eigh(H)
    intensity = np.abs(evecs[index[cfg.initial_state], :]) ** 2
    return basis, index, H, evals, intensity


def _W(vib1, vib2, sector, prim_basis, prim_index):
    """Change-of-basis: columns are vibronic sector states in primitive coords."""
    W = np.zeros((len(prim_basis), sector.dim))
    for col, (a, b, p) in enumerate(sector.states):
        cp, cm = (1, 0) if p == 1 else ((0, 1) if p == -1 else (0, 0))
        Ua, Ub = vib1.U[:, a], vib2.U[:, b]
        for pa, ca in zip(vib1.basis, Ua):
            if ca == 0.0:
                continue
            for pb, cb in zip(vib2.basis, Ub):
                if cb == 0.0:
                    continue
                key = (pa[0], pa[1], pa[2], pb[0], pb[1], pb[2], cp, cm)
                r = prim_index.get(key)
                if r is not None:
                    W[r, col] += ca * cb
    return W


def test_vibronic_matches_primitive():
    for Nv in (3, 4):
        cfg = Config()
        cfg.heatmap_nv = Nv
        cfg.delta1 = cfg.delta2 = 0.0

        pb, pidx, Hprim, evp, ip = _primitive_route(Nv, cfg)

        vib1 = vibronic.diagonalize_single_molecule(cfg.eps1, cfg)
        vib2 = vibronic.diagonalize_single_molecule(cfg.eps2, cfg)
        sol = polariton.solve(vib1, vib2, cfg)

        assert sol.sector.dim == len(pb)
        assert np.max(np.abs(np.sort(evp) - np.sort(sol.evals))) < 1e-9
        op, ov = np.argsort(evp), np.argsort(sol.evals)
        assert np.max(np.abs(ip[op] - sol.intensity[ov])) < 1e-9

        grid, Pv = participation.compute_Pv(sol, cfg)
        assert np.allclose(Pv.sum(axis=0), 1.0, atol=1e-9)

        Hvib = polariton.build_hamiltonian(vib1, vib2, sol.sector, cfg)
        W = _W(vib1, vib2, sol.sector, pb, pidx)
        assert np.max(np.abs(W.T @ W - np.eye(sol.sector.dim))) < 1e-9
        assert np.max(np.abs(W.T @ Hprim @ W - Hvib)) < 1e-9


def test_vibronic_matches_primitive_with_disorder():
    """With delta1 != delta2 the two molecules differ, but the vibronic route is still
    an exact change of basis of the primitive Hamiltonian."""
    Nv = 3
    for d1, d2 in [(0.10, -0.05), (0.30, 0.20)]:
        cfg = Config()
        cfg.heatmap_nv = Nv
        eps1, eps2 = cfg.eps + d1, cfg.eps + d2

        basis, index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
        H = build_static_hamiltonian(Nv, basis, index, cfg, show_progress=False)
        m1, m2 = electronic_masks(basis)
        d = np.arange(H.shape[0])
        H[d, d] += electronic_diagonal(m1, m2, eps1, eps2)
        evp, evecp = np.linalg.eigh(H)
        ip = np.abs(evecp[index[cfg.initial_state], :]) ** 2

        vib1 = vibronic.diagonalize_single_molecule(eps1, cfg)
        vib2 = vibronic.diagonalize_single_molecule(eps2, cfg)
        sol = polariton.solve(vib1, vib2, cfg)

        assert np.max(np.abs(np.sort(evp) - np.sort(sol.evals))) < 1e-9
        assert np.max(np.abs(ip[np.argsort(evp)] - sol.intensity[np.argsort(sol.evals)])) < 1e-9
        _, Pv = participation.compute_Pv(sol, cfg)
        assert np.allclose(Pv.sum(axis=0), 1.0, atol=1e-9)

        # Disorder breaks 1<->2 symmetry, so the two molecules' P(v) genuinely differ.
        cfg.heatmap_which_molecule = "1"
        _, P1 = participation.compute_Pv(sol, cfg)
        cfg.heatmap_which_molecule = "2"
        _, P2 = participation.compute_Pv(sol, cfg)
        bright = sol.intensity > 1e-6 * sol.intensity.max()
        assert np.max(np.abs(P1[:, bright] - P2[:, bright])) > 1e-3


def test_shift_reference_matches_direct_diagonalization():
    """shift_reference(reference, eps_k) must give the same eigenvalues/labels as
    diagonalizing directly at eps_k (the O(1) shift is algebraically exact)."""
    cfg = Config()
    cfg.heatmap_nv = 4
    reference = vibronic.diagonalize_single_molecule_reference(cfg)
    for eps_k in (cfg.eps, cfg.eps + 0.15, cfg.eps - 0.20):
        shifted = vibronic.shift_reference(reference, eps_k)
        direct = vibronic.diagonalize_single_molecule(eps_k, cfg)
        assert np.array_equal(shifted.v, direct.v)
        assert np.array_equal(shifted.is_excited, direct.is_excited)
        assert np.max(np.abs(np.sort(shifted.evals) - np.sort(direct.evals))) < 1e-9


def test_optimized_route_matches_primitive():
    """heatmap.solve_realization (reference + O(1) shift) must match the primitive
    route exactly, at sigma=0 and with disorder -- same guarantee as the from-scratch
    vibronic route, now exercising the cached/optimized code path."""
    from spectrum import heatmap as hm

    Nv = 3
    for d1, d2 in [(0.0, 0.0), (0.10, -0.05), (0.30, 0.20)]:
        cfg = Config()
        cfg.heatmap_nv = Nv
        eps1, eps2 = cfg.eps + d1, cfg.eps + d2

        basis, index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
        H = build_static_hamiltonian(Nv, basis, index, cfg, show_progress=False)
        m1, m2 = electronic_masks(basis)
        d = np.arange(H.shape[0])
        H[d, d] += electronic_diagonal(m1, m2, eps1, eps2)
        evp, evecp = np.linalg.eigh(H)
        ip = np.abs(evecp[index[cfg.initial_state], :]) ** 2

        sol = hm.solve_realization(cfg, eps1, eps2)  # reference + shift internally

        assert np.max(np.abs(np.sort(evp) - np.sort(sol.evals))) < 1e-9
        assert np.max(np.abs(ip[np.argsort(evp)] - sol.intensity[np.argsort(sol.evals)])) < 1e-9


def test_disorder_average_heatmap_sigma0_shortcut():
    """At sigma=0, disorder_average_heatmap must skip the loop but give the same
    result as it would if it looped n_real identical realizations."""
    from spectrum import heatmap as hm

    cfg = Config()
    cfg.heatmap_nv = 4
    cfg.heatmap_sigma = 0.0
    cfg.heatmap_E_points = 50

    dh1 = hm.disorder_average_heatmap(cfg, n_real=1, show_progress=False)
    dh5 = hm.disorder_average_heatmap(cfg, n_real=5, show_progress=False)

    assert np.allclose(dh1.M, dh5.M)
    assert np.allclose(dh1.spectrum, dh5.spectrum)
    assert np.allclose(dh1.PR, dh5.PR)
    assert dh5.n_realizations == 5  # metadata reflects the requested count

    # Cross-check against explicitly looping n_real identical draws the slow way.
    reference = vibronic.diagonalize_single_molecule_reference(cfg)
    E = hm.energy_grid(cfg)
    grid = hm.sector_grid(polariton.build_sector(
        vibronic.shift_reference(reference, cfg.eps),
        vibronic.shift_reference(reference, cfg.eps), cfg,
    ))
    M_manual = np.zeros((len(grid), len(E)))
    for _ in range(5):
        sol = hm.solve_realization(cfg, cfg.eps1, cfg.eps2, reference=reference)
        M_manual += hm._energy_map(sol, cfg, E, grid)
    M_manual /= 5
    assert np.allclose(dh5.M, M_manual)


def test_Pv_reduces_to_S10_symmetric_at_sigma0():
    """At sigma=0 the bright states are 1<->2 symmetric, so P^(1)=P^(2) (the S10 value)."""
    cfg = Config()
    cfg.heatmap_nv = 4
    cfg.heatmap_which_molecule = "1"
    vib = vibronic.diagonalize_single_molecule(cfg.eps, cfg)
    sol = polariton.solve(vib, vib, cfg)
    _, P1 = participation.compute_Pv(sol, cfg)
    cfg.heatmap_which_molecule = "2"
    _, P2 = participation.compute_Pv(sol, cfg)
    bright = sol.intensity > 1e-6 * sol.intensity.max()
    assert np.allclose(P1[:, bright], P2[:, bright], atol=1e-8)


if __name__ == "__main__":
    test_vibronic_matches_primitive()
    test_vibronic_matches_primitive_with_disorder()
    test_shift_reference_matches_direct_diagonalization()
    test_optimized_route_matches_primitive()
    test_disorder_average_heatmap_sigma0_shortcut()
    test_Pv_reduces_to_S10_symmetric_at_sigma0()
    print("vibronic-route tests passed")
