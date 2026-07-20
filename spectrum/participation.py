"""Single-molecule sector probabilities P(v), heatmap, and participation ratio.

Supplementary Material, Sec. II. For a polaritonic eigenstate expanded in the
two-molecule vibronic basis,

    |psi> = sum_{a,b,p} C_{a,b,p} |lambda_a^{(1)}, lambda_b^{(2)}, p> ,

the probability that (say) molecule 1 occupies vibronic sector v0 is

    P^{(1)}(v0) = sum_{a : v_a = v0} sum_{b,p} |C_{a,b,p}|^2 .          (*)

This is the distinguishable-molecule form of Eq. (S10). It is exact for any
eigenstate; at ``sigma = 0`` the bright eigenstates are symmetric under 1<->2
exchange and (*) collapses term-by-term onto the four contributions of Eq. (S10)
(the "both same state", "same sector different level", and the two 1/2-weighted
"different sector" terms). ``P^{(2)}(v0)`` is the analogous sum over molecule 2;
at ``sigma = 0`` the two are equal, and their average is the symmetric quantity
plotted in Fig. S1.

For every eigenstate, ``sum_v P(v) = 1`` (each basis state assigns each molecule
to exactly one sector), and the participation ratio is

    PR = 1 / sum_v P(v)^2 .
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import Config
from .polariton import PolaritonSector, PolaritonSolution


def sector_grid(sector: PolaritonSector) -> np.ndarray:
    """Contiguous integer grid of vibronic sectors present for either molecule."""
    vmin = int(min(sector.v1.min(), sector.v2.min()))
    vmax = int(max(sector.v1.max(), sector.v2.max()))
    return np.arange(vmin, vmax + 1)


def _grouped_weight(weights: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum ``weights`` (dim, n_eig) into ``(n_groups, n_eig)`` by ``group_idx`` rows."""
    out = np.zeros((n_groups, weights.shape[1]))
    np.add.at(out, group_idx, weights)
    return out


def compute_Pv(
    sol: PolaritonSolution,
    cfg: Config,
    grid: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(grid, Pv)`` with ``Pv`` of shape ``(len(grid), n_eig)``.

    ``Pv[g, n]`` is the single-molecule occupation of sector ``grid[g]`` in
    polaritonic eigenstate ``n``, per ``cfg.heatmap_which_molecule``
    (``"1"``, ``"2"``, or ``"avg"``). Columns sum to 1.
    """
    sector = sol.sector
    if grid is None:
        grid = sector_grid(sector)
    vmin = int(grid[0])
    n_groups = len(grid)

    weights = np.abs(sol.evecs) ** 2                      # (dim, n_eig)
    g1 = (sector.v1 - vmin).astype(int)                   # molecule-1 sector row
    g2 = (sector.v2 - vmin).astype(int)                   # molecule-2 sector row

    which = str(cfg.heatmap_which_molecule)
    if which == "1":
        Pv = _grouped_weight(weights, g1, n_groups)
    elif which == "2":
        Pv = _grouped_weight(weights, g2, n_groups)
    elif which == "avg":
        P1 = _grouped_weight(weights, g1, n_groups)
        P2 = _grouped_weight(weights, g2, n_groups)
        Pv = 0.5 * (P1 + P2)
    else:
        raise ValueError(
            f"cfg.heatmap_which_molecule={which!r} must be '1', '2', or 'avg'."
        )
    return grid, Pv


def participation_ratio(Pv: np.ndarray) -> np.ndarray:
    """PR = 1 / sum_v P(v)^2 for each eigenstate (column of ``Pv``)."""
    denom = np.sum(Pv ** 2, axis=0)
    pr = np.zeros_like(denom)
    nz = denom > 0
    pr[nz] = 1.0 / denom[nz]
    return pr


@dataclass
class HeatmapResult:
    """Discrete Fig. S1-style heatmap for a single realization."""

    grid: np.ndarray            # (n_v,) integer vibronic sectors (y-axis)
    bright_idx: np.ndarray      # (n_bright,) eigenstate indices, sorted by energy
    energies: np.ndarray        # (n_bright,) polariton energies of bright states
    intensity: np.ndarray       # (n_bright,) absorption intensities
    heatmap: np.ndarray         # (n_v, n_bright) = P(v) per bright state (x-axis)
    pr: np.ndarray              # (n_bright,) participation ratio of each bright state


def bright_heatmap(sol: PolaritonSolution, cfg: Config, grid: np.ndarray = None) -> HeatmapResult:
    """Build the Fig. S1 heatmap: P(v) of the bright polaritonic states, energy-ordered.

    "Bright" = absorption intensity above ``cfg.heatmap_bright_threshold`` times the
    brightest state's intensity. Columns are ordered by increasing energy, exactly
    as the discrete stems of Fig. 3(b) / the x-axis of Fig. S1.
    """
    grid, Pv = compute_Pv(sol, cfg, grid)
    pr_all = participation_ratio(Pv)

    imax = sol.intensity.max()
    thresh = cfg.heatmap_bright_threshold * imax if imax > 0 else 0.0
    bright = np.where(sol.intensity > thresh)[0]
    order = bright[np.argsort(sol.evals[bright])]

    return HeatmapResult(
        grid=grid,
        bright_idx=order,
        energies=sol.evals[order],
        intensity=sol.intensity[order],
        heatmap=Pv[:, order],
        pr=pr_all[order],
    )
