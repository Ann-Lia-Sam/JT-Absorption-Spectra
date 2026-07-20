"""Overlay computed spectra with an optional reference curve drawn on top."""

import os
from typing import Dict, List

import matplotlib
import numpy as np

from .config import Config


def _plot_reference(ax, cfg: Config) -> None:
    """Draw the 'without disorder' reference curve on top of everything.

    Uses columns 0 (energy) and 1 (intensity) of ``cfg.reference_file``. A high
    ``zorder`` and a bold black line keep it visually above the computed spectra.
    """
    ref = cfg.reference_file
    if not ref or not os.path.isfile(ref):
        return
    data = np.loadtxt(ref, comments="#")
    E_ref, I_ref = data[:, 0], data[:, 1]
    if cfg.NORMALIZATION == "reference":
        if I_ref.max() > 0:
            I_ref = I_ref / I_ref.max()
    elif cfg.NORMALIZATION in ("area", "reference_area"):
        area = np.trapezoid(I_ref, E_ref)
        if area > 0:
            I_ref = I_ref / area
    elif cfg.NORMALIZATION == "none":
        pass
    ax.plot(
        E_ref, I_ref,
        color="black", linewidth=1, linestyle="--",
        label="Without disorder (ref)", zorder=10,
    )


def _finish(fig, ax, cfg: Config, out_path: str, show: bool, ylabel: str = "Intensity") -> str:
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(cfg.E_min, cfg.E_max)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_path


def overlay(
    results: List[Dict],
    cfg: Config,
    out_path: str,
    show: bool = True,
) -> str:
    """Plot each Nv's spectrum on one axis; reference on top. Returns the path."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Disorder-averaged absorption spectra (κ/ω = 2.2)")

    for res in sorted(results, key=lambda r: r["Nv"]):
        ax.plot(res["E"], res["spectrum"], linewidth=1.5, label=f"Nv={res['Nv']}")

    _plot_reference(ax, cfg)  # drawn last -> sits on top
    return _finish(fig, ax, cfg, out_path, show)


def overlay_sigma(
    results: List[Dict],
    cfg: Config,
    out_path: str,
    show: bool = True,
) -> str:
    """Plot spectra for a single Nv across sigma; reference on top. Returns path."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Nv = results[0]["Nv"] if results else "?"

    # Every result's ``spectrum`` is already normalized (in spectrum.py) by
    # the same σ=0 reference peak, so they can be plotted as-is: the σ=0
    # curve peaks at 1 and disordered curves peak below it -- making the
    # disorder-driven peak reduction visible.
    have_ref = bool(results) and all(r.get("reference_max") is not None for r in results)
    ref_label = "σ=0 eV" if have_ref else None

    fig, ax = plt.subplots(figsize=(8, 5))
    title = f"Absorption vs disorder strength (Nv={Nv}, κ/ω = 2.2)"
    if have_ref:
        title += f"  [normalized to {ref_label}]"
    ax.set_title(title)

    for res in sorted(results, key=lambda r: r["sigma"]):
        ax.plot(res["E"], res["spectrum"], linewidth=1.5, label=f"σ={res['sigma']:g} eV")

    _plot_reference(ax, cfg)  # drawn last -> sits on top
    ylabel = f"Intensity (rel. to {ref_label})" if have_ref else "Intensity"
    return _finish(fig, ax, cfg, out_path, show, ylabel=ylabel)



def overlay_realizations(
    results: List[Dict],
    cfg: Config,
    out_path: str,
    show: bool = True,
) -> str:
    """Plot spectra for a single Nv across realization counts; reference on top."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Nv = results[0]["Nv"] if results else "?"
    sigma = results[0].get("sigma", cfg.sigma) if results else cfg.sigma
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        f"Absorption vs #realizations (Nv={Nv}, σ={sigma:g} eV, κ/ω = 2.2, 100real)"
    )

    for res in sorted(results, key=lambda r: r["n_realizations"]):
        ax.plot(
            res["E"], res["spectrum"], linewidth=1.5,
            label=f"{res['n_realizations']} realizations",
        )

    _plot_reference(ax, cfg)  # drawn last -> sits on top
    return _finish(fig, ax, cfg, out_path, show)


# ---------------------------------------------------------------------------
# Heatmap plots (vibronic / participation-ratio workflow)
# ---------------------------------------------------------------------------
def _crop_v_range(grid, weight, pad: int = 1):
    """Return (vlo, vhi) covering sectors that carry non-negligible weight."""
    col = np.asarray(weight)
    if col.ndim == 2:
        col = col.max(axis=1)
    signif = np.where(col > 1e-4 * (col.max() if col.max() > 0 else 1.0))[0]
    if signif.size == 0:
        return grid[0], grid[-1]
    lo = max(0, signif[0] - pad)
    hi = min(len(grid) - 1, signif[-1] + pad)
    return grid[lo], grid[hi]


def plot_figS1(hm, cfg: Config, out_path: str, show: bool = True) -> str:
    """Reproduce Fig. S1: single-molecule occupation P(v) of the bright polariton states.

    ``hm`` is a :class:`spectrum.participation.HeatmapResult`. y-axis = vibronic
    sector v, x-axis = bright polaritonic states ordered by increasing energy,
    color = P(v).
    """
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bright = hm.heatmap.shape[1]
    x = np.arange(n_bright)
    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(
        x, hm.grid, hm.heatmap, cmap="viridis", shading="nearest",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$P_v$")
    ax.set_xlabel("Bright polaritonic states")
    ax.set_ylabel("Sector $v$")
    ax.set_title(
        f"Single-molecule occupation of vibronic sectors (N=2, Nv={cfg.heatmap_nv}, "
        f"σ={cfg.heatmap_sigma:g})"
    )
    vlo, vhi = _crop_v_range(hm.grid, hm.heatmap)
    ax.set_ylim(vlo - 0.5, vhi + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_disorder_heatmap(dh, cfg: Config, out_path: str, show: bool = True) -> str:
    """Energy-resolved, disorder-averaged sector-population heatmap.

    ``dh`` is a :class:`spectrum.heatmap.DisorderHeatmap`. Top panel: the
    polariton absorption spectrum (= sum_v of the map) with the per-energy
    participation ratio overlaid. Bottom panel: P(v|E) heatmap.
    """
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_top, ax) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05},
    )

    # Top: spectrum + PR(E)
    ax_top.plot(dh.E, dh.spectrum, color="black", lw=1.2, label="Polariton absorption")
    ax_top.set_ylabel("Intensity")
    ax_pr = ax_top.twinx()
    ax_pr.plot(dh.E, dh.PR, color="tab:red", lw=1.0, alpha=0.8, label="PR")
    ax_pr.set_ylabel("PR", color="tab:red")
    ax_pr.tick_params(axis="y", labelcolor="tab:red")
    ax_top.set_title(
        f"Sector-resolved polariton spectrum (N=2, Nv={dh.Nv}, σ={dh.sigma:g} eV, "
        f"{dh.n_realizations} real.)"
    )

    # Bottom: P(v|E) heatmap
    mesh = ax.pcolormesh(dh.E, dh.grid, dh.Pv_norm, cmap="viridis", shading="nearest")
    cbar = fig.colorbar(mesh, ax=[ax_top, ax], location="right", fraction=0.05, pad=0.02)
    cbar.set_label(r"$P(v\,|\,E)$")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Sector $v$")
    ax.set_xlim(dh.E[0], dh.E[-1])
    vlo, vhi = _crop_v_range(dh.grid, dh.M)
    ax.set_ylim(vlo - 0.5, vhi + 0.5)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path
