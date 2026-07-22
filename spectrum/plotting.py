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
# Representative disorder realizations (analysis; see spectrum/representative.py)
# ---------------------------------------------------------------------------
def plot_disorder_scatter(
    metadata: List,
    representative: Dict[str, Dict],
    cfg: Config,
    Nv: int,
    out_path: str,
    show: bool = True,
) -> str:
    """Scatter (eps1, eps2) for every sampled realization; highlight the cases."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps1 = np.array([m.eps1 for m in metadata])
    eps2 = np.array([m.eps2 for m in metadata])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(eps1, eps2, s=10, alpha=0.35, color="steelblue",
               label=f"{len(metadata)} realizations")
    ax.plot(cfg.eps, cfg.eps, marker="+", color="black", markersize=14,
            markeredgewidth=2, linestyle="none", label="clean (eps, eps)", zorder=4)
    # Cavity-resonance reference lines (cases B-F are defined relative to omega_c).
    ax.axvline(cfg.omega_c, color="gray", linestyle=":", linewidth=1, zorder=1)
    ax.axhline(cfg.omega_c, color="gray", linestyle=":", linewidth=1,
               label=r"$\omega_c$ (cavity resonance)", zorder=1)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (name, res) in enumerate(representative.items()):
        letter = name.split(":")[0].strip()
        ax.scatter(res["eps1"], res["eps2"], s=110, color=colors[i % 10],
                   edgecolor="black", linewidth=0.8, zorder=5)
        ax.annotate(letter, (res["eps1"], res["eps2"]), textcoords="offset points",
                    xytext=(7, 7), fontsize=12, fontweight="bold")

    ax.set_xlabel(r"$\epsilon_1$ (eV)")
    ax.set_ylabel(r"$\epsilon_2$ (eV)")
    ax.set_title(f"Disorder realizations (Nv={Nv}, σ={cfg.sigma:g} eV)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_representative_spectrum(
    rep_result: Dict,
    avg_result: Dict,
    cfg: Config,
    out_path: str,
    show: bool = True,
) -> str:
    """One representative (Pass-2) spectrum, overlaid with the disorder average
    and the 'without disorder' reference curve (``cfg.reference_file``)."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(avg_result["E"], avg_result["spectrum"], color="tab:blue", linewidth=1.5,
            label=f"disorder average ({cfg.n_realizations} real.)")
    ax.plot(rep_result["E"], rep_result["spectrum"], color="tab:red", linewidth=1.5,
            label=f"realization #{rep_result['realization_index']}")
    # Reference curve (spectrum_ref_2mol_2.2.pl), same styling/normalization as
    # every other overlay in this module -- dashed black, drawn on top.
    _plot_reference(ax, cfg)

    ax.set_title(
        f"{rep_result['case']}\n"
        f"realization #{rep_result['realization_index']}: "
        rf"$\epsilon_1$={rep_result['eps1']:.3f} eV, "
        rf"$\epsilon_2$={rep_result['eps2']:.3f} eV "
        f"(σ={cfg.sigma:g} eV)"
    )
    return _finish(fig, ax, cfg, out_path, show)
