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
    if I_ref.max() > 0:
        I_ref = I_ref / I_ref.max()
    ax.plot(
        E_ref, I_ref,
        color="black", linewidth=1, linestyle="--",
        label="Without disorder (ref)", zorder=10,
    )


def _finish(fig, ax, cfg: Config, out_path: str, show: bool) -> str:
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity")
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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Absorption vs disorder strength (Nv={Nv}, κ/ω = 2.2)")

    for res in sorted(results, key=lambda r: r["sigma"]):
        ax.plot(
            res["E"], res["spectrum"], linewidth=1.5, label=f"σ={res['sigma']:g} eV"
        )

    _plot_reference(ax, cfg)  # drawn last -> sits on top
    return _finish(fig, ax, cfg, out_path, show)



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
