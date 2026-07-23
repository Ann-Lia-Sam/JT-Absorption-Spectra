#!/usr/bin/env python3
"""Overlay a reference curve onto selected representative cases  --  an
*additive, standalone* post-processing tool.

It does NOT modify or re-run any part of the existing pipeline. It simply reads
the per-case spectra that ``python main.py --representative`` already wrote to
``results/`` (files ``representative_spectrum_Nv{Nv}_sigma{sigma}_case{X}.npz``,
each holding ``E`` and ``spectrum``), loads an overlay curve -- a 2-column
``.csv`` (``Energy,Intensity`` with a header; default ``coordinates.csv``) or a
``.pl`` cross-section -- and writes one new overlay PNG per *chosen* case. The
overlay curve is normalized the *same way as the plotted case spectra* -- using
the mode each case ``.npz`` was written with (``reference``/``peak`` -> divide by
own peak; ``area``/``reference_area`` -> divide by own area; ``none`` -> leave
unchanged), so it always shares their vertical scale. Its raw (E, I) coordinates
are otherwise plotted exactly as given. Nothing in ``spectrum/`` is imported for
modification -- only :class:`Config` and the ``_sigma_tag`` filename helper are
reused read-only.

Typical workflow
----------------
1. Produce all the representative plots as usual::

       python main.py --representative --nv 12 --repr-sigma 0.5

2. Then choose which cases get the overlay (default curve: coordinates.csv)::

       # explicit
       python overlay_representative_1mol.py --nv 12 --sigma 0.5 --cases C,G
       # a different overlay file / label
       python overlay_representative_1mol.py --nv 12 --sigma 0.5 --cases C,G \\
              --overlay coordinates.csv --label "1-mol (matched)"
       # or interactive (lists the available cases and asks)
       python overlay_representative_1mol.py --nv 12 --sigma 0.5

Physics note (cases C and G)
----------------------------
Cases C (large mismatch) and G (single-molecule resonant) have one molecule near
resonance and the other detuned out, so their spectra approach a *single*-molecule
JT-polariton spectrum. But the per-molecule coupling inside the N=2 system is
``g = Omega/(2*sqrt(2)) = 0.2422 eV`` -- i.e. the standalone single-molecule
coupling ``Omega/2 = 0.3425 eV`` divided by sqrt(2). For a physically matched
overlay the single-molecule reference must be computed at that *reduced* coupling
(``Omega/(2*omega_c) = 0.05/sqrt(2) ~= 0.03536``), which puts its LP ~0.24 eV below
omega_c. A reference computed at the full standalone coupling instead sits ~0.34 eV
below omega_c and its Rabi splitting is ~sqrt(2) too wide for this comparison. The
tool measures the reference's splitting and warns which case it looks like.
"""

import argparse
import glob
import os
import sys

import numpy as np

OMEGA_C = 6.85  # eV, cavity resonance (cfg.omega_c); used only for the coupling note.


def _load_reference_curve(path):
    """Load a 2-column overlay curve from a ``.pl`` (whitespace, ``#`` comments)
    or ``.csv`` (comma-delimited, one header row) file.

    Returns the **raw** ``(E, I)`` exactly as given -- no shifting, rebinning,
    interpolation, or scaling. Vertical normalization is applied separately by
    :func:`_normalize_curve`, using the same mode the plotted case spectra were
    computed with (read from their ``.npz``), so the overlay shares their scale.
    """
    with open(path) as fh:
        first_line = fh.readline()
    is_csv = os.path.splitext(path)[1].lower() == ".csv" or "," in first_line
    if is_csv:
        # Comma-delimited. genfromtxt renders any header / non-numeric row as
        # NaN; we drop those, so no skiprows count is hardcoded and no genuine
        # data point is dropped or altered.
        data = np.genfromtxt(path, delimiter=",", comments="#")
        data = data[~np.isnan(data[:, 0]) & ~np.isnan(data[:, 1])]
    else:
        data = np.loadtxt(path, comments="#")
    E, I = data[:, 0], data[:, 1]
    return E, I


def _normalize_curve(E, I, mode="reference"):
    """Scale a curve the way ``spectrum.plotting._plot_reference`` scales the
    reference curve under ``NORMALIZATION == mode`` -- so an overlaid curve ends
    up on the same vertical scale as the plotted (already-normalized) spectra:

    * ``"reference"`` / ``"peak"``      -- divide by the curve's own maximum (peaks at 1)
    * ``"area"`` / ``"reference_area"`` -- divide by the curve's own trapezoidal area (∫=1)
    * ``"none"`` (or any unknown mode)  -- leave unchanged

    A non-positive max/area leaves the curve unchanged (nothing to divide by).
    """
    I = np.asarray(I, dtype=float)
    if mode in ("reference", "peak"):
        m = float(I.max())
        return I / m if m > 0 else I
    if mode in ("area", "reference_area"):
        area = float(np.trapezoid(I, E))
        return I / area if area > 0 else I
    return I


def _ylabel_for_mode(mode):
    """Y-axis label describing how curves in this figure were normalized."""
    if mode in ("reference", "peak"):
        return "Intensity (peak-normalized)"
    if mode in ("area", "reference_area"):
        return "Intensity (area-normalized)"
    return "Intensity"


def _lp_up(E, I, wc=OMEGA_C):
    """Brightest peak below (LP) and above (UP) ``wc``; either may be None."""
    lo, hi = E < wc, E >= wc
    lp = float(E[lo][np.argmax(I[lo])]) if lo.any() and I[lo].size else None
    up = float(E[hi][np.argmax(I[hi])]) if hi.any() and I[hi].size else None
    return lp, up


def _coupling_note(E1, I1):
    """Human-readable diagnosis of the reference's coupling from its splitting."""
    lp, up = _lp_up(E1, I1)
    lines = [f"[overlay] 1-mol reference LP={lp if lp is None else round(lp,3)} eV, "
             f"UP={up if up is None else round(up,3)} eV"]
    if lp is not None and up is not None:
        half = 0.5 * (up - lp)
        lines.append(f"          half-splitting = {half:.3f} eV  "
                     f"(standalone Omega/2 ~ 0.34 ; sqrt2-reduced g ~ 0.24)")
        if half > 0.30:
            lines.append("          => looks like the STANDALONE coupling. For a matched "
                         "comparison to cases C/G")
            lines.append("             (per-molecule g = Omega/(2*sqrt2)), regenerate the "
                         "1-mol run at Omega/(2*omega_c)=0.05/sqrt2 ~ 0.03536.")
        elif half < 0.28:
            lines.append("          => consistent with the sqrt2-reduced per-molecule "
                         "coupling (matched to cases C/G).")
    return "\n".join(lines)


def _discover_cases(results_dir, Nv, sigma_tag):
    """{'A': path, ...} for every case .npz matching this Nv/sigma."""
    pat = os.path.join(
        results_dir, f"representative_spectrum_Nv{Nv}_{sigma_tag}_case*.npz"
    )
    found = {}
    for p in sorted(glob.glob(pat)):
        letter = os.path.basename(p).split("_case")[-1].split(".npz")[0]
        found[letter] = p
    return found


def _resolve_sigma_tag(args, sigma_tag_fn, repo_root):
    """Return (sigma_tag, results_dir). Auto-detects sigma if not given and unique."""
    results_dir = args.results_dir
    if args.sigma is not None:
        return sigma_tag_fn(args.sigma), results_dir
    pat = os.path.join(results_dir, f"representative_spectrum_Nv{args.nv}_sigma*_case*.npz")
    tags = sorted({os.path.basename(p).split("_sigma")[1].split("_case")[0]
                   for p in glob.glob(pat)})
    if len(tags) == 1:
        print(f"[overlay] auto-detected sigma tag: sigma{tags[0]}")
        return f"sigma{tags[0]}", results_dir
    if not tags:
        print(f"No representative .npz found for Nv={args.nv} in '{results_dir}/'.")
        print(f"Run first:  python main.py --representative --nv {args.nv} --repr-sigma <SIGMA>")
        sys.exit(1)
    print(f"Multiple sigma runs found for Nv={args.nv}: {tags}")
    print("Re-run this tool with --sigma <one of these>.")
    sys.exit(1)


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, repo_root)
    from spectrum.config import Config          # read-only import
    from spectrum.storage import _sigma_tag      # read-only import (filename convention)

    cfg = Config()

    ap = argparse.ArgumentParser(
        description="Overlay a reference curve (.pl or .csv) on chosen "
                    "representative-case spectra (additive; reads results/*.npz).")
    ap.add_argument("--nv", type=int, default=cfg.representative_nv,
                    help=f"Nv of the representative run (default {cfg.representative_nv})")
    ap.add_argument("--sigma", type=float, default=None,
                    help="sigma of the run to load; auto-detected if a single run exists")
    ap.add_argument("--cases", type=str, default=None,
                    help="comma-separated case letters (e.g. C,G) or 'all'; "
                         "omit for an interactive prompt")
    ap.add_argument("--overlay", "--ref-1mol", dest="ref_1mol", type=str,
                    default="coordinates.csv",
                    help="curve to overlay: a 2-column .csv (Energy,Intensity) or a "
                         ".pl cross-section (default: coordinates.csv)")
    ap.add_argument("--label", type=str, default="1_molecule_with_g/√2",
                    help="legend label for the overlay curve "
                         "(default: '1_molecule_with_g/√2')")
    ap.add_argument("--results-dir", type=str, default=cfg.results_dir)
    ap.add_argument("--no-2mol-ref", action="store_true",
                    help="do not also draw the clean 2-mol reference (cfg.reference_file)")
    ap.add_argument("--no-show", action="store_true", help="save only, do not display")
    ap.add_argument("--out-suffix", type=str, default="withoverlay")
    ap.add_argument("--check-ref", action="store_true",
                    help="only report the overlay curve's peak/splitting diagnosis, then exit")
    args = ap.parse_args()

    overlay_label = args.label

    # ---- locate the overlay curve -------------------------------------------
    ref_path = args.ref_1mol
    if not os.path.isfile(ref_path):
        alt = os.path.join(repo_root, args.ref_1mol)
        if os.path.isfile(alt):
            ref_path = alt
        else:
            print(f"Overlay curve not found: '{args.ref_1mol}'")
            sys.exit(1)
    if os.path.getsize(ref_path) == 0:
        print(f"Overlay curve is empty (0 bytes): '{ref_path}'")
        sys.exit(1)
    E1, I1 = _load_reference_curve(ref_path)  # raw; normalized per-case below
    print(f"[overlay] overlay curve: {ref_path}  "
          f"({len(E1)} points, peak at {float(E1[np.argmax(I1)]):.4f} eV; "
          f"normalized to match each plotted case's mode)")
    if args.check_ref:
        # Detailed peak/coupling diagnosis is opt-in (meaningful only if the
        # curve is a single-molecule JT-polariton spectrum).
        print(_coupling_note(E1, I1))
        return

    # ---- discover the per-case spectra written by --representative ----------
    sigma_tag, results_dir = _resolve_sigma_tag(args, _sigma_tag, repo_root)
    available = _discover_cases(results_dir, args.nv, sigma_tag)
    if not available:
        print(f"No case .npz found for Nv={args.nv} {sigma_tag} in '{results_dir}/'.")
        print(f"Run first:  python main.py --representative --nv {args.nv} "
              f"--repr-sigma {args.sigma if args.sigma is not None else '<SIGMA>'}")
        sys.exit(1)

    # ---- choose cases (flag or interactive) ---------------------------------
    if args.cases:
        sel = args.cases.strip()
        chosen = list(available) if sel.lower() == "all" else \
            [c.strip().upper() for c in sel.split(",") if c.strip()]
    else:
        print("\nAvailable cases:")
        for L, p in available.items():
            try:
                name = str(np.load(p, allow_pickle=True)["case"])
            except Exception:
                name = "(case)"
            print(f"  {L} : {name}")
        raw = input("\nEnter case letters to overlay (comma-separated, or 'all'): ").strip()
        chosen = list(available) if raw.lower() == "all" else \
            [c.strip().upper() for c in raw.split(",") if c.strip()]
    if not chosen:
        print("No cases chosen; nothing to do.")
        return

    # ---- clean 2-mol reference (optional context curve) ---------------------
    ref2 = None
    if not args.no_2mol_ref and cfg.reference_file and os.path.isfile(cfg.reference_file):
        ref2 = _load_reference_curve(cfg.reference_file)

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    for L in chosen:
        if L not in available:
            print(f"  [skip] case {L}: no file for Nv={args.nv} {sigma_tag}")
            continue
        d = np.load(available[L], allow_pickle=True)
        E, spec = d["E"], d["spectrum"]
        name = str(d["case"])
        eps1, eps2 = float(d["eps1"]), float(d["eps2"])
        sig = float(d["sigma"])
        ridx = int(d["realization_index"])
        # Normalization this case's spectrum was computed with (the "chosen"
        # mode); scale every overlaid curve the same way so they share a scale.
        norm_mode = str(d["normalization"]) if "normalization" in d.files else "reference"

        fig, ax = plt.subplots(figsize=(8, 5))
        # ``spec`` is already normalized (saved that way); scale the overlays to match.
        ax.plot(E, spec, color="tab:red", linewidth=1.6,
                label=f"{name.split(':')[0]} realization #{ridx}")
        ax.plot(E1, _normalize_curve(E1, I1, norm_mode), color="tab:green", linewidth=1.4,
                label=overlay_label)
        if ref2 is not None:
            ax.plot(ref2[0], _normalize_curve(ref2[0], ref2[1], norm_mode),
                    color="black", linewidth=1.0, linestyle="--",
                    label="2-mol σ=0 (ref)", zorder=10)

        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(_ylabel_for_mode(norm_mode))
        ax.set_xlim(cfg.E_min, cfg.E_max)
        ax.set_title(
            f"{name}\n"
            rf"$\epsilon_1$={eps1:.3f} eV, $\epsilon_2$={eps2:.3f} eV "
            f"(σ={sig:g} eV)  +  {overlay_label}"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(
            results_dir,
            f"representative_spectrum_Nv{args.nv}_{sigma_tag}_case{L}_{args.out_suffix}.png",
        )
        fig.savefig(out, dpi=300, bbox_inches="tight")
        if not args.no_show:
            plt.show()
        plt.close(fig)
        made.append(out)
        print(f"  case {L} -> {out}")

    print(f"\nDone. {len(made)} overlay plot(s) written to '{results_dir}/'.")


if __name__ == "__main__":
    main()
