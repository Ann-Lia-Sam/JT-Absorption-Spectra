#!/usr/bin/env python3
"""Sweep Nv, compute disorder-averaged spectra, save each, and overlay them.

Examples
--------
    # full default sweep (Nv = 2..12)
    python main.py

    # quick smoke test
    python main.py --nv 2 3 --realizations 3 --no-show

    # recompute everything from scratch
    python main.py --force
"""

import argparse
import os

from tqdm import tqdm

from spectrum.config import Config
from spectrum.plotting import overlay, overlay_realizations, overlay_sigma
from spectrum.spectrum import (
    compute_spectrum_for_Nv,
    compute_spectrum_realization_sweep,
    compute_spectrum_sigma_sweep,
)
from spectrum.storage import (
    dat_path,
    ensure_results_dir,
    exists,
    load_realization_result,
    load_result,
    load_sigma_result,
    realization_dat_path,
    realization_exists,
    realization_result_path,
    result_path,
    save_realization_result,
    save_result,
    save_sigma_result,
    sigma_dat_path,
    sigma_exists,
    sigma_result_path,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nv", type=int, nargs="+", default=None,
                   help="Nv values to compute (default: config NV_LIST, i.e. 2..12).")
    p.add_argument("--realizations", type=int, default=None,
                   help="Override number of disorder realizations.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a saved result exists.")
    p.add_argument("--no-show", action="store_true",
                   help="Save the overlay figure without opening a window.")
    p.add_argument("--results-dir", default=None,
                   help="Override output directory (default: config results_dir).")
    p.add_argument("--workers", type=int, default=None,
                   help="Run disorder realizations across this many processes "
                        "(default: serial). >1 enables parallelism; each worker is "
                        "pinned to one BLAS thread. Results match the serial run.")
    # ---- sigma-sweep mode (fixed Nv, multiple disorder strengths) ----
    p.add_argument("--sigma-sweep", action="store_true",
                   help="Sweep disorder strength sigma for a single Nv instead of "
                        "sweeping Nv.")
    p.add_argument("--sigmas", type=float, nargs="+", default=None,
                   help="Sigma values for --sigma-sweep (default: config sigma_list).")
    # ---- realization-sweep mode (fixed Nv & sigma, multiple realization counts) ----
    p.add_argument("--realization-sweep", action="store_true",
                   help="Sweep the number of disorder realizations for a single Nv "
                        "(convergence study) instead of sweeping Nv.")
    p.add_argument("--realization-list", type=int, nargs="+", default=None,
                   help="Realization counts for --realization-sweep "
                        "(default: config realization_list).")
    # ---- reference ("without disorder") curve, overlaid on top ----
    p.add_argument("--reference", action=argparse.BooleanOptionalAction, default=True,
                   help="Overlay the reference curve on top (use --no-reference to hide).")
    p.add_argument("--reference-file", default=None,
                   help="Path to the reference curve (default: config reference_file).")
    # ---- heatmap mode (vibronic basis: P(v) / participation ratio) ----
    p.add_argument("--heatmap", action="store_true",
                   help="Run the vibronic-basis P(v)/PR heatmap workflow (SM Sec. I.A & II) "
                        "instead of the absorption-spectrum sweep. Always produces the σ=0 "
                        "Fig. S1 heatmap; adds a disorder-averaged, energy-resolved heatmap "
                        "when --heatmap-sigma>0 or --heatmap-realizations>1.")
    p.add_argument("--heatmap-nv", type=int, default=None,
                   help="Fock states per vibrational mode for the vibronic diagonalization "
                        "(default: config heatmap_nv, i.e. 18 as in the paper).")
    p.add_argument("--heatmap-sigma", type=float, default=None,
                   help="Disorder strength for the averaged heatmap (default: config).")
    p.add_argument("--heatmap-realizations", type=int, default=None,
                   help="Disorder realizations for the averaged heatmap (default: config).")
    p.add_argument("--heatmap-threshold", type=float, default=None,
                   help="Relative intensity threshold selecting bright polariton states "
                        "for the Fig. S1 heatmap (default: config heatmap_bright_threshold).")
    p.add_argument("--heatmap-which", choices=["1", "2", "avg"], default=None,
                   help="Which molecule's P(v) to report: '1', '2', or 'avg' (default: config).")
    return p.parse_args()


def apply_reference_args(args, cfg: Config) -> None:
    """Resolve the reference-curve CLI flags onto the config."""
    if not args.reference:
        cfg.reference_file = None
    elif args.reference_file is not None:
        cfg.reference_file = args.reference_file


def run_sigma_sweep(args, cfg: Config) -> None:
    """Compute spectra for one Nv across several disorder strengths and overlay."""
    # Nv for the sweep: first of --nv if given, else config default.
    Nv = args.nv[0] if args.nv else cfg.sigma_sweep_nv
    sigmas = args.sigmas if args.sigmas is not None else cfg.sigma_list

    ensure_results_dir(cfg)

    # Resume: figure out which sigmas still need computing.
    to_compute = [s for s in sigmas if args.force or not sigma_exists(Nv, s, cfg)]
    results = []
    summary = []  # (sigma, status)

    # Load any already-saved sigmas first.
    for s in sigmas:
        if s not in to_compute:
            res = load_sigma_result(Nv, s, cfg)
            results.append(res)
            summary.append((s, "loaded"))
            tqdm.write(f"[Nv={Nv} sigma={s:g}] loaded from {sigma_result_path(Nv, s, cfg)}")

    # Compute the rest (static H built once inside, reused across these sigmas).
    if to_compute:
        computed = compute_spectrum_sigma_sweep(Nv, to_compute, cfg, show_progress=True)
        for res in computed:
            save_sigma_result(res, cfg)
            results.append(res)
            summary.append((res["sigma"], "computed"))
            tqdm.write(
                f"[Nv={Nv} sigma={res['sigma']:g}] computed & saved -> "
                f"{sigma_result_path(Nv, res['sigma'], cfg)} (dim={res['dim']})"
            )

    out_fig = os.path.join(cfg.results_dir, f"overlay_sigma_Nv{Nv}.png")
    overlay_sigma(results, cfg, out_fig, show=not args.no_show)

    print("\n=== Sigma-sweep summary (Nv={}) ===".format(Nv))
    for s, status in sorted(summary):
        print(f"  sigma={s:<6g}  {status:<8}  {sigma_result_path(Nv, s, cfg)}  |  "
              f"{sigma_dat_path(Nv, s, cfg)}")
    print(f"\nOverlay figure: {out_fig}")


def run_realization_sweep(args, cfg: Config) -> None:
    """Compute spectra for one Nv across realization counts and overlay them."""
    Nv = args.nv[0] if args.nv else cfg.realization_sweep_nv
    counts = args.realization_list if args.realization_list is not None else cfg.realization_list
    counts = sorted(set(int(n) for n in counts))
    sigma = cfg.sigma

    ensure_results_dir(cfg)

    to_compute = [n for n in counts if args.force or not realization_exists(Nv, sigma, n, cfg)]
    results = []
    summary = []  # (n, status)

    # Load any already-saved counts.
    for n in counts:
        if n not in to_compute:
            res = load_realization_result(Nv, sigma, n, cfg)
            results.append(res)
            summary.append((n, "loaded"))
            tqdm.write(f"[Nv={Nv} n={n}] loaded from {realization_result_path(Nv, sigma, n, cfg)}")

    # Compute the rest in a single disorder pass (at the largest requested count).
    if to_compute:
        computed = compute_spectrum_realization_sweep(Nv, to_compute, cfg, show_progress=True)
        for res in computed:
            save_realization_result(res, cfg)
            results.append(res)
            summary.append((res["n_realizations"], "computed"))
            tqdm.write(
                f"[Nv={Nv} n={res['n_realizations']}] computed & saved -> "
                f"{realization_result_path(Nv, sigma, res['n_realizations'], cfg)} (dim={res['dim']})"
            )

    out_fig = os.path.join(cfg.results_dir, f"overlay_realizations_Nv{Nv}_sigma{sigma:g}.png")
    overlay_realizations(results, cfg, out_fig, show=not args.no_show)

    print(f"\n=== Realization-sweep summary (Nv={Nv}, sigma={sigma:g}) ===")
    for n, status in sorted(summary):
        print(f"  n={n:<6} {status:<8}  {realization_result_path(Nv, sigma, n, cfg)}  |  "
              f"{realization_dat_path(Nv, sigma, n, cfg)}")
    print(f"\nOverlay figure: {out_fig}")


def run_heatmap(args, cfg: Config) -> None:
    """Vibronic-basis P(v)/PR heatmap workflow (SM Sec. I.A & II).

    Always produces the σ=0 Fig. S1 heatmap (single-molecule occupation of the
    vibronic sectors for the bright polaritonic states). If disorder is requested
    (``heatmap_sigma>0`` or ``heatmap_realizations>1``), also produces the
    disorder-averaged, energy-resolved sector-population heatmap, using the same
    disorder draws as the absorption-spectrum pipeline.
    """
    # Deferred imports: keep the fast spectrum path free of these modules.
    from spectrum import heatmap as hm
    from spectrum.plotting import plot_disorder_heatmap, plot_figS1
    from spectrum.storage import save_disorder_heatmap, save_figS1

    ensure_results_dir(cfg)

    # ---- σ=0 Fig. S1 reproduction (always) ----
    print(f"[heatmap] σ=0 Fig. S1 (Nv={cfg.heatmap_nv}) ...")
    figS1 = hm.sigma0_heatmap(cfg)
    npz = save_figS1(figS1, cfg)
    out_figS1 = os.path.join(cfg.results_dir, f"heatmap_figS1_Nv{cfg.heatmap_nv}.png")
    plot_figS1(figS1, cfg, out_figS1, show=not args.no_show)
    n_bright = figS1.heatmap.shape[1]
    print(f"  bright states = {n_bright}  |  sector v in [{figS1.grid[0]}, {figS1.grid[-1]}]  |  "
          f"PR(bright) in [{figS1.pr.min():.2f}, {figS1.pr.max():.2f}]")
    print(f"  saved -> {npz}\n         {out_figS1}")

    # ---- disorder-averaged, energy-resolved heatmap (if requested) ----
    if cfg.heatmap_sigma > 0 or cfg.heatmap_realizations > 1:
        print(f"\n[heatmap] disorder average (Nv={cfg.heatmap_nv}, σ={cfg.heatmap_sigma:g}, "
              f"{cfg.heatmap_realizations} real.) ...")
        dh = hm.disorder_average_heatmap(cfg, show_progress=True)
        npz2 = save_disorder_heatmap(dh, cfg)
        out_dh = os.path.join(
            cfg.results_dir,
            f"heatmap_Nv{cfg.heatmap_nv}_sigma{cfg.heatmap_sigma:g}_real{cfg.heatmap_realizations}.png",
        )
        plot_disorder_heatmap(dh, cfg, out_dh, show=not args.no_show)
        good = dh.spectrum > 1e-6 * dh.spectrum.max()
        pr_lo = float(dh.PR[good].min()) if good.any() else 0.0
        pr_hi = float(dh.PR[good].max()) if good.any() else 0.0
        print(f"  PR(E) over the spectrum in [{pr_lo:.2f}, {pr_hi:.2f}]")
        print(f"  saved -> {npz2}\n         {out_dh}")


def _apply_heatmap_args(args, cfg: Config) -> None:
    if args.heatmap_nv is not None:
        cfg.heatmap_nv = args.heatmap_nv
    if args.heatmap_sigma is not None:
        cfg.heatmap_sigma = args.heatmap_sigma
    if args.heatmap_realizations is not None:
        cfg.heatmap_realizations = args.heatmap_realizations
    if args.heatmap_threshold is not None:
        cfg.heatmap_bright_threshold = args.heatmap_threshold
    if args.heatmap_which is not None:
        cfg.heatmap_which_molecule = args.heatmap_which


def main():
    args = parse_args()

    cfg = Config()
    if args.nv is not None:
        cfg.nv_list = args.nv
    if args.realizations is not None:
        cfg.n_realizations = args.realizations
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir
    if args.workers is not None:
        cfg.n_workers = args.workers
    apply_reference_args(args, cfg)
    _apply_heatmap_args(args, cfg)

    if args.heatmap:
        run_heatmap(args, cfg)
        return

    if args.sigma_sweep:
        run_sigma_sweep(args, cfg)
        return

    if args.realization_sweep:
        run_realization_sweep(args, cfg)
        return

    ensure_results_dir(cfg)

    results = []
    summary = []  # (Nv, status, dim)

    for Nv in tqdm(cfg.nv_list, desc="Nv sweep", unit="Nv"):
        if exists(Nv, cfg) and not args.force:
            res = load_result(Nv, cfg)
            results.append(res)
            summary.append((Nv, "loaded", res["dim"]))
            tqdm.write(f"[Nv={Nv}] loaded from {result_path(Nv, cfg)} (dim={res['dim']})")
            continue

        res = compute_spectrum_for_Nv(Nv, cfg, show_progress=True)
        save_result(res, cfg)  # persist immediately so progress survives a crash
        results.append(res)
        summary.append((Nv, "computed", res["dim"]))
        tqdm.write(
            f"[Nv={Nv}] computed & saved -> {result_path(Nv, cfg)} "
            f"(dim={res['dim']}, {cfg.n_realizations} realizations)"
        )

    # ---- Overlay plot ----
    out_fig = os.path.join(cfg.results_dir, "overlay_spectra.png")
    overlay(results, cfg, out_fig, show=not args.no_show)

    # ---- Summary ----
    print("\n=== Summary ===")
    for Nv, status, dim in summary:
        print(f"  Nv={Nv:>2}  {status:<8}  dim={dim:<6}  "
              f"{result_path(Nv, cfg)}  |  {dat_path(Nv, cfg)}")
    print(f"\nOverlay figure: {out_fig}")


if __name__ == "__main__":
    main()
