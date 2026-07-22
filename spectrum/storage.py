"""Per-Nv result persistence and resume/skip support."""

import csv
import os
from typing import Dict, List, Optional

import numpy as np

from .config import Config


def ensure_results_dir(cfg: Config) -> None:
    os.makedirs(cfg.results_dir, exist_ok=True)


def result_path(Nv: int, cfg: Config) -> str:
    return os.path.join(cfg.results_dir, f"spectrum_Nv{Nv}.npz")


def dat_path(Nv: int, cfg: Config) -> str:
    return os.path.join(cfg.results_dir, f"spectrum_Nv{Nv}.dat")


def exists(Nv: int, cfg: Config) -> bool:
    return os.path.isfile(result_path(Nv, cfg))


def save_result(result: Dict, cfg: Config) -> None:
    """Save a single Nv result: compressed .npz + a portable two-column .dat."""
    ensure_results_dir(cfg)
    Nv = int(result["Nv"])

    np.savez_compressed(
        result_path(Nv, cfg),
        Nv=Nv,
        dim=int(result["dim"]),
        E=result["E"],
        spectrum=result["spectrum"],
        reference_max=(
            float(result["reference_max"]) if result.get("reference_max") is not None else np.nan
        ),
        reference_area=(
            float(result["reference_area"]) if result.get("reference_area") is not None else np.nan
        ),
        normalization=result.get("normalization", "reference"),
        all_evals=result["all_evals"],
        all_intensity=result["all_intensity"],
        # provenance
        n_realizations=cfg.n_realizations,
        sigma=cfg.sigma,
        gamma=cfg.gamma,
        rng_seed=cfg.rng_seed,
        eps1=cfg.eps1,
        eps2=cfg.eps2,
        omega=cfg.omega,
        kappa=cfg.kappa,
        omega_c=cfg.omega_c,
        Omega=cfg.Omega,
    )

    np.savetxt(
        dat_path(Nv, cfg),
        np.column_stack([result["E"], result["spectrum"]]),
        header=(
            f"Nv={Nv}  dim={result['dim']}  n_real={cfg.n_realizations}  "
            f"normalization={result.get('normalization', 'reference')}\nE(eV)    intensity"
        ),
    )


def load_result(Nv: int, cfg: Config) -> Optional[Dict]:
    """Load a previously saved Nv result, or None if absent."""
    path = result_path(Nv, cfg)
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {
        "Nv": int(data["Nv"]),
        "sigma": float(data["sigma"]) if "sigma" in data.files else cfg.sigma,
        "dim": int(data["dim"]),
        "E": data["E"],
        "spectrum": data["spectrum"],
        "reference_max": (
            float(data["reference_max"])
            if "reference_max" in data.files and not np.isnan(data["reference_max"])
            else None
        ),
        "reference_area": (
            float(data["reference_area"])
            if "reference_area" in data.files and not np.isnan(data["reference_area"])
            else None
        ),
        "normalization": str(data["normalization"]) if "normalization" in data.files else "reference",
        "all_evals": data["all_evals"],
        "all_intensity": data["all_intensity"],
    }


# ---------------------------------------------------------------------------
# Sigma-sweep results (fixed Nv, one file per disorder strength)
# ---------------------------------------------------------------------------
def _sigma_tag(sigma: float) -> str:
    """Filesystem-safe tag for a sigma value, e.g. 0.03 -> 'sigma0.03'."""
    return f"sigma{sigma:g}"


def sigma_result_path(Nv: int, sigma: float, cfg: Config) -> str:
    return os.path.join(cfg.results_dir, f"spectrum_Nv{Nv}_{_sigma_tag(sigma)}.npz")


def sigma_dat_path(Nv: int, sigma: float, cfg: Config) -> str:
    return os.path.join(cfg.results_dir, f"spectrum_Nv{Nv}_{_sigma_tag(sigma)}.dat")


def sigma_exists(Nv: int, sigma: float, cfg: Config) -> bool:
    return os.path.isfile(sigma_result_path(Nv, sigma, cfg))


def save_sigma_result(result: Dict, cfg: Config) -> None:
    """Save a single (Nv, sigma) result: compressed .npz + portable .dat."""
    ensure_results_dir(cfg)
    Nv = int(result["Nv"])
    sigma = float(result["sigma"])

    np.savez_compressed(
        sigma_result_path(Nv, sigma, cfg),
        Nv=Nv,
        sigma=sigma,
        dim=int(result["dim"]),
        E=result["E"],
        spectrum=result["spectrum"],
        reference_max=(
            float(result["reference_max"]) if result.get("reference_max") is not None else np.nan
        ),
        reference_area=(
            float(result["reference_area"]) if result.get("reference_area") is not None else np.nan
        ),
        normalization=result.get("normalization", "reference"),
        all_evals=result["all_evals"],
        all_intensity=result["all_intensity"],
        # provenance
        n_realizations=cfg.n_realizations,
        gamma=cfg.gamma,
        rng_seed=cfg.rng_seed,
        eps1=cfg.eps1,
        eps2=cfg.eps2,
        omega=cfg.omega,
        kappa=cfg.kappa,
        omega_c=cfg.omega_c,
        Omega=cfg.Omega,
    )

    np.savetxt(
        sigma_dat_path(Nv, sigma, cfg),
        np.column_stack([result["E"], result["spectrum"]]),
        header=(
            f"Nv={Nv}  sigma={sigma:g}  dim={result['dim']}  "
            f"n_real={cfg.n_realizations}  "
            f"normalization={result.get('normalization', 'reference')}\nE(eV)    intensity"
        ),
    )


def load_sigma_result(Nv: int, sigma: float, cfg: Config) -> Optional[Dict]:
    """Load a previously saved (Nv, sigma) result, or None if absent."""
    path = sigma_result_path(Nv, sigma, cfg)
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {
        "Nv": int(data["Nv"]),
        "sigma": float(data["sigma"]),
        "dim": int(data["dim"]),
        "E": data["E"],
        "spectrum": data["spectrum"],
        "reference_max": (
            float(data["reference_max"])
            if "reference_max" in data.files and not np.isnan(data["reference_max"])
            else None
        ),
        "reference_area": (
            float(data["reference_area"])
            if "reference_area" in data.files and not np.isnan(data["reference_area"])
            else None
        ),
        "normalization": str(data["normalization"]) if "normalization" in data.files else "reference",
        "all_evals": data["all_evals"],
        "all_intensity": data["all_intensity"],
    }


# ---------------------------------------------------------------------------
# Realization-sweep results (fixed Nv & sigma, one file per realization count)
# ---------------------------------------------------------------------------
def realization_result_path(Nv: int, sigma: float, n: int, cfg: Config) -> str:
    return os.path.join(
        cfg.results_dir, f"spectrum_Nv{Nv}_{_sigma_tag(sigma)}_real{n}.npz"
    )


def realization_dat_path(Nv: int, sigma: float, n: int, cfg: Config) -> str:
    return os.path.join(
        cfg.results_dir, f"spectrum_Nv{Nv}_{_sigma_tag(sigma)}_real{n}.dat"
    )


def realization_exists(Nv: int, sigma: float, n: int, cfg: Config) -> bool:
    return os.path.isfile(realization_result_path(Nv, sigma, n, cfg))


def save_realization_result(result: Dict, cfg: Config) -> None:
    """Save a single (Nv, sigma, n_realizations) result: .npz + portable .dat."""
    ensure_results_dir(cfg)
    Nv = int(result["Nv"])
    sigma = float(result["sigma"])
    n = int(result["n_realizations"])

    np.savez_compressed(
        realization_result_path(Nv, sigma, n, cfg),
        Nv=Nv,
        sigma=sigma,
        n_realizations=n,
        dim=int(result["dim"]),
        E=result["E"],
        spectrum=result["spectrum"],
        reference_max=(
            float(result["reference_max"]) if result.get("reference_max") is not None else np.nan
        ),
        reference_area=(
            float(result["reference_area"]) if result.get("reference_area") is not None else np.nan
        ),
        normalization=result.get("normalization", "reference"),
        all_evals=result["all_evals"],
        all_intensity=result["all_intensity"],
        # provenance
        gamma=cfg.gamma,
        rng_seed=cfg.rng_seed,
        eps1=cfg.eps1,
        eps2=cfg.eps2,
        omega=cfg.omega,
        kappa=cfg.kappa,
        omega_c=cfg.omega_c,
        Omega=cfg.Omega,
    )

    np.savetxt(
        realization_dat_path(Nv, sigma, n, cfg),
        np.column_stack([result["E"], result["spectrum"]]),
        header=(
            f"Nv={Nv}  sigma={sigma:g}  n_realizations={n}  dim={result['dim']}  "
            f"normalization={result.get('normalization', 'reference')}"
            f"\nE(eV)    intensity"
        ),
    )


def load_realization_result(Nv: int, sigma: float, n: int, cfg: Config) -> Optional[Dict]:
    """Load a previously saved (Nv, sigma, n) result, or None if absent."""
    path = realization_result_path(Nv, sigma, n, cfg)
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {
        "Nv": int(data["Nv"]),
        "sigma": float(data["sigma"]),
        "n_realizations": int(data["n_realizations"]),
        "dim": int(data["dim"]),
        "E": data["E"],
        "spectrum": data["spectrum"],
        "reference_max": (
            float(data["reference_max"])
            if "reference_max" in data.files and not np.isnan(data["reference_max"])
            else None
        ),
        "reference_area": (
            float(data["reference_area"])
            if "reference_area" in data.files and not np.isnan(data["reference_area"])
            else None
        ),
        "normalization": str(data["normalization"]) if "normalization" in data.files else "reference",
        "all_evals": data["all_evals"],
        "all_intensity": data["all_intensity"],
    }


# ---------------------------------------------------------------------------
# Representative disorder realizations (analysis; see spectrum/representative.py)
# ---------------------------------------------------------------------------
def representative_metadata_path(cfg: Config, Nv: int) -> str:
    return os.path.join(cfg.results_dir, f"representative_metadata_Nv{Nv}_{_sigma_tag(cfg.sigma)}.csv")


def representative_summary_path(cfg: Config, Nv: int) -> str:
    return os.path.join(cfg.results_dir, f"representative_summary_Nv{Nv}_{_sigma_tag(cfg.sigma)}.csv")


def representative_scatter_path(cfg: Config, Nv: int) -> str:
    return os.path.join(cfg.results_dir, f"representative_scatter_Nv{Nv}_{_sigma_tag(cfg.sigma)}.png")


def _case_slug(case_name: str) -> str:
    """``"A: Resonant baseline"`` -> ``"caseA"``."""
    letter = case_name.split(":")[0].strip()
    return f"case{letter}"


def representative_spectrum_path(cfg: Config, Nv: int, case_name: str, ext: str = "png") -> str:
    return os.path.join(
        cfg.results_dir,
        f"representative_spectrum_Nv{Nv}_{_sigma_tag(cfg.sigma)}_{_case_slug(case_name)}.{ext}",
    )


def save_representative_metadata(metadata: List, cfg: Config, Nv: int) -> str:
    """Save every Pass-1 realization's lightweight metadata as a portable CSV."""
    ensure_results_dir(cfg)
    path = representative_metadata_path(cfg, Nv)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# Nv={Nv} sigma={cfg.sigma:g} n_real={len(metadata)} rng_seed={cfg.rng_seed}"])
        w.writerow(["index", "eps1", "eps2", "delta1", "delta2"])
        for m in metadata:
            w.writerow([m.index, f"{m.eps1:.6f}", f"{m.eps2:.6f}", f"{m.delta1:.6f}", f"{m.delta2:.6f}"])
    return path


def save_representative_summary(representative: Dict[str, Dict], cfg: Config, Nv: int) -> str:
    """Save the case-by-case conditions table: case, condition (description),
    realization index, eps1/eps2, delta1/delta2, disorder magnitude
    ``r = sqrt(delta1^2 + delta2^2)``, the case's selection score, and (for
    filter-based cases) whether that filter was actually satisfied.

    Written identically by both ``--representative`` and
    ``--representative-heatmap`` (each calls this once its Case A-H selections
    are known), so either workflow produces the same consolidated
    case-conditions file at ``representative_summary_Nv{Nv}_sigma{sigma}.csv``.
    ``score``/``filter_satisfied`` are written blank if a caller's per-case
    dict doesn't carry them (kept optional for backward compatibility).
    """
    ensure_results_dir(cfg)
    path = representative_summary_path(cfg, Nv)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "description", "realization_index", "eps1", "eps2",
                    "delta1", "delta2", "r", "score", "filter_satisfied"])
        for name, res in representative.items():
            r = (res["delta1"] ** 2 + res["delta2"] ** 2) ** 0.5
            score = res.get("score")
            w.writerow([
                name, res["description"], res["realization_index"],
                f"{res['eps1']:.6f}", f"{res['eps2']:.6f}",
                f"{res['delta1']:.6f}", f"{res['delta2']:.6f}",
                f"{r:.6f}",
                f"{score:.6f}" if score is not None else "",
                res.get("filter_satisfied", ""),
            ])
    return path


def save_representative_spectrum(res: Dict, cfg: Config, Nv: int) -> str:
    """Save one representative realization's Pass-2 spectrum: .npz + portable .dat."""
    ensure_results_dir(cfg)
    path = representative_spectrum_path(cfg, Nv, res["case"], ext="npz")
    np.savez_compressed(
        path,
        Nv=Nv, dim=int(res["dim"]), case=res["case"], description=res["description"],
        realization_index=int(res["realization_index"]),
        eps1=float(res["eps1"]), eps2=float(res["eps2"]),
        delta1=float(res["delta1"]), delta2=float(res["delta2"]),
        E=res["E"], spectrum=res["spectrum"],
        sigma=cfg.sigma, gamma=cfg.gamma, rng_seed=cfg.rng_seed,
        normalization=cfg.NORMALIZATION,
    )
    dat_path_ = representative_spectrum_path(cfg, Nv, res["case"], ext="dat")
    np.savetxt(
        dat_path_,
        np.column_stack([res["E"], res["spectrum"]]),
        header=(
            f"case={res['case']}  Nv={Nv}  realization={res['realization_index']}  "
            f"eps1={res['eps1']:.6f}  eps2={res['eps2']:.6f}\nE(eV)    intensity"
        ),
    )
    return path
