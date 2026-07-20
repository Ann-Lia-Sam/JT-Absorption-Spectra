"""Central configuration: all physical and run parameters.

Values are taken verbatim from the notebook. The electronic energies ``eps1`` /
``eps2`` were only present in a commented-out notebook cell; they are defined
here so the Hamiltonian is fully specified.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Config:
    # ---------------- Physical parameters (eV) ----------------
    omega: float = 0.08196          # vibrational quantum
    kappa: float = 0.180312         # Jahn-Teller (vibronic) coupling
    omega_c: float = 6.85           # cavity photon energy
    Omega: float = 2 * 6.85 * 0.05  # collective light-matter coupling scale

    N: int = 2                      # number of molecules
    Np: int = 2                     # cavity photon-number cutoff (states 0, 1)

    # Electronic site energies: eps_i = eps + delta_i
    eps: float = 7.0
    delta1: float = 0
    delta2: float = 0

    # ---------------- Normalization ----------------
    # Applied exactly once, after Lorentzian broadening and disorder averaging.
    # "reference"      -- divide by the raw peak of this Nv's σ=0 (no-disorder)
    #   spectrum, shared across every sigma/realization-count for that Nv.
    # "reference_area" -- divide by the raw trapezoidal area of this Nv's σ=0
    #   (no-disorder) spectrum, shared across every sigma/realization-count
    #   for that Nv (same reference spectrum as "reference", different
    #   summary statistic taken from it).
    # "area"           -- divide the final averaged spectrum by its own
    #   integral (trapezoidal over E), so that ∫I(E)dE = 1.
    # "none"           -- leave the averaged spectrum unchanged.
    NORMALIZATION: str = "reference_area"

    # ---------------- Disorder ----------------
    sigma: float = 0.24            # disorder strength W (eV)
    n_realizations: int = 1
    rng_seed: int = 1234
    # Parallel disorder realizations across worker processes. None or 1 -> the
    # (unchanged) serial path. >1 -> that many processes, each pinned to a
    # single BLAS thread. Results match the serial path (disorder draws are
    # pre-generated in the parent in the same order).
    n_workers: Optional[int] = None

    # ---------------- Spectrum grid / broadening ----------------
    E_min: float = 6.0
    E_max: float = 8.0
    E_points: int = 2000
    gamma: float = 0.044            # Lorentzian FWHM (eV)

    # ---------------- Basis / sector selection ----------------
    initial_state: Tuple = ("A", 0, 0, "A", 0, 0, 0, 1)
    nex_target: int = 1             # single-excitation subspace
    jz_target: int = -1             # Jz symmetry sector

    # ---------------- Nv sweep ----------------
    nv_list: List[int] = field(default_factory=lambda: list(range(2, 13)))

    # ---------------- Sigma sweep (fixed Nv) ----------------
    sigma_sweep_nv: int = 12        # Nv used for the disorder-strength sweep
    sigma_list: List[float] = field(
        default_factory=lambda: [0, 0.01, 0.03, 0.05, 0.08]
    )

    # ---------------- Realization sweep (fixed Nv, fixed sigma) ----------------
    realization_sweep_nv: int = 12  # Nv used for the realization-count sweep
    realization_list: List[int] = field(
        default_factory=lambda: [ 100 ]
    )

    # ---------------- Representative disorder realizations (analysis) ----------------
    # These parameters drive a *separate* analysis workflow (--representative):
    # Pass 1 reuses the existing disorder-averaged spectrum unchanged and records
    # lightweight per-realization metadata (eps1, eps2); a handful of physically
    # meaningful realizations are then automatically selected and, in Pass 2,
    # individually recomputed (reusing the same Hamiltonian/diagonalization/
    # broadening/normalization building blocks, unmodified). See
    # spectrum/representative.py. Does not affect the absorption-spectrum pipeline.
    representative_nv: int = 12          # Nv used for this analysis
    representative_n_realizations: int = 300  # realizations sampled for the (eps1,eps2) cloud

    # ---------------- Paths ----------------
    results_dir: str = "results"
    # "Without disorder" reference curve. Columns: E, I (0 and 1). Plotted on top
    # of everything. Set to None (or use --no-reference) to skip it.
    reference_file: Optional[str] = "spectrum_ref_2mol_2.2.pl"

    # ---------------- Derived ----------------
    @property
    def eps1(self) -> float:
        return self.eps + self.delta1

    @property
    def eps2(self) -> float:
        return self.eps + self.delta2

    @property
    def g(self) -> float:
        """Per-molecule matter-cavity coupling."""
        import numpy as np
        return self.Omega / (2 * np.sqrt(self.N))


# A ready-to-use default instance.
default_config = Config()
