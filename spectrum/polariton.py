"""Vibronic-to-polaritonic transformation (Supplementary Material, Sec. I.A).

Given the single-molecule vibronic eigenbases of molecule 1 and molecule 2
(:mod:`spectrum.vibronic`), we build the two-molecule + photon Hamiltonian in the
vibronic basis, restricted to the ``(nex = 1, j = -1)`` sector, and diagonalize it
to obtain the polaritonic eigenstates.

Basis states are labelled ``(a, b, p)``:

* ``a`` -- index of molecule 1's vibronic eigenstate ``|lambda_a^{(1)}>``,
* ``b`` -- index of molecule 2's vibronic eigenstate ``|lambda_b^{(2)}>``,
* ``p in {+1, 0, -1}`` -- one photon in the ``+`` (LCP) mode, no photon, or one
  photon in the ``-`` (RCP) mode.

This is the *distinguishable-molecule* construction demanded by disorder
(molecules 1 and 2 have different electronic energies ``eps + delta_1`` and
``eps + delta_2``, hence different vibronic bases). At ``sigma = 0`` the two
bases coincide and the construction reproduces the permutation-symmetric result
of the paper; the bright (nonzero-intensity) eigenstates are then symmetric under
1<->2 exchange, so the single-molecule ``P(v)`` computed here (see
:mod:`spectrum.participation`) equals Eq. (S10) exactly.

The Hamiltonian is Eq. (S1):

    H' = sum_{a,k} lambda_a |lambda_a>_k<lambda_a|_k
       + omega_c (a+^dag a+ + a-^dag a-)
       + g sum_{a,b,k} ( alpha_{a,b} |lambda_a>_k<lambda_b|_k a+^dag
                       + beta_{a,b}  |lambda_a>_k<lambda_b|_k a-^dag + h.c. )

with g = Omega / (2 sqrt(N)).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import Config
from .vibronic import VibronicSolution, alpha_beta

# A sector basis state: (a, b, p).
SectorState = Tuple[int, int, int]


@dataclass
class PolaritonSector:
    """The ``(nex = 1, j = -1)`` two-molecule + photon vibronic sector."""

    states: List[SectorState]
    index: Dict[SectorState, int]
    a: np.ndarray          # (dim,) molecule-1 vibronic eigenstate index
    b: np.ndarray          # (dim,) molecule-2 vibronic eigenstate index
    p: np.ndarray          # (dim,) photon label in {+1, 0, -1}
    v1: np.ndarray         # (dim,) vibronic sector of molecule 1
    v2: np.ndarray         # (dim,) vibronic sector of molecule 2

    @property
    def dim(self) -> int:
        return len(self.states)


def _group_by_v(indices: np.ndarray, v: np.ndarray) -> Dict[int, List[int]]:
    """Group eigenstate indices by their vibronic sector value."""
    out: Dict[int, List[int]] = {}
    for i in indices:
        out.setdefault(int(v[i]), []).append(int(i))
    return out


def build_sector(vib1: VibronicSolution, vib2: VibronicSolution, cfg: Config) -> PolaritonSector:
    """Enumerate the ``(nex = cfg.nex_target, j = cfg.jz_target)`` sector basis.

    Built case-by-case from the conservation laws instead of brute-force filtering:

    * photon states ``p = +/-1``: both molecules electronic-ground (A-type),
      ``v_a + v_b = j - p``;
    * matter states ``p = 0``: exactly one molecule electronically excited
      (E-type), ``v_a + v_b = j``.
    """
    nex_target = cfg.nex_target
    j_target = cfg.jz_target
    if nex_target != 1:
        raise NotImplementedError(
            f"The vibronic sector builder currently supports nex=1 only (got {nex_target})."
        )

    A1 = np.where(~vib1.is_excited)[0]
    E1 = np.where(vib1.is_excited)[0]
    A2 = np.where(~vib2.is_excited)[0]
    E2 = np.where(vib2.is_excited)[0]

    A1_by_v = _group_by_v(A1, vib1.v)
    A2_by_v = _group_by_v(A2, vib2.v)

    states: List[SectorState] = []

    # --- Photon states: both molecules A-type, one photon p = +/-1 ---
    for p in (+1, -1):
        need = j_target - p           # required v_a + v_b
        for va, a_list in A1_by_v.items():
            vb = need - va
            for a in a_list:
                for b in A2_by_v.get(vb, ()):
                    states.append((a, b, p))

    # --- Matter states: p = 0, exactly one molecule excited (E-type) ---
    E1_by_v = _group_by_v(E1, vib1.v)
    E2_by_v = _group_by_v(E2, vib2.v)
    #   molecule 1 excited, molecule 2 ground
    for va, a_list in E1_by_v.items():
        vb = j_target - va
        for a in a_list:
            for b in A2_by_v.get(vb, ()):
                states.append((a, b, 0))
    #   molecule 1 ground, molecule 2 excited
    for vb, b_list in E2_by_v.items():
        va = j_target - vb
        for a in A1_by_v.get(va, ()):
            for b in b_list:
                states.append((a, b, 0))

    index = {st: i for i, st in enumerate(states)}
    a_arr = np.array([s[0] for s in states], dtype=int)
    b_arr = np.array([s[1] for s in states], dtype=int)
    p_arr = np.array([s[2] for s in states], dtype=int)
    v1 = vib1.v[a_arr]
    v2 = vib2.v[b_arr]
    return PolaritonSector(states, index, a_arr, b_arr, p_arr, v1, v2)


def build_hamiltonian(
    vib1: VibronicSolution,
    vib2: VibronicSolution,
    sector: PolaritonSector,
    cfg: Config,
) -> np.ndarray:
    """Assemble the polaritonic Hamiltonian (Eq. (S1)) in the vibronic sector basis.

    Diagonal: ``lambda_a^{(1)} + lambda_b^{(2)} + omega_c * (photon number)``.
    Off-diagonal: matter-cavity coupling ``g * alpha`` (+ mode) / ``g * beta``
    (- mode), connecting each matter state (one molecule E-type, ``p = 0``) to the
    photon states reached by de-exciting that molecule and creating a photon.
    """
    dim = sector.dim
    H = np.zeros((dim, dim))

    # --- Diagonal ---
    photon_number = (sector.p != 0).astype(float)
    H[np.arange(dim), np.arange(dim)] = (
        vib1.evals[sector.a] + vib2.evals[sector.b] + cfg.omega_c * photon_number
    )

    # --- Matter-cavity coupling ---
    g = cfg.g
    alpha1, beta1 = alpha_beta(vib1)
    alpha2, beta2 = alpha_beta(vib2)

    A1_by_v = _group_by_v(np.where(~vib1.is_excited)[0], vib1.v)
    A2_by_v = _group_by_v(np.where(~vib2.is_excited)[0], vib2.v)

    def add(target: SectorState, source_i: int, m: float) -> None:
        if m == 0.0:
            return
        j = sector.index.get(target)
        if j is None:
            return
        H[j, source_i] += m
        H[source_i, j] += m

    for i, (a, b, p) in enumerate(sector.states):
        if p != 0:
            continue  # couplings are generated from the matter (p=0) side
        if vib1.is_excited[a]:
            # molecule 1 de-excites: E-type a -> A-type a', emitting a photon.
            va = int(vib1.v[a])
            for ap in A1_by_v.get(va - 1, ()):        # + photon: v_a' = v_a - 1
                add((ap, b, +1), i, g * alpha1[ap, a])
            for ap in A1_by_v.get(va + 1, ()):        # - photon: v_a' = v_a + 1
                add((ap, b, -1), i, g * beta1[ap, a])
        if vib2.is_excited[b]:
            # molecule 2 de-excites: E-type b -> A-type b', emitting a photon.
            vb = int(vib2.v[b])
            for bp in A2_by_v.get(vb - 1, ()):        # + photon
                add((a, bp, +1), i, g * alpha2[bp, b])
            for bp in A2_by_v.get(vb + 1, ()):        # - photon
                add((a, bp, -1), i, g * beta2[bp, b])

    return H


def initial_state_index(vib1: VibronicSolution, vib2: VibronicSolution, sector: PolaritonSector, cfg: Config) -> int:
    """Index of the absorption initial state: both molecules in ``|A,0,0>`` and one RCP photon.

    The RCP photon is the ``-`` mode (``p = -1``), matching the primitive initial
    state ``('A',0,0,'A',0,0,0,1)`` used by the spectrum pipeline (``cm = 1``).
    """
    a0 = vib1.ground_index()
    b0 = vib2.ground_index()
    key = (a0, b0, -1)
    if key not in sector.index:
        raise ValueError("Initial RCP-photon state is not in the (nex=1, j=-1) sector.")
    return sector.index[key]


@dataclass
class PolaritonSolution:
    """Diagonalized polaritonic system for one disorder realization."""

    sector: PolaritonSector
    evals: np.ndarray       # (dim,) polariton energies (ascending)
    evecs: np.ndarray       # (dim, dim) columns are polariton eigenvectors C_{(a,b,p)}
    i0: int                 # initial-state index in the sector basis
    intensity: np.ndarray   # (dim,) |<initial|psi_n>|^2 absorption intensities


def solve(vib1: VibronicSolution, vib2: VibronicSolution, cfg: Config) -> PolaritonSolution:
    """Full vibronic->polaritonic pipeline for one realization: build, diagonalize, intensities."""
    sector = build_sector(vib1, vib2, cfg)
    H = build_hamiltonian(vib1, vib2, sector, cfg)
    evals, evecs = np.linalg.eigh(H)
    i0 = initial_state_index(vib1, vib2, sector, cfg)
    intensity = np.abs(evecs[i0, :]) ** 2
    return PolaritonSolution(sector, evals, evecs, i0, intensity)
