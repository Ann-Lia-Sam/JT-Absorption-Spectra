"""Hamiltonian assembly.

The Hamiltonian splits into a disorder-independent part (vibrational + cavity
diagonal, Jahn-Teller coupling, matter-cavity coupling) that we build once per
``Nv``, and a disorder-dependent electronic diagonal that changes each
realization. This is numerically identical to rebuilding the full matrix every
realization, but far cheaper.
"""

#import time

from typing import Dict, List

import numpy as np
from tqdm import tqdm

from . import operators as ops
from .config import Config

State = tuple


def _accumulate_two_op_terms(H, basis_final, basis_index, terms, coeff):
    """For each state, apply first op then second op and add ``coeff*c1*c2``.

    ``terms`` is a list of (first_op, second_op) callables, each taking a single
    state argument and returning ``(c, new_state)`` or ``None``.
    """
    for state in basis_final:
        i = basis_index[state]
        for first_op, second_op in terms:
            r1 = first_op(state)
            if r1 is None:
                continue
            c1, s1 = r1
            r2 = second_op(s1)
            if r2 is None:
                continue
            c2, s2 = r2
            j = basis_index.get(s2)
            if j is not None:
                H[j, i] += coeff * c1 * c2


def build_static_hamiltonian(
    Nv: int,
    basis_final: List[State],
    basis_index: Dict[State, int],
    cfg: Config,
    show_progress: bool = True,
) -> np.ndarray:
    """Disorder-independent Hamiltonian (everything except electronic energy)."""
    dim = len(basis_final)
    H = np.zeros((dim, dim))

    iterator = basis_final
    if show_progress:
        iterator = tqdm(
            basis_final, desc=f"  H(Nv={Nv}) diag", leave=False, unit="state"
        )

    # ---- Diagonal: vibrational + cavity photon energy ----
    for state in iterator:
        i = basis_index[state]
        e1, np1, nm1, e2, np2, nm2, cp, cm = state
        E = cfg.omega * (np1 + nm1 + np2 + nm2) + cfg.omega_c * (cp + cm)
        H[i, i] += E

    Nv_ = Nv
    Np_ = cfg.Np

    # ---- Jahn-Teller (vibronic) coupling, molecule 1 ----
    jt1 = [
        (ops.Ep_to_Em_1, lambda s: ops.bp1_dag(s, Nv_)),
        (ops.Ep_to_Em_1, ops.bm1),
        (ops.Em_to_Ep_1, ops.bp1),
        (ops.Em_to_Ep_1, lambda s: ops.bm1_dag(s, Nv_)),
    ]
    _accumulate_two_op_terms(H, basis_final, basis_index, jt1, cfg.kappa)

    # ---- Jahn-Teller (vibronic) coupling, molecule 2 ----
    jt2 = [
        (ops.Ep_to_Em_2, lambda s: ops.bp2_dag(s, Nv_)),
        (ops.Ep_to_Em_2, ops.bm2),
        (ops.Em_to_Ep_2, ops.bp2),
        (ops.Em_to_Ep_2, lambda s: ops.bm2_dag(s, Nv_)),
    ]
    _accumulate_two_op_terms(H, basis_final, basis_index, jt2, cfg.kappa)

    # ---- Matter-cavity coupling ----
    g = cfg.g
    mc = [
        # molecule 1
        (ops.Ep_to_A_1, lambda s: ops.ap_dag(s, Np_)),
        (ops.Em_to_A_1, lambda s: ops.am_dag(s, Np_)),
        (ops.A_to_Ep_1, ops.ap),
        (ops.A_to_Em_1, ops.am),
        # molecule 2
        (ops.Ep_to_A_2, lambda s: ops.ap_dag(s, Np_)),
        (ops.Em_to_A_2, lambda s: ops.am_dag(s, Np_)),
        (ops.A_to_Ep_2, ops.ap),
        (ops.A_to_Em_2, ops.am),
    ]
    _accumulate_two_op_terms(H, basis_final, basis_index, mc, g)

    return H


def electronic_masks(basis_final: List[State]):
    """Boolean masks: whether molecule 1 / molecule 2 is electronically excited."""
    mask1 = np.array([s[0] != "A" for s in basis_final], dtype=float)
    mask2 = np.array([s[3] != "A" for s in basis_final], dtype=float)
    return mask1, mask2


def electronic_diagonal(mask1, mask2, eps1_dis: float, eps2_dis: float):
    """Electronic diagonal energy for one disorder realization."""
    return eps1_dis * mask1 + eps2_dis * mask2
