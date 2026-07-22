"""One-molecule vibronic basis construction (Supplementary Material, Sec. I.A).

Diagonalizing the single-molecule Jahn-Teller Hamiltonian (Eq. (1) of the main
text) gives the *vibronic eigenstates*

    |lambda^v_j>_k = sum_{s,n+,n-} C^{s}_{n+,n-} |s, n+, n->_k ,   s in {A, E+, E-}

each with a well-defined vibronic angular momentum quantum number

    v = <V_k> = 2 (n+ - n-) + S_z ,   S_z(E+)=+1, S_z(E-)=-1, S_z(A)=0 .

The single-molecule Hamiltonian (Eq. (1)) has no coupling between the electronic
ground block (|A, n+, n->, the "trivial" eigenvectors, even v, no electronic
excitation) and the excited block (superpositions of |E+/-, n+, n->, the
"non-trivial" eigenvectors, odd v, one electronic excitation). We therefore
diagonalize the two blocks separately, which classifies every eigenvector
cleanly and avoids any accidental A/E mixing within a numerically degenerate
subspace.

Because the excited electronic energy enters Eq. (1) only as ``eps_k * I`` on the
E-block, the vibronic *eigenvectors* (and hence the sector labels, alpha, beta of
Sec. I.A) are independent of ``eps_k`` and of disorder; only the E-block
*eigenvalues* shift rigidly by ``+eps_k``. :func:`diagonalize_single_molecule`
performs the full diagonalization from scratch every time it is called (used as
the from-scratch reference implementation and for validation). For repeated
calls at different ``eps_k`` (e.g. one per disorder realization, one per
molecule), :func:`diagonalize_single_molecule_reference` diagonalizes once at
``eps_k = 0`` and :func:`shift_reference` reuses that result for any ``eps_k``
via an O(1) rigid eigenvalue shift -- eliminating hundreds of redundant
diagonalizations in the disorder-averaged heatmap workflow.
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

from .config import Config

# Single-molecule primitive state: (s, n+, n-) with s in {"A", "Ep", "Em"}.
SingleState = Tuple[str, int, int]

ELECTRONIC = ("A", "Ep", "Em")
_SZ = {"A": 0, "Ep": +1, "Em": -1}


def single_state_v(state: SingleState) -> int:
    """Vibronic angular momentum v = 2(n+ - n-) + S_z of a primitive state."""
    s, npv, nmv = state
    return 2 * (npv - nmv) + _SZ[s]


def build_single_molecule_basis(Nv: int) -> Tuple[List[SingleState], Dict[SingleState, int]]:
    """Primitive single-molecule basis with ``Nv`` Fock states per mode.

    Ordered electronic-block-first (all ``A`` states, then ``Ep``, then ``Em``)
    so the ground/excited blocks are contiguous.
    """
    basis: List[SingleState] = []
    for s in ELECTRONIC:
        for npv, nmv in product(range(Nv), range(Nv)):
            basis.append((s, npv, nmv))
    index = {st: i for i, st in enumerate(basis)}
    return basis, index


@dataclass
class VibronicSolution:
    """Diagonalized single-molecule vibronic basis for one molecule.

    Attributes
    ----------
    Nv, eps_k : run parameters used to build the Hamiltonian.
    basis, index : primitive single-molecule basis and its lookup.
    U : (n_prim, n_eig) matrix; column ``a`` is vibronic eigenvector
        ``|lambda_a>`` in the primitive basis (``U[p, a] = C``-coefficient).
    evals : (n_eig,) vibronic eigenenergies ``lambda_a`` (includes ``eps_k``).
    v : (n_eig,) integer vibronic sector of each eigenvector.
    is_excited : (n_eig,) bool; True for non-trivial (E-type, one electronic
        excitation, odd v) eigenvectors, False for trivial (A-type) ones.
    """

    Nv: int
    eps_k: float
    basis: List[SingleState]
    index: Dict[SingleState, int]
    U: np.ndarray
    evals: np.ndarray
    v: np.ndarray
    is_excited: np.ndarray

    # ---- convenience views used when building the polaritonic Hamiltonian ----
    def nex(self) -> np.ndarray:
        """Electronic excitation number (0 for A-type, 1 for E-type)."""
        return self.is_excited.astype(int)

    def ground_index(self) -> int:
        """Index of the |A, 0, 0> vibronic ground state (v = 0, lowest energy)."""
        a_mask = ~self.is_excited
        cand = np.where(a_mask & (self.v == 0))[0]
        return int(cand[np.argmin(self.evals[cand])])


def _build_block(states: List[SingleState], eps_k: float, cfg: Config) -> np.ndarray:
    """Dense single-molecule Hamiltonian on a contiguous electronic block.

    Implements Eq. (1): vibrational energy ``omega (n+ + n-)``, electronic energy
    ``eps_k`` when excited, and the linear E x e Jahn-Teller coupling
    ``kappa [ (b+^dag + b-) |E-><E+| + h.c. ]``.
    """
    idx = {st: i for i, st in enumerate(states)}
    dim = len(states)
    H = np.zeros((dim, dim))
    omega, kappa, Nv = cfg.omega, cfg.kappa, cfg.heatmap_nv

    for i, (s, npv, nmv) in enumerate(states):
        # Diagonal: vibrational + electronic self-energy.
        H[i, i] += omega * (npv + nmv)
        if s in ("Ep", "Em"):
            H[i, i] += eps_k

        # Jahn-Teller coupling kappa (b+^dag + b-) |E-><E+| + h.c.
        # E+ -> E- with b+^dag  (raise n+)
        if s == "Ep" and npv < Nv - 1:
            j = idx[("Em", npv + 1, nmv)]
            H[j, i] += kappa * np.sqrt(npv + 1)
        # E+ -> E- with b-      (lower n-)
        if s == "Ep" and nmv > 0:
            j = idx[("Em", npv, nmv - 1)]
            H[j, i] += kappa * np.sqrt(nmv)
        # E- -> E+ with b+      (lower n+)   [= h.c. of the b+^dag term]
        if s == "Em" and npv > 0:
            j = idx[("Ep", npv - 1, nmv)]
            H[j, i] += kappa * np.sqrt(npv)
        # E- -> E+ with b-^dag  (raise n-)   [= h.c. of the b- term]
        if s == "Em" and nmv < Nv - 1:
            j = idx[("Ep", npv, nmv + 1)]
            H[j, i] += kappa * np.sqrt(nmv + 1)

    return H


def diagonalize_single_molecule(eps_k: float, cfg: Config) -> VibronicSolution:
    """Build and diagonalize one molecule's JT Hamiltonian; return its vibronic basis.

    ``H_{m,k}`` is block-diagonal in (electronic class, vibronic sector v): the A
    (ground) states never couple to the E (excited) states, and the JT coupling
    only mixes E states of the *same* v. We therefore diagonalize each
    ``(class, v)`` block separately. This guarantees that every returned
    eigenvector has a definite v -- even when eigenstates from different sectors
    are accidentally degenerate, which would otherwise let a full-matrix
    diagonalizer return v-mixed combinations.
    """
    Nv = cfg.heatmap_nv
    basis, index = build_single_molecule_basis(Nv)
    n_prim = len(basis)

    # Partition primitive states into (class, v) blocks. class 'A' -> ground
    # (is_excited False), 'E' -> excited (is_excited True).
    blocks: Dict[Tuple[str, int], List[SingleState]] = {}
    for st in basis:
        cls = "A" if st[0] == "A" else "E"
        blocks.setdefault((cls, single_state_v(st)), []).append(st)

    U = np.zeros((n_prim, n_prim))
    evals = np.empty(n_prim)
    v_arr = np.empty(n_prim, dtype=int)
    is_excited = np.zeros(n_prim, dtype=bool)

    col = 0
    for (cls, v), states in blocks.items():
        Hb = _build_block(states, eps_k, cfg)
        w, vecs = np.linalg.eigh(Hb)
        rows = np.array([index[st] for st in states])
        m = len(states)
        U[rows[:, None], np.arange(col, col + m)] = vecs
        evals[col:col + m] = w
        v_arr[col:col + m] = v
        is_excited[col:col + m] = (cls == "E")
        col += m

    assert col == n_prim, "block partition did not cover the full basis"

    return VibronicSolution(
        Nv=Nv, eps_k=eps_k, basis=basis, index=index,
        U=U, evals=evals, v=v_arr, is_excited=is_excited,
    )


def diagonalize_single_molecule_reference(cfg: Config) -> VibronicSolution:
    """Baseline vibronic solution at ``eps_k = 0``, meant to be diagonalized once.

    ``eps_k`` enters the excited-electronic block only as a uniform diagonal
    shift ``eps_k * I`` (see module docstring), so this reference's eigenvectors,
    vibronic sector labels ``v``, and excited-state mask are shared by *every*
    ``eps_k`` -- only the excited-state eigenvalues need to shift. Use
    :func:`shift_reference` to build the molecule- and realization-specific
    solution from this reference in O(1) instead of re-diagonalizing.
    """
    return diagonalize_single_molecule(0.0, cfg)


def shift_reference(reference: VibronicSolution, eps_k: float) -> VibronicSolution:
    """O(1) vibronic solution at electronic energy ``eps_k``, reusing ``reference``.

    ``reference`` must be built at ``eps_k = 0`` (i.e. via
    :func:`diagonalize_single_molecule_reference`). Its basis, transformation
    matrix ``U``, sector labels ``v``, and excited-state mask are reused
    unchanged; only the excited-state eigenvalues shift rigidly by ``eps_k``.
    This is algebraically exact, not an approximation: for any operator
    restricted to the excited block, ``(H_block + eps_k I) u = (lambda + eps_k) u``
    whenever ``H_block u = lambda u``, so ``reference.U``'s columns remain exact
    eigenvectors at the new ``eps_k`` with correspondingly shifted eigenvalues,
    regardless of degeneracies.
    """
    evals = reference.evals.copy()
    evals[reference.is_excited] += eps_k
    return VibronicSolution(
        Nv=reference.Nv, eps_k=eps_k, basis=reference.basis, index=reference.index,
        U=reference.U, evals=evals, v=reference.v, is_excited=reference.is_excited,
    )


def alpha_beta(vib: VibronicSolution) -> Tuple[np.ndarray, np.ndarray]:
    """Vibronic-basis matter-cavity coupling matrices (SM Sec. I.A).

    Returns ``(alpha, beta)`` with

        alpha[i, j] = sum_{n+,n-} U*_{A,n+,n-,i} U_{E+,n+,n-,j}
        beta[i, j]  = sum_{n+,n-} U*_{A,n+,n-,i} U_{E-,n+,n-,j}

    i.e. ``alpha = <lambda_i| (|A><E+|) |lambda_j>`` and likewise ``beta`` for
    ``|A><E-|``. These are the vibronic-basis matrix elements of the primitive
    matter-cavity operator in Eq. (3): the ``+`` cavity mode couples to ``E+``
    (via ``alpha``) and the ``-`` mode to ``E-`` (via ``beta``). ``alpha`` is
    nonzero only for an A-type row ``i`` and an E-type column ``j`` with
    ``v_i = v_j - 1``; ``beta`` only for ``v_i = v_j + 1`` -- which is exactly the
    ``j`` (total angular momentum) conservation of the coupling.
    """
    Nv = vib.Nv
    # Row (n+,n-) blocks of U for each electronic label, in a shared (n+,n-) order.
    nn = [(npv, nmv) for npv in range(Nv) for nmv in range(Nv)]
    rowsA = np.array([vib.index[("A", npv, nmv)] for npv, nmv in nn])
    rowsEp = np.array([vib.index[("Ep", npv, nmv)] for npv, nmv in nn])
    rowsEm = np.array([vib.index[("Em", npv, nmv)] for npv, nmv in nn])

    P_A = vib.U[rowsA, :]    # (Nv^2, n_eig)
    P_Ep = vib.U[rowsEp, :]  # (Nv^2, n_eig)
    P_Em = vib.U[rowsEm, :]

    alpha = P_A.T @ P_Ep     # (n_eig, n_eig)
    beta = P_A.T @ P_Em
    return alpha, beta
