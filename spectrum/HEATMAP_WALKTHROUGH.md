# P(v) heatmap workflow — line-by-line walkthrough

This document explains, line by line, every module that makes up the
`--representative-heatmap` workflow: how a single molecule's Jahn–Teller basis
is built, how two molecules + a cavity photon are combined into the
polaritonic Hamiltonian, how the single-molecule vibronic-sector population
`P(v)` is extracted from a polaritonic eigenstate, and how all of that is
wired into the CLI. It complements
[`SPECTRUM_WALKTHROUGH.md`](SPECTRUM_WALKTHROUGH.md) (the absorption pipeline)
and [`REPRESENTATIVE_ANALYSIS.md`](../REPRESENTATIVE_ANALYSIS.md)-style docs for
the Case A–K selection; here the focus is the **new physics** (vibronic basis,
polaritonic Hamiltonian, P(v)) and the **driver** that ties it to the CLI.

Five files are covered, in the order data flows through them:

1. [`spectrum/vibronic.py`](vibronic.py) — one molecule's Jahn–Teller vibronic basis.
2. [`spectrum/polariton.py`](polariton.py) — two molecules + cavity photon → polaritonic Hamiltonian.
3. [`spectrum/participation.py`](participation.py) — `P(v)`, participation ratio, the Fig. S1 heatmap.
4. [`spectrum/representative_heatmap.py`](representative_heatmap.py) — the driver: selection, parallel solve, plotting, storage.
5. [`main.py`](../main.py) — the `--representative-heatmap` CLI wiring.

Nothing in this workflow imports or mutates the frozen absorption-spectrum
modules (`basis.py`, `operators.py`, `hamiltonian.py`, `spectrum.py`); they are
untouched. The two pipelines are validated to agree (see §6).

---

## 1. `spectrum/vibronic.py` — one molecule's vibronic basis

### 1.1 The physics being encoded (module docstring, lines 1–31)

A single JT-active molecule's Hamiltonian (Eq. 1 of the paper) is

```
H_{m,k} = omega (b+^dag b+ + b-^dag b-) + eps_k |E><E|
        + kappa [ (b+^dag + b-) |E-><E+| + h.c. ]
```

Diagonalizing it gives **vibronic eigenstates** `|lambda^v_j>`, each carrying a
well-defined **vibronic angular momentum** quantum number

```
v = 2(n+ - n-) + Sz ,   Sz(E+)=+1, Sz(E-)=-1, Sz(A)=0
```

The key fact this whole module exploits: `H_{m,k}` has **no coupling between
the electronic ground block** (`|A, n+, n-⟩`, "trivial" eigenvectors, even
`v`, no excitation) **and the excited block** (superpositions of
`|E±, n+, n-⟩`, "non-trivial", odd `v`). The electronic energy `eps_k` enters
only as a uniform shift `eps_k · I` on the excited block. Consequently:

- the vibronic **eigenvectors** don't depend on `eps_k` (hence not on
  disorder — disorder only changes `eps_k`);
- only the excited-block **eigenvalues** shift, rigidly, by `+eps_k`.

This is why the module offers a "diagonalize once, shift for free"
API (`diagonalize_single_molecule_reference` + `shift_reference`,
§1.6–1.7) instead of one full diagonalization per molecule per realization.

### 1.2 Basis bookkeeping (lines 33–65)

```python
SingleState = Tuple[str, int, int]
ELECTRONIC = ("A", "Ep", "Em")
_SZ = {"A": 0, "Ep": +1, "Em": -1}

def single_state_v(state: SingleState) -> int:
    s, npv, nmv = state
    return 2 * (npv - nmv) + _SZ[s]

def build_single_molecule_basis(Nv: int) -> Tuple[List[SingleState], Dict[SingleState, int]]:
    basis: List[SingleState] = []
    for s in ELECTRONIC:
        for npv, nmv in product(range(Nv), range(Nv)):
            basis.append((s, npv, nmv))
    index = {st: i for i, st in enumerate(basis)}
    return basis, index
```

- A single-molecule **primitive** state is `(s, n+, n-)`: electronic label `s`
  (`"A"`, `"Ep"`, `"Em"`) and vibrational quanta `n+, n-` in `[0, Nv)`.
- `single_state_v` computes that state's `v` label directly from the formula
  above — used to bucket states by sector before diagonalizing (§1.5).
- `build_single_molecule_basis` enumerates every `(s, n+, n-)` combination,
  **electronic-block-first** (all `"A"` states, then all `"Ep"`, then all
  `"Em"`) so the ground and excited blocks are contiguous ranges — convenient
  for the block-diagonalization in §1.5, though not required by it (blocks are
  looked up by `(class, v)`, not by contiguous slicing).

### 1.3 `VibronicSolution` — the per-molecule result (lines 68–102)

```python
@dataclass
class VibronicSolution:
    Nv: int
    eps_k: float
    basis: List[SingleState]
    index: Dict[SingleState, int]
    U: np.ndarray
    evals: np.ndarray
    v: np.ndarray
    is_excited: np.ndarray

    def nex(self) -> np.ndarray:
        return self.is_excited.astype(int)

    def ground_index(self) -> int:
        a_mask = ~self.is_excited
        cand = np.where(a_mask & (self.v == 0))[0]
        return int(cand[np.argmin(self.evals[cand])])
```

- `U` is the **change-of-basis matrix**: column `a` is vibronic eigenvector
  `|lambda_a⟩` expressed in the primitive `(s, n+, n-)` basis, so
  `U[p, a]` is the coefficient of primitive state `p` in eigenvector `a`.
- `evals[a]` is that eigenvector's energy (already includes `eps_k`).
- `v[a]` is its vibronic sector label; `is_excited[a]` is `True` for
  excited-block ("E-type") eigenvectors.
- `nex()` is just `is_excited` as 0/1 — the electronic excitation number,
  used later when enumerating the two-molecule `(nex=1, j=-1)` sector.
- `ground_index()` finds the `|A, 0, 0⟩` vibronic ground state: filter to
  ground-block (`~is_excited`), `v == 0` eigenvectors, and take the
  lowest-energy one among them. This is the initial-state anchor used by
  `polariton.initial_state_index` (§2.5).

### 1.4 `_build_block` — the dense Hamiltonian on one `(class, v)` block (lines 105–141)

```python
def _build_block(states: List[SingleState], eps_k: float, cfg: Config) -> np.ndarray:
    idx = {st: i for i, st in enumerate(states)}
    dim = len(states)
    H = np.zeros((dim, dim))
    omega, kappa, Nv = cfg.omega, cfg.kappa, cfg.heatmap_nv

    for i, (s, npv, nmv) in enumerate(states):
        H[i, i] += omega * (npv + nmv)
        if s in ("Ep", "Em"):
            H[i, i] += eps_k

        if s == "Ep" and npv < Nv - 1:
            j = idx[("Em", npv + 1, nmv)]
            H[j, i] += kappa * np.sqrt(npv + 1)
        if s == "Ep" and nmv > 0:
            j = idx[("Em", npv, nmv - 1)]
            H[j, i] += kappa * np.sqrt(nmv)
        if s == "Em" and npv > 0:
            j = idx[("Ep", npv - 1, nmv)]
            H[j, i] += kappa * np.sqrt(npv)
        if s == "Em" and nmv < Nv - 1:
            j = idx[("Ep", npv, nmv + 1)]
            H[j, i] += kappa * np.sqrt(nmv + 1)

    return H
```

Builds a **dense Hamiltonian restricted to one already-identified block** of
states (all sharing the same electronic class and `v`; see §1.5 for how
blocks are formed).

- **Diagonal**: `omega * (n+ + n-)` (vibrational energy) plus `eps_k` if the
  state is electronically excited (`"Ep"`/`"Em"`).
- **Off-diagonal (Jahn–Teller coupling)**: implements
  `kappa[(b+^dag + b-)|E-⟩⟨E+| + h.c.]` term by term:
  - `Ep --b+^dag--> Em`: raises `n+` by one, amplitude `kappa·√(n++1)`
    (bosonic raising-operator matrix element), only valid if `n+ < Nv-1`
    (stays inside the truncated Fock space).
  - `Ep --b---> Em`: lowers `n-` by one, amplitude `kappa·√(n-)`, valid if
    `n- > 0`.
  - The remaining two `if` blocks are the Hermitian-conjugate transitions
    `Em → Ep` (lower `n+` / raise `n-`), with matching square-root amplitudes
    — so the resulting matrix is exactly Hermitian by construction (each pair
    of `if`s writes both `H[j,i]` and, via its conjugate branch, `H[i,j]`).

Because states here all share one `(class, v)` block (enforced by the caller,
§1.5), every generated coupling `(i → j)` automatically stays within the same
block: the JT ladder operators change `n+`/`n-` by exactly one quantum while
flipping `Ep ↔ Em` (which flips `Sz` by 2), so `v = 2(n+-n-) + Sz` is
conserved — this is *why* the Hamiltonian block-diagonalizes by `v` in the
first place, and `_build_block` simply builds one such block.

### 1.5 `diagonalize_single_molecule` — block-by-block diagonalization (lines 144–188)

```python
def diagonalize_single_molecule(eps_k: float, cfg: Config) -> VibronicSolution:
    Nv = cfg.heatmap_nv
    basis, index = build_single_molecule_basis(Nv)
    n_prim = len(basis)

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
```

1. **Partition** the full primitive basis into `(class, v)` groups — `class`
   is `"A"` for ground states, `"E"` for either excited label (`single_state_v`
   already folds the `Sz` distinction into `v`, so `"Ep"` and `"Em"` states of
   the same `v` end up together and *do* mix under the JT coupling — see §1.4).
2. For each block: build its (small) dense Hamiltonian (`_build_block`) and
   diagonalize it with `np.linalg.eigh` — **many small diagonalizations
   instead of one big one**.
3. **Scatter the results back** into the full-size arrays: `rows` maps this
   block's local state indices to their position in the *global* primitive
   basis; `U[rows[:, None], np.arange(col, col+m)] = vecs` writes this block's
   eigenvectors into the corresponding rows and a contiguous run of `m`
   columns (`col` is a running column offset across blocks). `evals`, `v_arr`,
   `is_excited` are filled in lockstep.
4. The `assert` is a coverage check: every primitive state must land in
   exactly one block, so summing block sizes must reach `n_prim`.

**Why block-by-block, not one big `eigh`?** The docstring's stated reason:
this *guarantees* every returned eigenvector has a definite `v`, even under
accidental degeneracy across sectors — a single full-matrix `eigh` could
return an eigenvector that's an arbitrary linear combination of two
degenerate states from *different* blocks, which would break the `v`
bookkeeping everything downstream depends on. Diagonalizing pre-separated
blocks makes that structurally impossible.

### 1.6–1.7 The exact O(1) shortcut: reference + shift (lines 191–222)

```python
def diagonalize_single_molecule_reference(cfg: Config) -> VibronicSolution:
    return diagonalize_single_molecule(0.0, cfg)


def shift_reference(reference: VibronicSolution, eps_k: float) -> VibronicSolution:
    evals = reference.evals.copy()
    evals[reference.is_excited] += eps_k
    return VibronicSolution(
        Nv=reference.Nv, eps_k=eps_k, basis=reference.basis, index=reference.index,
        U=reference.U, evals=evals, v=reference.v, is_excited=reference.is_excited,
    )
```

- `diagonalize_single_molecule_reference` is just
  `diagonalize_single_molecule(0.0, cfg)` — the baseline solve at `eps_k = 0`,
  meant to be computed **once**.
- `shift_reference` builds the solution at any other `eps_k` **without
  touching `eigh` at all**: it copies `reference.evals` and adds `eps_k` only
  to the excited-block entries (`reference.is_excited` masks which rows get
  the shift); `basis`, `index`, `U`, and `v` are reused unchanged (not even
  copied — same array objects).

This is **exact, not an approximation** (as the docstring argues): for any
operator restricted to the excited block, adding `eps_k · I` to that block
shifts every eigenvalue of the block by `eps_k` while leaving its eigenvectors
completely unchanged — a basic fact about how a matrix responds to adding a
multiple of the identity, and it holds *regardless of degeneracies* within the
block. So `reference.U`'s columns remain exact eigenvectors at the new
`eps_k`, just with a relabeled eigenvalue. This is the mechanism that lets
the workflow solve many molecules/realizations without paying for repeated
single-molecule diagonalizations (each one is fast anyway — this is a modest,
not dominant, cost saving; see §6's "what's expensive" note).

### 1.8 `alpha_beta` — vibronic-basis matter-cavity coupling matrices (lines 225–254)

```python
def alpha_beta(vib: VibronicSolution) -> Tuple[np.ndarray, np.ndarray]:
    Nv = vib.Nv
    nn = [(npv, nmv) for npv in range(Nv) for nmv in range(Nv)]
    rowsA = np.array([vib.index[("A", npv, nmv)] for npv, nmv in nn])
    rowsEp = np.array([vib.index[("Ep", npv, nmv)] for npv, nmv in nn])
    rowsEm = np.array([vib.index[("Em", npv, nmv)] for npv, nmv in nn])

    P_A = vib.U[rowsA, :]
    P_Ep = vib.U[rowsEp, :]
    P_Em = vib.U[rowsEm, :]

    alpha = P_A.T @ P_Ep
    beta = P_A.T @ P_Em
    return alpha, beta
```

This computes the vibronic-basis matrix elements of the primitive
matter-cavity coupling operator (`|A⟩⟨E+|` and `|A⟩⟨E-|`), needed to build the
*two-molecule* polaritonic Hamiltonian in `polariton.py`:

1. `nn` is every `(n+, n-)` pair, in a fixed shared order.
2. `rowsA`/`rowsEp`/`rowsEm` are the primitive-basis row indices of, respectively,
   every `("A", n+, n-)`, `("Ep", n+, n-)`, `("Em", n+, n-)` state, in that
   same `nn` order.
3. `P_A = vib.U[rowsA, :]` extracts, from the full change-of-basis matrix
   `U`, just the rows belonging to `"A"`-type primitive states — i.e. the
   projection of every vibronic eigenvector onto the ground-electronic
   subspace, still indexed by vibrational quanta `(n+, n-)` (via the `nn`
   ordering) in its rows and by eigenvector index in its columns. Likewise for
   `P_Ep`/`P_Em`.
4. `alpha = P_A.T @ P_Ep` is then exactly
   `alpha[i,j] = Σ_{n+,n-} U*_{A,n+,n-,i} · U_{Ep,n+,n-,j}` — the docstring's
   formula — computed as one matrix product instead of an explicit double
   sum, because both `P_A` and `P_Ep` share the same row ordering (`nn`), so
   `P_A.T @ P_Ep` sums exactly over that shared `(n+, n-)` index. `beta` is
   the analogous product with `P_Em`.

Physically: the `+` cavity photon mode couples `A ↔ E+` (this is what
`alpha` encodes in the vibronic basis), and the `-` mode couples `A ↔ E-`
(`beta`). `alpha`/`beta` are non-zero only between an `A`-type row and an
`E`-type column whose `v` differ by exactly 1 — the conservation law that
`polariton.py` exploits directly rather than re-deriving.

---

## 2. `spectrum/polariton.py` — two molecules + a cavity photon

### 2.1 The Hamiltonian being built (module docstring, lines 1–31)

Given both molecules' `VibronicSolution`s, this module builds and diagonalizes
Eq. (S1):

```
H' = sum_{a,k} lambda_a |lambda_a>_k<lambda_a|_k
   + omega_c (a+^dag a+ + a-^dag a-)
   + g sum_{a,b,k} ( alpha_{a,b} |lambda_a>_k<lambda_b|_k a+^dag
                   + beta_{a,b}  |lambda_a>_k<lambda_b|_k a-^dag + h.c. )
```

restricted to the same `(nex=1, j=-1)` sector the absorption pipeline uses,
but now expressed in the **vibronic** basis (molecule-1 eigenstate index `a`,
molecule-2 eigenstate index `b`, photon label `p`) instead of the primitive
8-tuple basis.

Because molecules 1 and 2 can have different `eps1`/`eps2` under disorder,
they generally have *different* vibronic bases — this module builds the
**distinguishable-molecule** sector `(a, b, p)` rather than assuming
permutation symmetry, which is the form disorder demands. At `sigma=0` (both
molecules identical) this construction reproduces the paper's
permutation-symmetric result exactly (validated in §6).

### 2.2 `PolaritonSector` — the sector basis container (lines 45–59)

```python
@dataclass
class PolaritonSector:
    states: List[SectorState]
    index: Dict[SectorState, int]
    a: np.ndarray
    b: np.ndarray
    p: np.ndarray
    v1: np.ndarray
    v2: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.states)
```

Each basis state is `(a, b, p)`: molecule-1 vibronic eigenstate index `a`,
molecule-2 index `b`, photon label `p ∈ {+1, 0, -1}` (`+1`/`-1` one photon in
the `+`/`-` mode, `0` no photon). `v1`/`v2` cache each state's molecule-1/2
vibronic sector (`vib1.v[a]` / `vib2.v[b]`) for convenient vectorized lookups
later (used heavily in `participation.py`).

### 2.3 `_group_by_v` — a small indexing helper (lines 62–67)

```python
def _group_by_v(indices: np.ndarray, v: np.ndarray) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for i in indices:
        out.setdefault(int(v[i]), []).append(int(i))
    return out
```

Given a set of eigenstate indices and their `v` labels, groups them into
`{v_value: [indices with that v]}`. Used repeatedly below to answer "which
eigenstates of this molecule have sector `v`?" in O(1) after one grouping
pass, instead of rescanning every time.

### 2.4 `build_sector` — enumerating the `(nex=1, j=-1)` basis (lines 70–128)

```python
def build_sector(vib1, vib2, cfg) -> PolaritonSector:
    nex_target = cfg.nex_target
    j_target = cfg.jz_target
    if nex_target != 1:
        raise NotImplementedError(...)

    A1 = np.where(~vib1.is_excited)[0]
    E1 = np.where(vib1.is_excited)[0]
    A2 = np.where(~vib2.is_excited)[0]
    E2 = np.where(vib2.is_excited)[0]

    A1_by_v = _group_by_v(A1, vib1.v)
    A2_by_v = _group_by_v(A2, vib2.v)

    states: List[SectorState] = []

    for p in (+1, -1):
        need = j_target - p
        for va, a_list in A1_by_v.items():
            vb = need - va
            for a in a_list:
                for b in A2_by_v.get(vb, ()):
                    states.append((a, b, p))

    E1_by_v = _group_by_v(E1, vib1.v)
    E2_by_v = _group_by_v(E2, vib2.v)
    for va, a_list in E1_by_v.items():
        vb = j_target - va
        for a in a_list:
            for b in A2_by_v.get(vb, ()):
                states.append((a, b, 0))
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
```

Rather than enumerating *every* `(a, b, p)` triple and filtering by
`nex`/`j` (which would be `O(n_eig² × 3)` with mostly-rejected candidates),
this builds the sector **directly from the two conservation laws** the
absorption pipeline's `basis.py` also encodes, just now applied to vibronic
eigenstate indices instead of primitive quantum numbers:

- `nex = 1`: exactly one "excitation" (one electronic excitation *or* one
  photon) is present in the whole system.
- `j = v1 + v2 + p` is fixed at `cfg.jz_target`.

**Case 1 — one photon present (`p = ±1`), lines 97–104.** Then neither
molecule can be electronically excited (that would be a second excitation), so
both must be `A`-type. `need = j_target - p` is the required `v_a + v_b`. The
nested loop: for every molecule-1 `A`-type sector `va` (from `A1_by_v`), the
matching molecule-2 sector is `vb = need - va`; every `(a, b)` pair drawn from
those two sector buckets is a valid sector state `(a, b, p)`.

**Case 2 — no photon (`p = 0`), lines 106–120.** Then exactly one molecule
must be `E`-type (the excitation is electronic) and the other `A`-type. Two
sub-cases, symmetric: "molecule 1 excited, molecule 2 ground" (lines
110–114) and "molecule 1 ground, molecule 2 excited" (lines 116–120) — each
pairs an `E`-type sector of one molecule with the `A`-type sector of the
other that makes `v_a + v_b = j_target`.

**Assembly (lines 122–128).** `index` is the state→position lookup (built
once the full list is known, needed by `build_hamiltonian` to look up target
states by their `(a,b,p)` key). `a_arr`/`b_arr`/`p_arr` are vectorized views
of the three components (used throughout for array-based computation instead
of Python loops). `v1`/`v2` are precomputed by advanced-indexing
`vib1.v[a_arr]`/`vib2.v[b_arr]` — one array lookup instead of `dim` individual
ones.

### 2.5 `build_hamiltonian` — assembling `H'` in the sector basis (lines 131–188)

```python
def build_hamiltonian(vib1, vib2, sector, cfg) -> np.ndarray:
    dim = sector.dim
    H = np.zeros((dim, dim))

    photon_number = (sector.p != 0).astype(float)
    H[np.arange(dim), np.arange(dim)] = (
        vib1.evals[sector.a] + vib2.evals[sector.b] + cfg.omega_c * photon_number
    )

    g = cfg.g
    alpha1, beta1 = alpha_beta(vib1)
    alpha2, beta2 = alpha_beta(vib2)

    A1_by_v = _group_by_v(np.where(~vib1.is_excited)[0], vib1.v)
    A2_by_v = _group_by_v(np.where(~vib2.is_excited)[0], vib2.v)

    def add(target, source_i, m):
        if m == 0.0:
            return
        j = sector.index.get(target)
        if j is None:
            return
        H[j, source_i] += m
        H[source_i, j] += m

    for i, (a, b, p) in enumerate(sector.states):
        if p != 0:
            continue
        if vib1.is_excited[a]:
            va = int(vib1.v[a])
            for ap in A1_by_v.get(va - 1, ()):
                add((ap, b, +1), i, g * alpha1[ap, a])
            for ap in A1_by_v.get(va + 1, ()):
                add((ap, b, -1), i, g * beta1[ap, a])
        if vib2.is_excited[b]:
            vb = int(vib2.v[b])
            for bp in A2_by_v.get(vb - 1, ()):
                add((a, bp, +1), i, g * alpha2[bp, b])
            for bp in A2_by_v.get(vb + 1, ()):
                add((a, bp, -1), i, g * beta2[bp, b])

    return H
```

**Diagonal (lines 147–151).** Every basis state `(a, b, p)`'s bare energy is
`lambda_a^{(1)} + lambda_b^{(2)}` (both molecules' vibronic eigenenergies —
`vib1.evals[sector.a]` / `vib2.evals[sector.b]`, vectorized over every sector
state at once via `sector.a`/`sector.b`) plus `omega_c` if a photon is present
(`photon_number = (sector.p != 0)`). This is set via fancy-indexed diagonal
assignment in one line — no Python loop.

**Off-diagonal / matter-cavity coupling (lines 153–186).** `alpha_beta` is
called once per molecule to get the vibronic-basis coupling matrices (§1.8).
`A1_by_v`/`A2_by_v` group each molecule's ground-block eigenstates by sector,
so "find the `A`-type eigenstate(s) of molecule 1 with sector `va-1`" is a
dict lookup.

The `add` closure (lines 161–168) adds a **symmetric** off-diagonal matrix
element: given a `target` state key and a coupling amplitude `m` from
`source_i`, it looks up `target`'s index in the sector (`sector.index.get`,
returning `None`, silently skipped, if that target isn't in this
sector — occurs when the paired state would violate `j` or `nex`, though by
construction from §2.4 conservation the lookups here should always succeed),
and adds `m` to *both* `H[j, source_i]` and `H[source_i, j]` — enforcing
Hermiticity directly instead of building one triangle and symmetrizing after.

The main loop (lines 170–186) iterates every sector state, but **only acts
on `p == 0` (matter) states** — couplings are generated once, from the matter
side, rather than from both a matter state and its two photon partners (which
would double-count and require careful de-duplication). For a matter state
where molecule 1 is excited (`vib1.is_excited[a]`):
- de-exciting molecule 1 to an `A`-type eigenstate `ap` with `v_ap = va - 1`
  **and emitting a `+` photon** — the `alpha` (A↔E+) coupling, amplitude
  `g · alpha1[ap, a]`; target state `(ap, b, +1)`.
- de-exciting to `v_ap = va + 1` **and emitting a `-` photon** — the `beta`
  (A↔E-) coupling, amplitude `g · beta1[ap, a]`; target `(ap, b, -1)`.

The analogous two lines apply if molecule 2 is excited instead. This exactly
mirrors the `v`-conservation rule stated in `alpha_beta`'s docstring
(§1.8): the `+` mode connects `v_a` to `v_a - 1`, the `-` mode to `v_a + 1`.

### 2.6 `initial_state_index` — locating the absorption starting point (lines 191–202)

```python
def initial_state_index(vib1, vib2, sector, cfg) -> int:
    a0 = vib1.ground_index()
    b0 = vib2.ground_index()
    key = (a0, b0, -1)
    if key not in sector.index:
        raise ValueError(...)
    return sector.index[key]
```

The absorption initial state is: both molecules in their vibronic ground
state (`ground_index()`, §1.3) and **one photon in the RCP (`-`) mode** —
`p = -1`, matching the primitive pipeline's `cm=1` convention
(`('A',0,0,'A',0,0,0,1)`). Looks up that `(a0, b0, -1)` triple's position in
the sector; raises if it's somehow missing (a configuration bug, not a normal
runtime path).

### 2.7 `PolaritonSolution` and `solve` — putting it together (lines 205–223)

```python
@dataclass
class PolaritonSolution:
    sector: PolaritonSector
    evals: np.ndarray
    evecs: np.ndarray
    i0: int
    intensity: np.ndarray


def solve(vib1, vib2, cfg) -> PolaritonSolution:
    sector = build_sector(vib1, vib2, cfg)
    H = build_hamiltonian(vib1, vib2, sector, cfg)
    evals, evecs = np.linalg.eigh(H)
    i0 = initial_state_index(vib1, vib2, sector, cfg)
    intensity = np.abs(evecs[i0, :]) ** 2
    return PolaritonSolution(sector, evals, evecs, i0, intensity)
```

`solve` is the **one-shot entry point**: build the sector (§2.4), build the
Hamiltonian in it (§2.5), diagonalize (**the one expensive step — a dense
`eigh` on a `dim × dim` matrix, same order of cost as the absorption
pipeline's own `eigh`**), locate the initial state (§2.6), and compute
absorption intensities exactly as the primitive pipeline does:
`|⟨i0|eigenstate⟩|² = |evecs[i0, :]|²`. Crucially, **`evecs` (the full
eigenvector matrix) is kept** in `PolaritonSolution` — unlike the absorption
pipeline, which discards it after extracting one row — because `P(v)` (next
section) needs every eigenvector's full expansion, not just its overlap with
the initial state.

---

## 3. `spectrum/participation.py` — P(v) and the Fig. S1 heatmap

### 3.1 The formula being computed (module docstring, lines 1–24)

For a polaritonic eigenstate `|ψ⟩ = Σ_{a,b,p} C_{a,b,p} |a,b,p⟩`, the
probability that **molecule 1** occupies vibronic sector `v0` is

```
P^(1)(v0) = sum_{a: v_a=v0} sum_{b,p} |C_{a,b,p}|^2
```

— sum the eigenvector's squared amplitude over every basis state whose
molecule-1 label falls in sector `v0`. This is the *distinguishable-molecule*
form of the paper's Eq. (S10); at `sigma=0` it reduces to the paper's
symmetric formula exactly (validated, §6). `P^(2)(v0)` is the analogous sum
over molecule 2. Since every basis state assigns each molecule to exactly one
sector, `Σ_v P(v) = 1` for every eigenstate, and the **participation ratio**
`PR = 1/Σ_v P(v)²` measures how many sectors meaningfully participate (PR=1
if fully localized in one sector; larger if spread across many).

### 3.2 `sector_grid` — the y-axis (lines 35–39)

```python
def sector_grid(sector: PolaritonSector) -> np.ndarray:
    vmin = int(min(sector.v1.min(), sector.v2.min()))
    vmax = int(max(sector.v1.max(), sector.v2.max()))
    return np.arange(vmin, vmax + 1)
```

The contiguous integer range of every `v` value that appears for *either*
molecule across the whole sector — this becomes the heatmap's y-axis (every
integer sector from the most negative to the most positive present, even ones
with zero population, so the axis is evenly spaced).

### 3.3 `_grouped_weight` — vectorized "sum by group" (lines 42–46)

```python
def _grouped_weight(weights: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.zeros((n_groups, weights.shape[1]))
    np.add.at(out, group_idx, weights)
    return out
```

Given a `(dim, n_eig)` array of per-basis-state weights and a `(dim,)` array
mapping each basis state to a group (here: a shifted vibronic-sector index),
`np.add.at` scatter-adds every row of `weights` into `out[group_idx[i]] +=
weights[i]` — this is exactly the `Σ_{a: v_a=v0}` sum in the P(v) formula,
done for every eigenstate (column) simultaneously via one vectorized call
instead of a Python double loop over eigenstates and basis states.

### 3.4 `compute_Pv` — assembling `P(v)` for every eigenstate (lines 49–83)

```python
def compute_Pv(sol, cfg, grid=None):
    sector = sol.sector
    if grid is None:
        grid = sector_grid(sector)
    vmin = int(grid[0])
    n_groups = len(grid)

    weights = np.abs(sol.evecs) ** 2
    g1 = (sector.v1 - vmin).astype(int)
    g2 = (sector.v2 - vmin).astype(int)

    which = str(cfg.heatmap_which_molecule)
    if which == "1":
        Pv = _grouped_weight(weights, g1, n_groups)
    elif which == "2":
        Pv = _grouped_weight(weights, g2, n_groups)
    elif which == "avg":
        P1 = _grouped_weight(weights, g1, n_groups)
        P2 = _grouped_weight(weights, g2, n_groups)
        Pv = 0.5 * (P1 + P2)
    else:
        raise ValueError(...)
    return grid, Pv
```

1. `weights = |evecs|²` — the `(dim, n_eig)` matrix of every basis state's
   squared amplitude in every eigenstate; this is `|C_{a,b,p}|²` for every
   `(a,b,p)` row and every eigenstate column at once.
2. `g1 = sector.v1 - vmin` shifts each basis state's molecule-1 sector label
   into a **0-based group index** matching `grid`'s ordering (so `g1` can
   directly index into a `grid`-sized output array); likewise `g2`.
3. Depending on `cfg.heatmap_which_molecule` (`"1"`, `"2"`, or `"avg"`),
   `_grouped_weight` sums `weights` by `g1` (molecule 1's `P(v)`), by `g2`
   (molecule 2's), or both and averages — the `"avg"` case is the paper's
   symmetric quantity (identical to either molecule alone when `sigma=0`,
   since both molecules are then physically equivalent).
4. Returns `(grid, Pv)` where `Pv` has shape `(len(grid), n_eig)` and, by
   construction (every basis state contributes to exactly one sector-group
   for each molecule), **every column sums to 1** — verified numerically in
   the test suite (§6).

### 3.5 `participation_ratio` (lines 86–92)

```python
def participation_ratio(Pv: np.ndarray) -> np.ndarray:
    denom = np.sum(Pv ** 2, axis=0)
    pr = np.zeros_like(denom)
    nz = denom > 0
    pr[nz] = 1.0 / denom[nz]
    return pr
```

`PR = 1 / Σ_v P(v)²`, computed per eigenstate (each column of `Pv`). The
`nz` mask guards against division by zero for a (degenerate, shouldn't
normally occur) all-zero column; such columns get `PR = 0` instead of `inf`
or a runtime warning.

### 3.6 `HeatmapResult` and `bright_heatmap` — the discrete Fig. S1 (lines 95–129)

```python
@dataclass
class HeatmapResult:
    grid: np.ndarray
    bright_idx: np.ndarray
    energies: np.ndarray
    intensity: np.ndarray
    heatmap: np.ndarray
    pr: np.ndarray


def bright_heatmap(sol, cfg, grid=None) -> HeatmapResult:
    grid, Pv = compute_Pv(sol, cfg, grid)
    pr_all = participation_ratio(Pv)

    imax = sol.intensity.max()
    thresh = cfg.heatmap_bright_threshold * imax if imax > 0 else 0.0
    bright = np.where(sol.intensity > thresh)[0]
    order = bright[np.argsort(sol.evals[bright])]

    return HeatmapResult(
        grid=grid,
        bright_idx=order,
        energies=sol.evals[order],
        intensity=sol.intensity[order],
        heatmap=Pv[:, order],
        pr=pr_all[order],
    )
```

1. Get `P(v)` for **every** eigenstate first (`compute_Pv`) and its `PR`.
2. **Filter to "bright" states**: `thresh` is `heatmap_bright_threshold ×`
   the brightest eigenstate's intensity (a *relative*, not absolute,
   cutoff — default `1e-2`, i.e. 1% of the brightest peak); `bright` is every
   eigenstate index exceeding that. (If `imax == 0`, i.e. every eigenstate
   is dark, `thresh=0` and the `>` comparison correctly still yields no
   bright states.)
3. **Order by energy**: `order = bright[np.argsort(sol.evals[bright])]` —
   `argsort` on the *subset* of energies, then that ordering permutes the
   bright indices themselves, giving bright states sorted low-to-high energy.
4. Slice every quantity (`energies`, `intensity`, columns of `heatmap`, `pr`)
   by `order` — this produces the object plotted directly: `heatmap` has
   shape `(n_v, n_bright)`, one `P(v)` column per bright state, in energy
   order — exactly the x-axis of the paper's Fig. S1.

---

## 4. `spectrum/representative_heatmap.py` — the driver

### 4.1 Purpose (module docstring, lines 1–24)

This module is the glue between:
- the **Case A–K selection** in `spectrum/representative.py` (reused
  unmodified — it needs only each realization's drawn `(eps1, eps2)`, no
  absorption diagonalization), and
- the **vibronic → polaritonic → P(v)** machinery above,

to produce one Fig. S1 heatmap per selected representative realization. It
does exactly **one polaritonic diagonalization per selected case** (~11, not
the full 100–300-realization disorder cloud), and can spread those across
worker processes.

### 4.2 `solve_representative` — one `(eps1, eps2)` pair (lines 45–62)

```python
def solve_representative(cfg, eps1_dis, eps2_dis, reference=None):
    if reference is None:
        reference = vibronic.diagonalize_single_molecule_reference(cfg)
    vib1 = vibronic.shift_reference(reference, eps1_dis)
    vib2 = vibronic.shift_reference(reference, eps2_dis)
    return polariton.solve(vib1, vib2, cfg)
```

The full pipeline for one explicit disorder realization: get (or reuse) the
single-molecule reference solve, shift it to each molecule's actual site
energy (§1.7 — O(1), no `eigh`), then hand both vibronic solutions to
`polariton.solve` (§2.7) for the one real diagonalization. `reference` is an
optional pass-through so callers that solve *many* realizations (the parallel
path, §4.4) build it once and pass it in every time, rather than rebuilding it
per call.

### 4.3 `_heatmap_entry` — bundling a case with its result (lines 65–84)

```python
def _heatmap_entry(sel: Dict, hm: participation.HeatmapResult) -> Dict:
    m: RealizationMetadata = sel["metadata"]
    return {
        "case": sel["case"], "letter": sel["letter"], "description": sel["description"],
        "realization_index": m.index, "eps1": m.eps1, "eps2": m.eps2,
        "delta1": m.delta1, "delta2": m.delta2,
        "grid": hm.grid, "bright_idx": hm.bright_idx, "energies": hm.energies,
        "intensity": hm.intensity, "heatmap": hm.heatmap, "pr": hm.pr,
    }
```

A small assembly function: takes a `selections` entry (from
`representative.select_representative_realizations` — case name, description,
and the selected `RealizationMetadata`) and a computed `HeatmapResult`, and
flattens both into one plain dict — the unit that storage and plotting
downstream consume.

### 4.4 The parallel path (lines 87–148) — mirrors `spectrum.py`'s pattern

```python
_THREAD_ENV_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_WORKER: Dict = {}

def _parallel_enabled(cfg: Config) -> bool:
    n = getattr(cfg, "n_workers", None)
    return bool(n) and n > 1

def _worker_init(cfg: Config) -> None:
    _WORKER["cfg"] = cfg
    _WORKER["reference"] = vibronic.diagonalize_single_molecule_reference(cfg)

def _worker_task(task):
    key, eps1, eps2 = task
    cfg = _WORKER["cfg"]
    sol = solve_representative(cfg, eps1, eps2, reference=_WORKER["reference"])
    hm = participation.bright_heatmap(sol, cfg)
    return key, hm
```

- `_parallel_enabled` — same convention as `spectrum.py`: parallelism is
  opt-in via `cfg.n_workers > 1`.
- `_WORKER` is a **module-level dict living inside each worker process**
  (each spawned process gets its own copy — this is not shared state across
  processes). `_worker_init` runs once when a worker process starts: it
  builds the single-molecule reference **once per worker** and stores both it
  and `cfg` for that worker's lifetime.
- `_worker_task` is what each worker executes per assigned case: unpack the
  task `(key, eps1, eps2)`, solve (reusing this worker's cached reference —
  no re-diagonalization of the single-molecule basis), compute its bright
  heatmap, and return `(key, heatmap)` so the parent can reassemble results
  by key regardless of completion order.

```python
def _compute_parallel(cfg, tasks, show_progress, desc):
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    saved = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
    for k in _THREAD_ENV_VARS:
        os.environ[k] = "1"

    out: Dict[str, participation.HeatmapResult] = {}
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=cfg.n_workers, mp_context=ctx,
            initializer=_worker_init, initargs=(cfg,),
        ) as ex:
            it = ex.map(_worker_task, tasks, chunksize=1)
            if show_progress:
                it = tqdm(it, total=len(tasks), desc=desc, unit="case")
            for key, hm in it:
                out[key] = hm
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out
```

Exactly the pattern from `spectrum.py`'s `_disorder_average_parallel`
(same rationale — see [`SPECTRUM_WALKTHROUGH.md`](SPECTRUM_WALKTHROUGH.md)
§11): pin every BLAS-related thread-count env var to `"1"` **before**
spawning workers (each worker's single `eigh` doesn't benefit from
multithreading; parallelism instead comes from running many workers, each
single-threaded, concurrently — avoiding the oversubscription documented in
this project's `blas-thread-perf-fix` history), spawn a `ProcessPoolExecutor`
with `mp.get_context("spawn")` (not `fork` — spawn re-imports cleanly, needed
for this to work outside a `__main__`-guarded script), map `_worker_task`
over every `(key, eps1, eps2)` task, optionally wrap with a progress bar,
collect `{key: heatmap}` as results arrive, then **restore** the original
environment variables afterward (so this function has no side effect on the
calling process's env once it returns).

### 4.5 Public entry points (lines 151–202)

```python
def compute_representative_heatmaps(selections, cfg, show_progress=True):
    tasks = [
        (key, sel["metadata"].eps1, sel["metadata"].eps2)
        for key, sel in selections.items()
    ]

    if _parallel_enabled(cfg):
        hm_by_key = _compute_parallel(cfg, tasks, show_progress, "representative heatmap")
    else:
        reference = vibronic.diagonalize_single_molecule_reference(cfg)
        iterator = tasks
        if show_progress:
            iterator = tqdm(tasks, desc="representative heatmap", unit="case")
        hm_by_key = {}
        for key, eps1, eps2 in iterator:
            sol = solve_representative(cfg, eps1, eps2, reference=reference)
            hm_by_key[key] = participation.bright_heatmap(sol, cfg)

    return {key: _heatmap_entry(sel, hm_by_key[key]) for key, sel in selections.items()}
```

Takes an already-computed `selections` dict (case name → `RealizationMetadata`
+ description) and produces one heatmap per case: build the flat task list
`(key, eps1, eps2)`, dispatch either to the parallel path or a simple serial
loop (which still only builds the single-molecule reference **once**, shared
across every case, mirroring the parallel path's per-worker reuse), then
reassemble results **in the original `selections` order** (dict
comprehension over `selections.items()`, not over `hm_by_key`) — so downstream
consumers see cases in the same A→K order regardless of which finished first.

```python
def run_representative_heatmaps(Nv, cfg, show_progress=True) -> Dict:
    metadata = collect_realization_metadata(cfg)
    selections = select_representative_realizations(metadata, cfg)
    heatmaps = compute_representative_heatmaps(selections, cfg, show_progress=show_progress)
    return {"Nv": Nv, "metadata": metadata, "selections": selections, "heatmaps": heatmaps}
```

The full standalone workflow: replay the RNG stream to get the `(eps1, eps2)`
metadata cloud (`collect_realization_metadata`, from `representative.py` —
*not* the absorption disorder average itself, just the lightweight per-draw
bookkeeping), select the 11 physical cases from it
(`select_representative_realizations`, also from `representative.py`,
unmodified), then compute their heatmaps. This is why the heatmap workflow
is cheap: it never touches `spectrum.compute_spectrum_for_Nv` or the
disorder-averaging loop at all.

### 4.6 Plotting (lines 205–253)

```python
def _crop_v_range(grid: np.ndarray, heatmap: np.ndarray, floor: float = 1e-4):
    rows = np.where(heatmap.max(axis=1) > floor)[0]
    if rows.size == 0:
        return int(grid[0]), int(grid[-1])
    return int(grid[rows[0]]), int(grid[rows[-1]])
```

Finds the tightest contiguous `v`-range that holds any non-negligible weight
(`heatmap.max(axis=1)` is each row's peak value across all bright states;
`floor=1e-4` filters out rows that are essentially always zero) — used to
crop the plot's y-axis so it isn't dominated by empty high-`|v|` sectors that
`sector_grid` (§3.2) included for completeness but which never get populated.

```python
def plot_representative_heatmap(entry, cfg, out_path, show=True) -> str:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap = entry["heatmap"]
    grid = entry["grid"]
    n_bright = heatmap.shape[1]
    x = np.arange(n_bright)

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(x, grid, heatmap, cmap="viridis", shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$P(v)$")
    ax.set_xlabel("Bright polaritonic states (energy-ordered)")
    ax.set_ylabel(r"Vibronic sector $v$")
    ax.set_title(
        f"{entry['case']}\n"
        f"realization #{entry['realization_index']}: "
        rf"$\epsilon_1$={entry['eps1']:.3f} eV, $\epsilon_2$={entry['eps2']:.3f} eV "
        f"(Nv={cfg.heatmap_nv}, σ={cfg.sigma:g})"
    )
    vlo, vhi = _crop_v_range(grid, heatmap)
    ax.set_ylim(vlo - 0.5, vhi + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path
```

- `matplotlib.use("Agg")` (headless backend) is selected *before* importing
  `pyplot`, only when `show=False` — the same "no GUI needed" pattern used
  throughout `plotting.py`.
- `x = np.arange(n_bright)` — the x-axis is simply **position** among the
  bright states (0, 1, 2, …), not their raw energy — matching the paper's
  Fig. S1, which plots against discrete polariton *index*, not a continuous
  energy axis (unlike the broadened absorption spectrum plots elsewhere in
  this project).
- `ax.pcolormesh(x, grid, heatmap, ...)` draws the actual heatmap: for
  `pcolormesh(X, Y, C)`, `C[i, j]` is drawn as the cell at column `X[j]`, row
  `Y[i]` — here that's `heatmap[i, j] = P(v=grid[i])` for bright state `j`,
  exactly right for `heatmap`'s `(n_v, n_bright)` shape.
  `shading="nearest"` treats `x`/`grid` as cell centers rather than requiring
  `N+1`-length edge arrays.
- Title embeds the case name, the actual realization index and its
  `(eps1, eps2)`, plus the run's `heatmap_nv` and `sigma` for full provenance
  at a glance.
- y-limits use `_crop_v_range` with half-cell padding (`±0.5`) so the
  outermost populated row isn't clipped by the axis edge.

### 4.7 Storage (lines 256–299)

```python
def representative_heatmap_path(cfg, Nv, case_name, ext="png") -> str:
    return os.path.join(
        cfg.results_dir,
        f"representative_heatmap_Nv{Nv}_{_sigma_tag(cfg.sigma)}_{_case_slug(case_name)}.{ext}",
    )
```

Builds the output path, reusing `storage.py`'s existing `_sigma_tag`
(formats `sigma` filesystem-safely, e.g. `0.24` → `"sigma0.24"`) and
`_case_slug` (`"A: Resonant baseline"` → `"caseA"`) helpers — imported, not
duplicated, so heatmap filenames stay consistent with the representative
spectrum filenames.

```python
def save_representative_heatmap(entry, cfg, Nv) -> str:
    ensure_results_dir(cfg)
    path = representative_heatmap_path(cfg, Nv, entry["case"], ext="npz")
    np.savez_compressed(
        path,
        case=entry["case"], letter=entry["letter"], description=entry["description"],
        realization_index=int(entry["realization_index"]),
        eps1=float(entry["eps1"]), eps2=float(entry["eps2"]),
        delta1=float(entry["delta1"]), delta2=float(entry["delta2"]),
        grid=entry["grid"], energies=entry["energies"], intensity=entry["intensity"],
        heatmap=entry["heatmap"], pr=entry["pr"],
        Nv=Nv, heatmap_nv=cfg.heatmap_nv, sigma=cfg.sigma, rng_seed=cfg.rng_seed,
        n_realizations=cfg.n_realizations, which_molecule=str(cfg.heatmap_which_molecule),
        bright_threshold=cfg.heatmap_bright_threshold,
        omega=cfg.omega, kappa=cfg.kappa, omega_c=cfg.omega_c, Omega=cfg.Omega, eps=cfg.eps,
    )
    return path
```

Persists one case's full result as a compressed `.npz`: the plotted arrays
(`grid`, `energies`, `intensity`, `heatmap`, `pr`), the case's identifying
metadata, and a provenance block (every config value that could affect the
result: grid resolution `heatmap_nv`, `sigma`, `rng_seed`, `n_realizations`
used for the selection cloud, which-molecule convention, bright threshold, and
the physical constants) — enough to fully audit or reproduce the plot later
without recomputation, following the same provenance convention as
`storage.py`'s `save_representative_spectrum`.

---

## 5. `main.py` — CLI wiring

### 5.1 The flag (near the other `--repr-*` flags)

```python
p.add_argument("--representative-heatmap", action="store_true",
               help="For each physically selected representative realization "
                    "(Cases A-K, same selection as --representative), build the "
                    "vibronic->polaritonic Hamiltonian and plot the paper's "
                    "discrete Fig. S1 P(v) heatmap. ...")
p.add_argument("--heatmap-nv", type=int, default=None,
               help="Vibrational Fock cutoff for the heatmap vibronic basis "
                    "(default: config heatmap_nv=12).")
```

`--representative-heatmap` is a plain boolean flag selecting this workflow
instead of the default Nv sweep (or `--representative`, `--sigma-sweep`,
etc.). `--heatmap-nv` overrides `cfg.heatmap_nv` specifically for this
workflow's vibronic basis size, independent of the absorption pipeline's own
`Nv`/`nv_list`.

### 5.2 `run_representative_heatmap` (lines 293–341)

```python
def run_representative_heatmap(args, cfg: Config) -> None:
    from spectrum import representative_heatmap as rh

    Nv = args.nv[0] if args.nv else cfg.representative_nv
    cfg.n_realizations = (
        args.repr_realizations if args.repr_realizations is not None
        else cfg.representative_n_realizations
    )
    if args.repr_sigma is not None:
        cfg.sigma = args.repr_sigma
    if args.heatmap_nv is not None:
        cfg.heatmap_nv = args.heatmap_nv

    if cfg.n_workers is None:
        usable = max(1, (os.cpu_count() or 1) - 2)
        cfg.n_workers = max(1, min(usable, 11))
        if cfg.n_workers > 1:
            print(f"[repr-heatmap] auto-parallelizing over {cfg.n_workers} "
                  f"worker processes (override with --workers N).")

    ensure_results_dir(cfg)

    print(f"[repr-heatmap] selecting Cases A-K (Nv={Nv}, sigma={cfg.sigma:g}, "
          f"{cfg.n_realizations} realizations) and building P(v) heatmaps "
          f"(heatmap_nv={cfg.heatmap_nv}) ...")
    result = rh.run_representative_heatmaps(Nv, cfg, show_progress=True)

    header = f"{'Case':<32}{'Idx':>6}{'eps1':>10}{'eps2':>10}{'#bright':>9}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, entry in result["heatmaps"].items():
        n_bright = entry["heatmap"].shape[1]
        print(f"{name:<32}{entry['realization_index']:>6}{entry['eps1']:>10.4f}"
              f"{entry['eps2']:>10.4f}{n_bright:>9}")
        rh.save_representative_heatmap(entry, cfg, Nv)
        out_path = rh.representative_heatmap_path(cfg, Nv, entry["case"])
        rh.plot_representative_heatmap(entry, cfg, out_path, show=not args.no_show)

    print(f"\nDone. All heatmaps in {cfg.results_dir}/")
```

1. **Deferred import** (`from spectrum import representative_heatmap as rh`
   *inside* the function): keeps this analysis's modules — and their
   `matplotlib`/`tqdm` import weight — completely out of the default code
   path; running plain `python main.py` never touches this module.
2. **Resolve `Nv`, `n_realizations`, `sigma`, `heatmap_nv`** from CLI overrides
   with config defaults as fallback — the same override pattern as
   `run_representative` in the same file, so the two workflows behave
   consistently (e.g. `--nv` picks the first value if given, same as the
   representative-spectrum runner).
3. **Auto-parallelize** (lines 316–321): if the user didn't pin `--workers`,
   pick `min(cpu_count - 2, 11)` workers — capped at 11 because that's (at
   most) how many cases there are; more workers than tasks would be wasted.
   Prints a one-line notice only if actually parallelizing (`> 1`).
4. **Run the workflow** (`rh.run_representative_heatmaps`) with a progress
   bar enabled.
5. **Print a summary table** — case name, realization index, `(eps1, eps2)`,
   and how many bright states it produced — then for each case: save the
   `.npz` (§4.7) and render the plot (§4.6), respecting `--no-show` for
   headless runs.

### 5.3 Dispatch in `main()`

```python
if args.representative_heatmap:
    run_representative_heatmap(args, cfg)
    return

if args.representative:
    run_representative(args, cfg)
    return
```

`--representative-heatmap` is checked **before** `--representative`, so if
(hypothetically) both were passed, the heatmap workflow would win — though in
normal use they're mutually exclusive alternatives, each an early return
that skips every other workflow (Nv sweep, sigma sweep, realization sweep).

---

## 6. Validation: how we know this matches the frozen absorption physics

`tests/test_representative_heatmap.py` anchors every new module against the
**existing, unmodified** absorption pipeline (`basis.py` +
`hamiltonian.py`):

- **`test_vibronic_route_matches_primitive_disordered`** — builds the
  primitive-basis Hamiltonian for an explicit *disordered* `(eps1, eps2)`
  pair (not just `sigma=0`) exactly as `spectrum.py` would, diagonalizes it,
  and separately runs `solve_representative` for the *same* pair. Asserts the
  sector dimension matches and every eigenvalue/intensity agrees to `1e-9`
  once both are sorted by energy. This is the strongest check: it proves the
  vibronic route (built entirely from `vibronic.py`/`polariton.py`) is just a
  change of basis of the *same physical Hamiltonian* the frozen absorption
  pipeline diagonalizes — not a different or approximate model.
- **`test_Pv_columns_sum_to_one`** / **`test_bright_heatmap_columns_sum_to_one`**
  — numerically confirm `Σ_v P(v) = 1` for every eigenstate and every bright
  column, and that `PR ≥ 1` everywhere — direct checks of the §3
  probability-conservation claims.
- **`test_end_to_end_representative_heatmaps`** /
  **`test_solve_matches_selected_representative_index`** — run the full
  `run_representative_heatmaps` workflow, check all 11 letters A–K appear with
  valid heatmaps, and confirm that a selected case's heatmap was built from
  *exactly* that realization's `(eps1, eps2)` (cross-checked again against the
  primitive route for that pair).

All of these pass, and — as stated at the top of this document and verified
by `git diff` against the branch point — `basis.py`, `operators.py`,
`hamiltonian.py`, and `spectrum.py` are byte-identical to before this
workflow was added.
