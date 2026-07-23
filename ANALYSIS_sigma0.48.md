# Disorder analysis at σ = 0.48 eV — polaritonic spectra & vibronic-cascade heatmaps, case by case

**System:** two (E×e) Jahn–Teller molecules collectively coupled to the two circularly
polarized modes of a Fabry–Perot cavity (the model of the 2025 JT paper), single-excitation
`(n_ex=1, j=−1)` sector. Nv = 12. All eight representative realizations were drawn from a
Gaussian site-energy disorder of width **σ = 0.48 eV** and are compared against the
**no-disorder reference** `heatmap_figS1_Nv12` (the paper's Fig. S1, clean N=2) and the clean
absorption spectrum `spectrum_ref_2mol_2.2.pl`.

**Reference energy scales (all from `spectrum/config.py`):**

| scale | value | meaning |
|---|---|---|
| ω (vibrational quantum) | 0.082 eV | vibronic level spacing |
| **g** = Ω/(2√N) | **0.242 eV** | per-molecule matter–cavity coupling (`cfg.g`) |
| √N·g = Ω/2 | 0.342 eV | collective coupling (= LP offset below the bright line) |
| **Ω = 2√N·g** | **0.685 eV** | collective Rabi splitting (LP↔UP separation) |
| ε = 7.0, ω_c = 6.85 | — | electronic origin / cavity; bright transition sits at ε−0.149 ≈ ω_c |

At σ = 0.48: **σ/g ≈ 2.0, σ/(√N·g) ≈ 1.4, σ/Ω ≈ 0.70.** In the language of the arrowhead
paper the disorder-averaged system sits just *below* the `W = 2g` crossover (disorder width =
Rabi splitting), i.e. still nominally strong-coupling but with heavy bright/dark mixing — which
is exactly what the case set shows.

---

## 0. TL;DR

- The **cascade** (spread of the bright polaritons over vibronic angular-momentum sectors *v*,
  measured by the participation ratio **PR**) is controlled by **two independent disorder axes**:
  the **inter-molecular mismatch** Δ = ε₁−ε₂ (vs **g**) and the **common-mode detuning**
  s = (δ₁+δ₂)/2 (vs the collective coupling **√N·g ≈ Ω**).
- **Bright/dark mixing is the single mechanism behind everything below** (§1): disorder breaks the
  permutation symmetry that separates the bright polariton from the dark manifold, so dark states
  *brighten* (gain photon weight ∝ σ²/2g²) and the polariton eventually *dissolves* into the
  disorder band.
- **Case A ≈ no-disorder reference** (internal control). **F, G are enhanced** (disorder-focused,
  dark-state brightening). **B, E, D are partial.** **C, H are dead** — by two *different*
  mechanisms (localization vs decoupling).

---

## 1. Where does bright/dark mixing come in? (the core mechanism)

This is the conceptual hinge, so it comes first.

### 1.1 Clean case: a bright state and a dark manifold

In the single-excitation sector the Hamiltonian has the **arrowhead structure** of the disordered
Tavis–Cummings model (arrowhead paper, Eq. 2): the cavity photon is the "arrow tip", coupled with
strength g/√N to the matter states on the diagonal. For **N = 2, matched, resonant** molecules
(case A / figS1) the permutation symmetry splits matter into:

- a **bright** (symmetric) combination `(|E,A⟩+|A,E⟩)/√2` that carries *all* the photon coupling →
  forms the lower/upper polaritons (LP/UP), split off by ±√N·g; and
- a **dark** (antisymmetric) combination `(|E,A⟩−|A,E⟩)/√2` with **zero** photon weight and **zero**
  absorption intensity.

Because absorption starts from `|A,A; photon⟩`, the **absorption intensity of eigenstate ψ is
exactly its photon weight** `PW = |⟨G,1|ψ⟩|²`. So your **polaritonic (absorption) spectrum *is* the
arrowhead paper's photon spectral function** `A(ω) = Σ_a PW_a δ(ε−ε_a)` (arrowhead Eq. 12), and the
**heatmap** is the vibronic-sector decomposition P(v) of those photon-weighted (bright) states.

The JT vibronic coupling then lets the bright polariton exchange vibrational angular momentum
repeatedly → the **collective vibronic cascade** (2025 JT paper): the bright states spread over many
*v* sectors (PR up to ~20 for N=2, vs the single-molecule bound v ∈ {0,−1,−2}, PR≈3 of the 2023 JT
paper).

### 1.2 Disorder turns the mixing on

Disorder makes ε₁ ≠ ε₂. The site-energy difference enters as a term
`≈ (Δ/2)(|E,A⟩⟨E,A| − |A,E⟩⟨A,E|)`, whose off-diagonal part **couples bright ↔ dark** with an
amplitude set by the **mismatch Δ/2**; a common shift s rigidly moves the whole bright manifold
relative to the fixed photon. Two consequences, both quantified in the papers:

1. **Dark states brighten.** They acquire photon weight — per the disorder paper,
   `Σ_dark |⟨d|1_ph⟩|² ≈ W²/(2g_c²)` (W ↔ σ) — and per the arrowhead paper each dark state gains
   `O(1/N)` photon weight, `O(1)` in total (arrowhead Eq. 10–11). Brightened dark states **appear in
   the absorption spectrum** (extra peaks, inhomogeneous broadening) and **as extra columns in the
   heatmap**.
2. **The protection is the collective splitting.** LP/UP are pushed *outside* the disorder band by
   ±√N·g (arrowhead interlacing, Eq. 18: dark eigenvalues interlace the bare energies; the two
   polaritons are the outliers). The polariton stays a well-defined 50/50 light–matter mode only
   while the coupling exceeds the disorder — the **`W = 2g` crossover** (arrowhead Fig. 4–5). Past
   it, the polariton sinks into the band and **loses photon weight** → it reverts to bare photon /
   bare molecule.

### 1.3 The competition (this is the whole story)

| bright↔dark mixing strength | outcome | cases |
|---|---|---|
| mixing ≪ √N·g (Δ, σ small) | dark stays dark; clean LP/UP; full cascade | **A** (≈figS1) |
| mixing ~ √N·g (Δ or s ~ g, near resonance) | dark manifold **brightens**; oscillator strength spread over many polaritonic states; cascade preserved or **enhanced** | **F, G** (and partially E, B) |
| mixing ≫ √N·g (Δ ≫ g, *or* \|s\| ≫ √N·g) | polariton **dissolves** → localization (one molecule) or decoupling (bare photon); cascade dies | **C, H** (D intermediate) |

So bright/dark mixing is not a side effect — it is *the* variable. Moderate mixing **exposes** the
cascade (dark states that were invisible become bright and reveal their own vibronic spread);
excessive mixing **destroys** it.

---

## 2. No-disorder reference (baseline for every comparison)

| quantity | value |
|---|---|
| Heatmap `heatmap_figS1_Nv12`: #bright | 42 |
| PR_max / PR_mean | **20.08 / 12.66** |
| accessed v-range | v ∈ [−15, 14], span 29 |
| Clean absorption `spectrum_ref_2mol_2.2.pl` | LP 6.514 (+ shoulders 6.606, 6.672); UP cluster 6.98 / 7.07 / 7.23 / 7.25 / 7.34 |

This is the fully collective cascade of the 2025 JT paper (Figs. 3, S1): LP well-resolved, UP
broadened into many short stems (the cascade fingerprint), PR reaching ~20.

---

## 3. Case-by-case (σ = 0.48 eV)

Notation: δᵢ = εᵢ − 7.0 (detuning from resonance), Δ = ε₁−ε₂ (mismatch), s = (δ₁+δ₂)/2
(common-mode). "PR" = participation ratio of the bright polaritons (cascade metric).

| case | (ε₁, ε₂) | Δ | s | #bright | PR_max | v-span | verdict |
|---|---|---|---|---|---|---|---|
| **A** Resonant baseline | (6.984, 7.006) | 0.02 | ~0 | 39 | 20.6 | 28 | ≈ no-disorder |
| **B** Mismatch ≈ g | (7.129, 7.368) | 0.24 (g) | +1.0g | 14 | 11.6 | 23 | partial |
| **C** Large mismatch | (8.082, 7.078) | 1.00 (4g) | +2.4g | 11 | 3.8 | (see text) | **dead: localized** |
| **D** Opposite disorder | (6.277, 7.763) | 1.49 (6g) | ~0 | 14 | 8.5 | (bounded) | partial→dead |
| **E** Common blue | (7.334, 7.169) | 0.17 | +1.0g | 12 | 12.5 | 22 | partial |
| **F** Common red (matched) | (6.674, 6.702) | 0.03 | −1.3g | 104 | 22.4 | 31 | **enhanced** |
| **G** Single-mol resonant | (7.010, 6.752) | 0.26 (g) | −0.5g | 132 | 21.6 | 29 | **enhanced** |
| **H** Bare-molecule limit | (6.246, 6.146) | 0.10 | −3.3g | 8 | 4.5 | 9 | **dead: decoupled** |

### A — Resonant baseline (internal control)
Both molecules at resonance, matched (Δ≈0, s≈0). **Absorption:** LP 6.518 + shoulders 6.611/6.676;
UP triplet 7.222/7.247/7.331 — *quantitatively identical* to the clean `.pl` spectrum. **Heatmap:**
39 bright states, PR_max 20.6, v ∈ [−14,14] — indistinguishable from figS1 (42, 20.1, [−15,14]).
**Meaning:** a near-zero-disorder draw reproduces the clean collective cascade — confirms the
pipeline and anchors every other case. (2025 JT paper, Figs. 3/S1.)

### F — Common red detuning, matched (**enhanced**)
Both molecules ≈ 0.3 eV *below* resonance but **matched** (Δ = 0.03). **Absorption:** whole spectrum
**red-shifted** (LP cluster 6.31/6.40/6.49) and, strikingly, the **UP at 7.125 becomes the brightest
peak** (intensity inverted vs A) — a redistribution of oscillator strength. **Heatmap:** **104**
bright states (2.6× the reference!), PR_max 22.4, v ∈ [−16,15] — the *richest* cascade of all.
**Mechanism:** matched molecules keep the collective coupling intact, but the −1.3g common shift
lowers the bright polariton toward the dark manifold → strong **bright/dark mixing** brightens the
whole dark band (dark photon weight ∝ σ²/2g², arrowhead Eq. 11). Because those dark states are
themselves JT-cascading, brightening them **exposes more of the cascade** → this is the disorder
paper's *disorder-enhanced vibrational dynamics* (peaks at W ~ g_c), here landing in the ±g_c "sweet
spot".

### G — Single-molecule resonant, but Δ ≈ g (**enhanced**, not single-molecule-like)
One molecule on resonance (δ₁≈0), the other only −g away — so Δ = 0.26 ≈ g is at the *crossover*, not
`≫g`; the two molecules **still hybridize collectively**. **Absorption:** the most structured of all,
9 resolved peaks 6.39–6.95. **Heatmap:** **132** bright states (most of any case), PR_max 21.6,
v ∈ [−15,14], visibly the broadest P(v) spread. **Meaning:** despite the label, G is a *mildly
asymmetric collective* case, not a localized one; the slight symmetry breaking maximally brightens
the dark manifold. Contrast with C below, where Δ ≫ g genuinely localizes.

### E — Common blue detuning (**partial**)
Both molecules ≈ +g above resonance, near-matched (Δ = 0.17). **Absorption:** blue-shifted, collapsed
to ~2 resolved peaks (LP 6.623, 6.708), UP pushed up to 7.40. **Heatmap:** 12 bright, PR_max 12.5,
v ∈ [−12,10]. **Meaning:** joint blue detuning of ~+g moves the bright manifold partly off the photon;
cascade reduced to ~60% of baseline. Symmetric partner of F but blue and *weaker* (fewer brightened
states — the +g shift here pushes the bright polariton *away* from, not into, the dense dark band).

### B — Mismatch ≈ coupling (**partial / crossover**)
Δ = 0.24 = g, plus a +g common blue shift. **Absorption:** blue-shifted, only 2 resolved peaks
(6.615/6.698), UP at 7.40. **Heatmap:** 14 bright, PR_max 11.6, v ∈ [−12,11]. **Meaning:** the
textbook **collective→localized crossover** — mismatch equal to the coupling half-disrupts the
sharing; PR ≈ half the baseline. (Arrowhead: mixing ~ splitting.)

### D — Opposite disorder, large mismatch (**partial → localizing**)
Molecules straddle resonance oppositely (6.28 and 7.76), Δ = 1.49 = 6g, but s ≈ 0. **Absorption:**
collapses to essentially a **single peak at 6.876 ≈ ω_c** — the near-bare cavity photon, since
neither molecule is near the bright line. **Heatmap:** 14 bright but weight mostly pinned to
v ∈ {0,−1,−2}, with only a faint high-v tail (PR_max 8.5). **Meaning:** the huge mismatch localizes,
yet the *symmetric straddle* keeps a little residual two-molecule coupling → intermediate PR,
higher than the fully-dead C/H.

### C — Large mismatch, same side (**dead: localization**)
ε₂ = 7.078 on resonance, ε₁ = 8.082 detuned by +1.08 = 4g ≫ g. **Absorption:** LP 6.620 + a clean
vibrational **progression** 6.701/6.763 + one UP feature 7.113 — i.e. the spectrum of a **single**
JT-polariton (the resonant molecule), exactly the 2023 JT paper's one-molecule result. **Heatmap:**
11 bright states, but **all** with P(v) confined to **v ∈ {0,−1,−2}** (a flat horizontal band),
PR_max 3.8 ≈ the single-molecule bound. **Mechanism (kill #1 — mismatch axis):** Δ ≫ g decouples the
molecules (arrowhead: bright↔dark mixing overwhelms the splitting); the excitation **localizes** on
the resonant molecule → bounded single-molecule cascade. The 11 bright states are its
Franck–Condon vibrational progression, not a cascade.

### H — Bare-molecule limit (**dead: decoupling**)
Molecules **matched** (Δ = 0.10) but both ≈ 0.8 eV = 3.3g *below* the cavity (s = −3.3g). **Absorption:**
a **single peak at 6.998 ≈ ω_c** — the essentially uncoupled cavity photon. **Heatmap:** only 8
bright states, v ∈ [−5,4], one state ~90% in v = 0 (PR_max 4.5). **Mechanism (kill #2 — common-mode
axis):** although perfectly matched, both molecules lie *outside* the polariton band (|s| ≫ √N·g), so
per the arrowhead picture the bright eigenstate **reverts to the bare photon** with negligible matter/
vibronic character. The cavity-mediated exchange that drives the cascade is switched off → cascade
dead. **This is a distinct route from C:** C kills by mismatch (localization), H kills by joint
detuning (decoupling). Both land at PR ≈ 4.

---

## 4. Synthesis — two axes, three regimes

Ordering by cascade strength: **F (22.4) ≈ G (21.6) ≈ A (20.6) > E (12.5) ≈ B (11.6) > D (8.5) > H (4.5) ≈ C (3.8).**

```
                     common-mode detuning s  (vs √N·g ≈ Ω)
                     small                         large
                ┌───────────────────────────┬──────────────────────────┐
 mismatch Δ     │ A  full cascade (≈figS1)   │ F  matched+red → ENHANCED │  small Δ
 (vs g)  small  │ E/B  mild blue → partial   │ H  DEAD (decoupled photon)│  (matched)
                ├───────────────────────────┼──────────────────────────┤
         large  │ C  DEAD (localized, 1-mol) │ D  single peak, localizing│  large Δ
                │ G* (*Δ≈g only → still coll.)│                          │  (mismatched)
                └───────────────────────────┴──────────────────────────┘
```

- **Mismatch axis (Δ vs g):** Δ ≲ g → collective (A, E, B, G); Δ ≫ g → localize on one molecule →
  single-molecule bounded cascade (C; D partially). This is the 2023↔2025 JT crossover.
- **Common-mode axis (s vs √N·g):** |s| ≲ √N·g → pair stays in the polariton (A, F even at 1.3g);
  |s| ≫ √N·g → both fall out of the polariton, bright state → bare photon (H). This is the arrowhead
  `W = 2g` dissolution applied to the joint detuning.
- **Enhancement band:** when the bright polariton is pushed *toward* the dense dark manifold (F, G:
  offsets ~±g_c, near resonance) moderate mixing brightens many dark states and **enriches** the
  observed cascade — the disorder paper's central result, and the reason #bright jumps 42 → 104/132.

**Everything is one story (from §1):** the amount of **bright/dark mixing** (set by Δ and s relative
to the couplings) decides whether disorder *exposes* the cascade (F, G), leaves it intact (A),
partially disrupts it (E, B, D), or destroys it (C, H).

---

## 5. Caveats (read before drawing quantitative conclusions)

1. **Single realizations, not statistics.** Each case is one extremal (ε₁,ε₂) draw chosen by the
   selection criteria; PR and #bright fluctuate strongly between realizations. Use these as
   *mechanism illustrations*. For quantitative claims, average PR over all realizations at σ=0.48.
2. **#bright is threshold-dependent** (`heatmap_bright_threshold = 1e-2`, relative to each case's
   max). The 42 → 104/132 jump partly reflects a *lower peak* with more states above 1%. **PR (per
   state) is the cleaner cascade metric** — trust it over the raw count. (The brightening it reflects
   is real; its magnitude via #bright is not a hard number.)
3. **σ = 0.48 is one strong value** (σ/Ω ≈ 0.70, just below the `W=2g` crossover). To *demonstrate*
   the two-axis picture you want the same 8 cases at σ ≈ g ≈ 0.24 (already computed) and, ideally,
   PR mapped over the (Δ, s) plane — then C/H should switch from near-collective to dead as σ grows.
4. The 2-molecule "bright/dark" language is the N→∞ arrowhead limit specialized to N=2; with only two
   molecules there is one dark combination per vibronic channel, so "brightening" here means the
   antisymmetric channels gaining photon weight, not a large dark *band*.

---

## 6. References (what each result rests on)

- **2023 JT** — Nandipati & Vendrell, *Cavity Jahn–Teller polaritons in molecules*, PRA **107**,
  L061101 (2023). Single-molecule JT-polariton; Eq. (1) JT Hamiltonian, Eq. (4) cavity coupling Ω/2;
  Fig. 3 LP/UP; bounded single-molecule sector. → cases **C, H** (bounded v ∈ {0,−1,−2}).
- **2025 JT** — Pandit, Pandey, Shankar, Nandipati, *Collective Vibronic Cascade…* (2025). Collective
  cascade; Fig. 2(b) cascade mechanism; Fig. 3 PR up to ~20 for N=2; Fig. S1 heatmap = our reference.
  → cases **A, F, G** (full/enhanced cascade); the disorder outlook motivates this whole study.
- **Arrowhead** — Dubail, Botzung, Schachenmayer, Pupillo, Hagenmüller, *Large Random Arrowhead
  Matrices*, arXiv:2105.08444. Arrowhead Eq. (2); interlacing Eq. (18); photon spectral function
  A(ω) Eq. (12) = our absorption; polariton photon weight O(1) Eq. (10), dark O(1/N) Eq. (11);
  `W=2g` crossover Figs. 4–5; semi-localization. → the bright/dark **structure and dissolution**
  (§1, cases **C, D, H**, and the brightening in **F, G**).
- **Disorder** — Wellnitz, Pupillo, Schachenmayer, *Disorder enhanced vibrational entanglement and
  dynamics in polaritonic chemistry*, Commun. Phys. **5**, 120 (2022). Disordered Holstein–Tavis–
  Cummings (same g = g_c/√N); disorder-focused transfer to molecules at offset ~±g_c; dark photon
  weight ~W²/2g_c²; enhancement peaks at W ~ g_c. → the **enhancement** in **F, G** and the
  brightening picture (§1.2).

*Generated from the σ=0.48 results in `results/` (heatmap + spectrum `.npz`) and the four papers.
No project code was modified.*
