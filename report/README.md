# Internship report

`report.tex` — *Disorder in Cavity-Coupled Jahn–Teller Active Molecules*.

## Build

There is no system TeX Live on this machine, so a self-contained engine
([tectonic](https://tectonic-typesetting.github.io/)) is installed locally in
`../.texenv` (via `conda create -p ../.texenv -c conda-forge tectonic`; not
tracked in git — see `.gitignore`). Recompile after editing `report.tex` with:

```bash
cd report
../.texenv/bin/tectonic report.tex     # writes report.pdf, in one shot
```

Tectonic downloads and caches any additional LaTeX packages/fonts it needs on
first run (needs network access once); reruns are fully offline and fast.
`report.log` is written alongside for debugging (`--keep-logs`) and is
git-ignored.

No BibTeX/biber run is needed — the reference list is a plain
`thebibliography` environment.

Anywhere else (e.g. Overleaf, or a machine with `pdflatex`/TeX Live), the
ordinary two-pass build also works — upload `report.tex` plus the `figures/`
folder:

```bash
pdflatex report.tex
pdflatex report.tex     # second pass resolves \ref / \cite
```

## Contents

| Section | Source |
|---|---|
| 1–2 Cavity–matter interaction, Jahn–Teller $(E\times e)$ | background |
| 3 Single molecule: cavity-JT polaritons | Nandipati & Vendrell, PRA **107**, L061101 (2023) — Eqs. (1)–(5) reproduced verbatim |
| 4 Collective vibronic cascade, $P(v)$, PR | Pandit *et al.*, arXiv:2511.07880 (2025) — Eqs. (1)–(7), SM Eqs. (S7)–(S10) |
| 5 Introducing disorder | Wellnitz *et al.*, Commun. Phys. **5**, 120 (2022), Eqs. (1)–(4); Dubail *et al.*, arXiv:2105.08444 (2021) |
| 6 Method & implementation | this work |
| 7 Results: cases A–H at $W = 0.48$ eV | this work |
| 8–9 Discussion, summary | this work |

The section order follows `../slides/present.pdf`.

## Figures

`figures/` holds copies of the plots so the folder is self-contained. Their
originals are:

| In report | Original |
|---|---|
| `fig_1mol_sticks.png`, `fig_2mol_two_kappa.png`, `fig_disorder_twolevel.png`, `fig_case_*_spec.png` | `../slides/` |
| `fig_clean_heatmap.png` | `../results/heatmap_figS1_Nv18.png` |
| `fig_gaussian.png` | `../results/disorder_gaussian_sigma0.48.png` |
| `fig_scatter.png` | `../results/representative_scatter_Nv12_sigma0.48.png` |
| `fig_case_*_heatmap.png` | `../results/representative_heatmap_Nv12_sigma0.48_case*.png` |

Regenerate the underlying data with:

```bash
venv/bin/python main.py --representative         --nv 12 --repr-realizations 100 --repr-sigma 0.48 --no-show
venv/bin/python main.py --representative-heatmap --nv 12 --repr-realizations 100 --repr-sigma 0.48 \
                        --heatmap-energy-axis --workers 8 --no-show
```

## Numbers quoted in the report

All per-case values (#bright, `PR_max`, `v`-range, peak positions) were read
back from `../results/representative_{spectrum,heatmap}_Nv12_sigma0.48_case*.npz`;
the `v`-range convention is "sectors with `P(v) > 0.01` for at least one bright
state", which is stated in the caption of Table 2.
