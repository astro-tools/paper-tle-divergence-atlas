# Contributing

This is a research-paper repository, not a software project. The bar for contributions is different from a tool repo:

- **Issues are welcome** for reproducibility problems, methodological questions, suggestions for additional analyses, and bug reports on figure scripts.
- **PRs are welcome** for figure improvements, additional sensitivity checks, methodology refinements, and prose corrections (typos, clarity).
- **Material changes to the science** (new factors, new constellation, different methodology) should start as an issue for discussion before any PR.

## Local setup

```bash
git clone https://github.com/astro-tools/paper-tle-divergence-atlas.git
cd paper-tle-divergence-atlas
make env
conda activate paper-tle-divergence-atlas
```

To build the manuscript:

```bash
make build
```

To re-run the underlying sweep (slow; requires GMAT):

```bash
make install-egm2008    # one-time: fetch NGA's EGM2008 coefficients and install
                        # the .cof file into $GMAT_ROOT/data/gravity/earth/.
                        # If NGA is unreachable, download
                        # https://earth-info.nga.mil/php/download.php?file=egm-08spherical
                        # manually and pass it with `--source <path>`.
make fetch-gmat-sw      # refresh src/static/SpaceWeather-All-v1.2.txt from
                        # CelesTrak if a newer observed-data horizon is needed.
                        # Optional — the committed snapshot covers the April
                        # 2026 corpus window. GMAT reads this via the
                        # FM.Drag.CSSISpaceWeatherFile script override.
make sweep              # or invoke sweep/run_sweep.py directly
```

## Local checks before pushing

```bash
ruff check sweep/ src/scripts/      # lint Python
ruff format --check sweep/ src/scripts/  # formatting
pytest                              # any tests under tests/
showyourwork build                  # full manuscript rebuild
```

CI runs the manuscript build on every PR. The sweep itself does **not** run in CI (GMAT is not available; sweep outputs are fetched from Zenodo).

## Branches and PRs

- Branch off `main`. Branch names use a short prefix:
  - `fix/<slug>` — bug fix in code or prose
  - `chore/<slug>` — tooling / hygiene
  - `feat/<slug>` — new analysis or figure
  - `docs/<slug>` — README, CONTRIBUTING, comments
- Open a PR against `main`. Reference any related issue with `Closes #N`.
- Squash-merge is the only merge method.

## Manuscript conventions

- LaTeX source lives in `src/tex/`. Edit `ms.tex` directly.
- Citations go in `src/tex/bib.bib`. Use `\citep{}` / `\citet{}` natbib style.
- Every figure in the manuscript has a generator script with a matching name in `src/scripts/`. E.g. `\includegraphics{figures/fig_sobol.pdf}` ↔ `src/scripts/fig_sobol.py`.
- Do **not** commit `src/tex/figures/` (generated) or `src/tex/ms.pdf`.

## Reproducibility expectations

This repository is structured around `showyourwork`'s reproducibility contract: every figure must be regenerable from a clean checkout. If you add a figure, you must add the script that generates it from inputs that are either committed (small data) or fetched (Zenodo deposit).
