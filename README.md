# paper-tle-divergence-atlas

[![build](https://github.com/astro-tools/paper-tle-divergence-atlas/actions/workflows/build.yml/badge.svg)](https://github.com/astro-tools/paper-tle-divergence-atlas/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> *How long can you trust a Starlink TLE? An empirical comparison of SGP4 and high-fidelity propagation against operator-updated truth across a megaconstellation.*

Author: Dimitrije Jankovic (Independent researcher — astro-tools)
Status: **In progress** (pre-deposit draft)

## What this repository is

The full source for an open-science paper on TLE propagation accuracy in the Starlink megaconstellation. The repository contains:

- The manuscript LaTeX source (`src/tex/`)
- The figure-generation scripts (`src/scripts/`)
- The GMAT mission script and the gmat-sweep driver that produce all underlying data (`sweep/`)
- A reproducibility manifest (`sweep/manifest.jsonl`) linking the figures back to a Zenodo-archived sweep bundle

The full data product (sweep outputs, ~50–100 MB of Parquet) is deposited on Zenodo and fetched automatically by the build.

## Reproducing the paper

### Render the PDF (no GMAT required)

```bash
# Once: install conda-forge & showyourwork
make env
conda activate paper-tle-divergence-atlas

# Build (fetches the Zenodo data bundle, regenerates figures, compiles LaTeX)
make build
```

The PDF lands at `ms.pdf` (repo root).

### Reproduce the sweep from scratch (requires GMAT)

Bypass the Zenodo cache and recompute every parquet:

```bash
# In addition to the build env, install a local GMAT R2026a
# (https://gmat.gsfc.nasa.gov) and set GMAT_ROOT.
export GMAT_ROOT=~/gmat-R2026a

make sweep              # ~10 h wall on 8 cores over the 24,641-pair corpus
make aggregate          # concat outputs/run_*.parquet → outputs/all_runs.parquet
make sweep-stats        # per-bucket / per-(shell, gen) medians + manifest failures
make diagnostics        # outputs/_diagnostic_sweep_scatter.png — sanity scatter
```

The sweep is pausable: a Ctrl-C, sleep, or reboot leaves a partial
`sweep/manifest.jsonl`; re-running `make sweep` automatically resumes
and dispatches only the failed/missing runs. To start over from
scratch (different corpus, or just a clean slate), delete
`sweep/manifest.jsonl` and the contents of `outputs/` first.

## Compute requirements

| Stage | Hardware | Time |
|---|---|---|
| Render PDF from Zenodo cache | Any laptop | ~5 min |
| Full sweep (3,000 runs) | 8-core workstation, GMAT installed | ~3 h |
| Extended sensitivity run | Same | ~6 h additional |

## Data and code availability

- **Code:** this repository (MIT licensed).
- **Sweep outputs:** Zenodo concept DOI [TBD before v0.1.0 release].
- **Input data:** Starlink TLEs from CelesTrak (cached in `src/static/`); solar activity from CelesTrak's space weather file.
- **Spacecraft properties:** per-NORAD-ID dry mass and structural span are taken from Jonathan McDowell's *General Catalog of Artificial Space Objects* ([GCAT](https://planet4589.org/space/gcat/)), with a snapshot of the relevant subset cached in `src/static/`. Cite McDowell, J. C. 2020, AJ, 159, 5.

## Citation

```bibtex
@misc{jankovic_tle_divergence_atlas,
  author = {Jankovic, Dimitrije},
  title = {How long can you trust a Starlink TLE? An empirical comparison of SGP4 and high-fidelity propagation against operator-updated truth across a megaconstellation},
  year = {2026},
  doi = {TBD},
  url = {https://github.com/astro-tools/paper-tle-divergence-atlas}
}
```

DOI minted at v0.1.0 release.

## Methodology — quick reference

For full detail see Section 4 of the manuscript. In short, for each pair of consecutive Starlink TLEs (TLE_i, TLE_j):

1. Initialize a state at t_i from TLE_i (SGP4 internal evaluation at Δt=0, then TEME→J2000).
2. Propagate forward to t_j with two propagators in parallel: SGP4-from-TLE_i and GMAT high-fidelity (EGM2008 70×70 + Sun/Moon/SRP + MSISE-90).
3. Compare both predicted states against the operator's next-TLE truth (SGP4(TLE_j, Δt=0)).
4. Aggregate position errors by orbital regime, time since epoch, and solar activity.

This methodology follows Vallado & Cefola (2012); the contribution is constellation-scale application with open code and data.

## Issues and feedback

Bug reports and reproducibility issues: please open an issue. PRs welcome for figure improvements, additional sensitivity analyses, or methodological refinements. See `CONTRIBUTING.md` for workflow.
