# Changelog

All notable changes to `paper-tle-divergence-atlas` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-24

Reproducibility refresh of v0.1.0 under the gmat-sweep / gmat-run 0.6 toolchain. Per-row Δr against the v0.1.0 outputs is at floating-point-roundoff scale across every aggregate, summary, and diagnostic — no manuscript reissue, no arXiv v2, no science changes.

### Changed

- Sweep, CdA sensitivity arm, maneuver-threshold sensitivity arm, and `sweep/manifest.jsonl` regenerated end-to-end against gmat-sweep / gmat-run 0.6 (#68). The refreshed Zenodo deposit is published as version 0.2.0 under the existing concept DOI `10.5281/zenodo.20277028` (version DOI `10.5281/zenodo.20370326`).
- Pipeline upgraded to gmat-sweep / gmat-run 0.6 (#62), dropping the 0.4-era script-templating workaround.
- Phase-3 post-processing folded into the sweep via gmat-sweep's postprocess hook (#63); the hook is now re-applied on `Sweep` resume so partial-restart runs match cold-start runs bit-for-bit (#66).

### Added

- arXiv preprint badge and citation metadata pointing at [arXiv:2605.19850](https://arxiv.org/abs/2605.19850) (#58).
- Regression test pinning `run_id` preservation when aggregating the CdA sensitivity arm (#67).

### Removed

- `make figures` Makefile target (#71); figures are produced by `make build` via showyourwork's Snakemake DAG.

## [0.1.0] — 2026-05-18

Initial citeable release of *How long can you trust a Starlink TLE? An empirical comparison of SGP4 and high-fidelity propagation against operator-updated truth across a megaconstellation*.

### Added

- 24,641-pair locked sweep corpus across 501 Starlink satellites, stratified by altitude shell (540 / 550 / 560 km) and platform generation (v1.0, v1.5, v2-mini) over April 2026, with per-pair SGP4 and GMAT high-fidelity propagation evaluated against operator next-TLE proxy truth.
- High-fidelity force model: EGM2008 70×70 + Sun/Moon point-mass third bodies + NRLMSISE-00 drag + conical-shadow SRP, integrated with Runge–Kutta 8(9) to 10⁻¹² km tolerance.
- Truth-floor diagnostic (§2.1, Table 1) framing the 6-hour headline as floor-limited at ~1 km median against the next-TLE proxy.
- Dynamical-consistency caveat (§3.6.1) acknowledging the Brouwer mean-element mismatch between the SGP4 initial state and the high-fidelity propagator.
- Statistical-estimator section (§3.7.1): per-pair power-law fit on log-log with satellite-level bootstrap CIs, likelihood-ratio tests against the *k* = 1 / *k* = 2 nulls, and a mixed-effects parametric cross-check in Appendix B.4.
- H3 regression specification (§3.7.2): per-shell ANCOVA fit of the per-satellite SGP4 staleness coefficient against daily-observed F10.7, with the 81-day-centred robustness check.
- Sensitivity studies bundled with the main sweep: CdA × {0.8, 1.2} on the v2-mini cohort (Appendix B.2) and maneuver-threshold ±50 m / +200 m perturbations (Appendix B.3).
- §5.2 cohort-resolved view of the v2-mini long-Δ*t* H2 reading, with Table 8 reporting per-(shell × generation × Δ*t*) win fractions and quantifying the v2-mini majority-wins regime at 7 d across both populated shells.
- Per-cell power-law `(A, k)` atlas in Table 3, intended for downstream use as a benchmark target for enhanced-propagator work (SGP4-XP, differentiable SGP4, ML-residual correctors).
- `make bundle` target producing the canonical 17-file Zenodo deposit at the repository root.
- `make arxiv-tarball` target wrapping `showyourwork tarball` with a post-processor that strips dotfiles and showyourwork v0.4.3's root-level figure/table duplicates.
- Reproducibility surface: `make build` renders `ms.pdf` from a Zenodo-cached sweep bundle on a clean checkout with no local GMAT installation; `make sweep` reproduces the bundle from scratch given GMAT R2026a.
