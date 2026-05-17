.PHONY: help env fetch-tles fetch-satcat fetch-sw install-egm2008 build-corpus build-maneuver-jumps build-rejection-counts build-sensitivity-subset build smoke sweep aggregate sweep-stats diagnostics cda-sensitivity cda-sensitivity-table maneuver-threshold-sensitivity maneuver-threshold-table h3-regression propagator-wins figures clean

help:
	@echo "Targets:"
	@echo "  env          -- create the conda env (one-time setup; env stored at ~/miniforge3/envs/)"
	@echo "  fetch-tles   -- one-time Space-Track fetch into src/data/tles_raw.parquet"
	@echo "                  Requires SPACETRACK_USERNAME and SPACETRACK_PASSWORD env vars."
	@echo "  fetch-satcat -- one-time fetch of McDowell's GCAT satcat.tsv (~18 MB) used for"
	@echo "                  per-NORAD-ID dry mass and span."
	@echo "  fetch-sw     -- one-time fetch of CelesTrak's space-weather file used for"
	@echo "                  per-run F10.7 / Ap annotations."
	@echo "  install-egm2008 -- one-time download + convert of NGA's EGM2008 coefficient"
	@echo "                     file into \$$GMAT_ROOT/data/gravity/earth/EGM2008.cof"
	@echo "                     (idempotent; safe to re-run)."
	@echo "  build-corpus -- pairs + maneuver filter + stratified sample + GCAT props →"
	@echo "                  src/static/tles_cache.parquet"
	@echo "  build-maneuver-jumps -- per-consecutive-pair |Δa| from the raw cache →"
	@echo "                          src/static/maneuver_jumps.parquet (F8 input)."
	@echo "  build-rejection-counts -- per-(shell × Δt) candidate-vs-surviving pair"
	@echo "                            counts at the 100 m baseline threshold →"
	@echo "                            src/static/maneuver_rejection_counts.json"
	@echo "                            (Appendix A rejection table input)."
	@echo "  build-selection-stats -- inter-TLE intervals + per-sat longest-gap series for"
	@echo "                           the 501-sat corpus → src/static/selection_stats.parquet"
	@echo "                           (selection-effect appendix figure input)."
	@echo "  build-sensitivity-subset -- 1,000-pair stratified subset of the corpus →"
	@echo "                              src/static/sensitivity_subset_pair_ids.txt"
	@echo "                              (consumed by #28 CdA and #31 maneuver-threshold)."
	@echo "  build        -- render PDF via showyourwork (uses Zenodo-cached outputs)"
	@echo "  smoke        -- run an N=8 sweep against the cached corpus (requires GMAT)"
	@echo "  sweep        -- run the gmat-sweep locally (requires GMAT; ~10 h on 8 cores)"
	@echo "  aggregate    -- concat outputs/run_*.parquet + join corpus cols →"
	@echo "                  outputs/all_runs.parquet"
	@echo "  sweep-stats  -- per-bucket / per-(shell,gen) median+IQR and manifest"
	@echo "                  failure tally → outputs/sweep_stats.txt"
	@echo "  diagnostics  -- render outputs/_diagnostic_sweep_scatter.png"
	@echo "  cda-sensitivity -- run the v2-mini CdA ±20% sensitivity sub-sweep"
	@echo "                     (~800 GMAT runs, ~2 h on 8 cores; requires GMAT)."
	@echo "  cda-sensitivity-table -- emit src/tex/tables/tab_cda_sensitivity.tex"
	@echo "                           and outputs/cda_sensitivity_summary.json from"
	@echo "                           the three CdA frames."
	@echo "  maneuver-threshold-sensitivity -- run the 200 m augment sub-sweep"
	@echo "                                    (~2,500 GMAT runs, ~1-3 h on 8 cores;"
	@echo "                                    requires GMAT). Requires the 50 m and"
	@echo "                                    200 m candidate corpora to be built first."
	@echo "  maneuver-threshold-table -- emit src/tex/tables/tab_maneuver_threshold.tex,"
	@echo "                              src/tex/tables/tab_maneuver_rejections.tex, and"
	@echo "                              outputs/maneuver_threshold_summary.json from the"
	@echo "                              baseline, augment, corpora, and rejection-count"
	@echo "                              JSON. No GMAT required."
	@echo "  h3-regression -- per-(shell x gen) H3 OLS fits with sat-level bootstrap CIs"
	@echo "                   on the daily-mean and 81-day-centred F10.7 predictors,"
	@echo "                   emitted as outputs/h3_regression.json. Run by"
	@echo "                   fig_solar_modulation.py internally; this target is for"
	@echo "                   local inspection only."
	@echo "  propagator-wins -- per-(shell x Δt) hi-fid-vs-SGP4 win fractions with"
	@echo "                     sat-level bootstrap CIs on 3D L2 and along-track,"
	@echo "                     emitted as outputs/propagator_wins.json plus the"
	@echo "                     §4.2 main-body table src/tex/tables/tab_propagator_wins.tex."
	@echo "  figures      -- regenerate figures from outputs/"
	@echo "  clean        -- remove generated artifacts (PDF, figures, snakemake state)"
	@echo ""
	@echo "After 'make env', activate with: conda activate paper-tle-divergence-atlas"

env:
	conda env create -f environment.yml

fetch-tles:
	python -m sweep.tle_pipeline fetch \
	    --window src/static/window.json \
	    --out src/data/tles_raw.parquet

fetch-satcat:
	python -m sweep.spacecraft_props fetch-satcat \
	    --out src/data/gcat_satcat.tsv

fetch-sw: src/static/sw_cache.parquet

src/static/sw_cache.parquet:
	python -m sweep.space_weather fetch \
	    --out $@

install-egm2008:
	python -m sweep.install_egm2008

build-corpus: src/static/sw_cache.parquet
	python -m sweep.tle_pipeline build \
	    --raw src/data/tles_raw.parquet \
	    --satcat src/data/gcat_satcat.tsv \
	    --out src/static/tles_cache.parquet

build-maneuver-jumps:
	python -m sweep.tle_pipeline maneuver-jumps \
	    --raw src/data/tles_raw.parquet \
	    --out src/static/maneuver_jumps.parquet

build-rejection-counts:
	python -m sweep.tle_pipeline rejection-counts \
	    --raw src/data/tles_raw.parquet \
	    --out src/static/maneuver_rejection_counts.json

build-selection-stats:
	python -m sweep.tle_pipeline selection-stats \
	    --raw src/data/tles_raw.parquet \
	    --cache src/static/tles_cache.parquet \
	    --out src/static/selection_stats.parquet

build-sensitivity-subset:
	python -m sweep.sensitivity_subset \
	    --tles src/static/tles_cache.parquet \
	    --out src/static/sensitivity_subset_pair_ids.txt

build:
	showyourwork build

smoke:
	python -m sweep.run_sweep \
	    --smoke \
	    --workers 4 \
	    --mission sweep/mission.script \
	    --tles src/static/tles_cache.parquet \
	    --sw-cache src/static/sw_cache.parquet \
	    --output-dir outputs/ \
	    --manifest sweep/manifest.jsonl

sweep:
	python -m sweep.run_sweep \
	    --mission sweep/mission.script \
	    --tles src/static/tles_cache.parquet \
	    --sw-cache src/static/sw_cache.parquet \
	    --output-dir outputs/ \
	    --manifest sweep/manifest.jsonl

aggregate:
	python -m sweep.aggregate \
	    --output-dir outputs/ \
	    --tles src/static/tles_cache.parquet \
	    --manifest sweep/manifest.jsonl \
	    --out outputs/all_runs.parquet

sweep-stats:
	python -m sweep.sweep_stats \
	    --all-runs outputs/all_runs.parquet \
	    --manifest sweep/manifest.jsonl \
	    --out outputs/sweep_stats.txt

diagnostics:
	python src/scripts/_diagnostic_sweep_scatter.py \
	    --all-runs outputs/all_runs.parquet \
	    --out outputs/_diagnostic_sweep_scatter.png

cda-sensitivity:
	python -m sweep.cda_sensitivity \
	    --mission sweep/mission.script \
	    --tles src/static/tles_cache.parquet \
	    --subset-ids src/static/sensitivity_subset_pair_ids.txt \
	    --sw-cache src/static/sw_cache.parquet \
	    --output-root outputs/_cda_sensitivity \
	    --output-dir outputs

cda-sensitivity-table:
	python -m sweep.cda_sensitivity_table \
	    --all-runs outputs/all_runs.parquet \
	    --cda-low outputs/all_runs_cda_low.parquet \
	    --cda-high outputs/all_runs_cda_high.parquet \
	    --subset-ids src/static/sensitivity_subset_pair_ids.txt \
	    --table-out src/tex/tables/tab_cda_sensitivity.tex \
	    --summary-out outputs/cda_sensitivity_summary.json

maneuver-threshold-sensitivity:
	python -m sweep.maneuver_threshold_sensitivity \
	    --mission sweep/mission.script \
	    --baseline-corpus src/static/tles_cache.parquet \
	    --augment-corpus outputs/tles_cache_200m.parquet \
	    --sw-cache src/static/sw_cache.parquet \
	    --output-root outputs/_maneuver_threshold_sensitivity \
	    --out outputs/all_runs_maneuver_augment.parquet

maneuver-threshold-table:
	python -m sweep.maneuver_threshold_table \
	    --all-runs outputs/all_runs.parquet \
	    --augment outputs/all_runs_maneuver_augment.parquet \
	    --corpus-50m outputs/tles_cache_50m.parquet \
	    --corpus-200m outputs/tles_cache_200m.parquet \
	    --rejection-counts src/static/maneuver_rejection_counts.json \
	    --threshold-table-out src/tex/tables/tab_maneuver_threshold.tex \
	    --rejections-table-out src/tex/tables/tab_maneuver_rejections.tex \
	    --summary-out outputs/maneuver_threshold_summary.json

h3-regression:
	python src/scripts/_h3_regression.py \
	    --all-runs outputs/all_runs.parquet \
	    --sw-cache src/static/sw_cache.parquet \
	    --out outputs/h3_regression.json

propagator-wins:
	python src/scripts/_propagator_wins.py \
	    --all-runs outputs/all_runs.parquet \
	    --json-out outputs/propagator_wins.json \
	    --table-out src/tex/tables/tab_propagator_wins.tex

figures:
	snakemake --cores 1 src/tex/figures

clean:
	rm -rf .snakemake .showyourwork src/tex/figures ms.pdf
	rm -f ms.aux ms.log ms.out ms.bbl ms.blg ms.synctex.gz ms.fdb_latexmk ms.fls ms.toc
