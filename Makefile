.PHONY: help env fetch-tles fetch-satcat fetch-sw install-egm2008 build-corpus build-maneuver-jumps build-sensitivity-subset build smoke sweep aggregate sweep-stats diagnostics figures clean

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

figures:
	snakemake --cores 1 src/tex/figures

clean:
	rm -rf .snakemake .showyourwork src/tex/figures ms.pdf
	rm -f ms.aux ms.log ms.out ms.bbl ms.blg ms.synctex.gz ms.fdb_latexmk ms.fls ms.toc
