.PHONY: help env fetch-tles fetch-satcat fetch-sw install-egm2008 build-corpus build smoke sweep figures clean

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
	@echo "                  src/data/tles_cache.parquet"
	@echo "  build        -- render PDF via showyourwork (uses Zenodo-cached outputs)"
	@echo "  smoke        -- run an N=8 sweep against the cached corpus (requires GMAT)"
	@echo "  sweep        -- run the gmat-sweep locally (requires GMAT; ~3 h on 8 cores)"
	@echo "  figures      -- regenerate figures from outputs/"
	@echo "  clean        -- remove generated artifacts (PDF, figures, snakemake state)"
	@echo ""
	@echo "After 'make env', activate with: conda activate paper-tle-divergence-atlas"

env:
	conda env create -f environment.yml

fetch-tles:
	python -m sweep.tle_pipeline fetch \
	    --window src/data/window.json \
	    --out src/data/tles_raw.parquet

fetch-satcat:
	python -m sweep.spacecraft_props fetch-satcat \
	    --out src/data/gcat_satcat.tsv

fetch-sw: src/data/sw_cache.parquet

src/data/sw_cache.parquet:
	python -m sweep.space_weather fetch \
	    --out $@

install-egm2008:
	python -m sweep.install_egm2008

build-corpus: src/data/sw_cache.parquet
	python -m sweep.tle_pipeline build \
	    --raw src/data/tles_raw.parquet \
	    --satcat src/data/gcat_satcat.tsv \
	    --out src/data/tles_cache.parquet

build:
	showyourwork build

smoke:
	python -m sweep.run_sweep \
	    --smoke \
	    --workers 4 \
	    --mission sweep/mission.script \
	    --tles src/data/tles_cache.parquet \
	    --sw-cache src/data/sw_cache.parquet \
	    --output-dir outputs/ \
	    --manifest sweep/manifest.jsonl

sweep:
	python -m sweep.run_sweep \
	    --mission sweep/mission.script \
	    --tles src/data/tles_cache.parquet \
	    --sw-cache src/data/sw_cache.parquet \
	    --output-dir outputs/ \
	    --manifest sweep/manifest.jsonl

figures:
	snakemake --cores 1 src/tex/figures

clean:
	rm -rf .snakemake .showyourwork src/tex/figures src/tex/ms.pdf
	rm -f src/tex/ms.aux src/tex/ms.log src/tex/ms.out
	rm -f src/tex/ms.bbl src/tex/ms.blg src/tex/ms.synctex.gz
	rm -f src/tex/ms.fdb_latexmk src/tex/ms.fls src/tex/ms.toc
