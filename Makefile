.PHONY: help env build sweep figures clean

help:
	@echo "Targets:"
	@echo "  env      -- create the conda env (one-time setup; env stored at ~/miniforge3/envs/)"
	@echo "  build    -- render PDF via showyourwork (uses Zenodo-cached outputs)"
	@echo "  sweep    -- run the gmat-sweep locally (requires GMAT; ~3 h on 8 cores)"
	@echo "  figures  -- regenerate figures from outputs/"
	@echo "  clean    -- remove generated artifacts (PDF, figures, snakemake state)"
	@echo ""
	@echo "After 'make env', activate with: conda activate paper-tle-divergence-atlas"

env:
	conda env create -f environment.yml

build:
	showyourwork build

sweep:
	python sweep/run_sweep.py \
	    --mission sweep/mission.script \
	    --tles src/data/tles_cache.parquet \
	    --output-dir outputs/ \
	    --manifest sweep/manifest.jsonl

figures:
	snakemake --cores 1 src/tex/figures

clean:
	rm -rf .snakemake .showyourwork src/tex/figures src/tex/ms.pdf
	rm -f src/tex/ms.aux src/tex/ms.log src/tex/ms.out
	rm -f src/tex/ms.bbl src/tex/ms.blg src/tex/ms.synctex.gz
	rm -f src/tex/ms.fdb_latexmk src/tex/ms.fls src/tex/ms.toc
