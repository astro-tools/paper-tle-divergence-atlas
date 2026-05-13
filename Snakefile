# Custom Snakemake rules for paper-tle-divergence-atlas.
#
# showyourwork handles the standard figure-and-manuscript DAG automatically.
# Rules here extend it with project-specific data pipelines.
#
# Day-2 will add:
#   rule fetch_tles    -- download Starlink TLEs and cache as parquet
#   rule build_pairs   -- construct (TLE_i, TLE_j) pairs with maneuver filter
# Day-3 will add:
#   rule sweep         -- gmat-sweep parallel run producing outputs/
#
# The sweep rule is local-only; CI fetches outputs/ from Zenodo via showyourwork's
# datasets mechanism configured in showyourwork.yml.

# Custom rule: fig_powerlaw_fits.py emits both the figure PDF and the
# auto-generated LaTeX table that ms.tex \input{}-s. showyourwork's
# auto-generated rule for figure scripts only knows about the PDF, so
# without this rule Snakemake has no producer for `tab_powerlaw.tex`
# and the LaTeX build fails on a missing \input target. Declaring both
# outputs in one rule keeps the fit logic single-source: the bootstrap
# resamples once and both artifacts fall out together.

rule fig_powerlaw_fits:
    input:
        all_runs="outputs/all_runs.parquet",
        script="src/scripts/fig_powerlaw_fits.py",
    output:
        pdf="src/tex/figures/fig_powerlaw_fits.pdf",
        table="src/tex/tables/tab_powerlaw.tex",
    conda:
        "environment.yml"
    shell:
        "python {input.script} "
        "--all-runs {input.all_runs} "
        "--out {output.pdf} "
        "--table-out {output.table}"
