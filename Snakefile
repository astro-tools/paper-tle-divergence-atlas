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

# Currently empty -- the manuscript renders without any custom data pipeline.
