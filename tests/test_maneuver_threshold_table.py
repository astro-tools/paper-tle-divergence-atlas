"""Unit tests for sweep.maneuver_threshold_table.

Synthetic frames let us assert on per-cell medians, bootstrap CI shape,
sign of the relative shifts, and the LaTeX renderer's output without
needing the real ~24,641-pair sweep frames on disk. The bootstrap
itself is exercised in isolation against a tiny per-sat frame so the
sat-level resample structure is testable without 1,000 draws.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sweep import maneuver_threshold_table as mtt

# --- Frame builders --------------------------------------------------------


def _runs(
    *,
    medians_km: dict[tuple[str, int], float],
    sats_per_cell: int = 8,
    pairs_per_sat: int = 4,
    run_id_offset: int = 0,
    epoch_offset_h: int = 0,
) -> pd.DataFrame:
    """Build a synthetic per-run frame with deterministic per-cell medians.

    `medians_km` maps (alt_shell, target_dt_sec) → median |Δr|_hifi in
    the cell. We emit `sats_per_cell × pairs_per_sat` rows per cell;
    all `pairs_per_sat` rows for a sat share its norad_id (so the
    sat-level bootstrap has structure to resample over). The numeric
    median over each cell equals `medians_km[cell]` exactly.

    Pair-key columns (norad_id, epoch_i, epoch_j) are filled in so the
    set-difference logic in `filter_to_corpus` and `assemble_populations`
    has something to match against. `epoch_offset_h` lets the caller
    derive disjoint frames from the same medians_km dict by shifting
    every epoch by a constant.
    """
    rows: list[dict] = []
    run_id = run_id_offset
    norad_id_offset = run_id_offset + 50_000
    for (shell, target_dt_sec), median_km in medians_km.items():
        for s in range(sats_per_cell):
            norad_id = norad_id_offset + s + (hash((shell, target_dt_sec)) % 1000)
            for p in range(pairs_per_sat):
                # alternate above and below the median so the cell-median lands exactly on `median_km`.
                offset = 0.01 if p % 2 == 0 else -0.01
                # last row for odd-length cells is exactly the median
                if p == pairs_per_sat - 1 and pairs_per_sat % 2 == 1:
                    offset = 0.0
                epoch_i = pd.Timestamp("2026-04-01T00:00:00Z") + pd.Timedelta(
                    hours=epoch_offset_h + run_id,
                )
                epoch_j = epoch_i + pd.Timedelta(seconds=int(target_dt_sec))
                rows.append(
                    {
                        "run_id": run_id,
                        "norad_id": norad_id,
                        "target_dt_sec": int(target_dt_sec),
                        "epoch_i": epoch_i,
                        "epoch_j": epoch_j,
                        "actual_dt_sec": float(target_dt_sec),
                        "alt_shell": shell,
                        "dr_hifi_km": median_km + offset,
                        "dr_sgp4_km": 0.5,
                        "generation": "v1.5",
                        "drag_area_m2": 2.0,
                        "srp_area_m2": 5.0,
                        "dry_mass_kg": 260.0,
                    },
                )
                run_id += 1
    return pd.DataFrame(rows)


# --- Tests: filter_to_corpus & assemble_populations ------------------------


class TestFilterToCorpus:
    def test_filter_returns_only_matching_pair_keys(self) -> None:
        # Build a synthetic baseline frame
        runs = _runs(medians_km={("550", 21_600): 1.5}, sats_per_cell=2, pairs_per_sat=2)
        # Corpus that only includes the first half of the runs
        corpus = runs.iloc[:2][list(mtt.PAIR_KEY)].copy()
        out = mtt.filter_to_corpus(runs, corpus)
        assert len(out) == 2
        # All output keys are in the corpus
        out_keys = set(map(tuple, out[list(mtt.PAIR_KEY)].itertuples(index=False, name=None)))
        corpus_keys = set(map(tuple, corpus.itertuples(index=False, name=None)))
        assert out_keys == corpus_keys

    def test_empty_corpus_returns_empty(self) -> None:
        runs = _runs(medians_km={("550", 21_600): 1.5}, sats_per_cell=2, pairs_per_sat=2)
        empty = pd.DataFrame(columns=list(mtt.PAIR_KEY))
        out = mtt.filter_to_corpus(runs, empty)
        assert out.empty


class TestAssemblePopulations:
    def _synthetic(self) -> dict[str, pd.DataFrame]:
        # all_runs (= 100 m baseline): 4 sats × 2 pairs each at one cell.
        all_runs = _runs(medians_km={("550", 21_600): 1.5}, sats_per_cell=4, pairs_per_sat=2)
        # 50 m corpus is a strict subset (first 4 rows).
        corpus_50m = all_runs.iloc[:4].copy()
        # 200 m corpus is all_runs + 2 augment pairs.
        augment_only = _runs(
            medians_km={("550", 21_600): 1.5},
            sats_per_cell=1,
            pairs_per_sat=2,
            run_id_offset=10_000,
            epoch_offset_h=999,
        )
        corpus_200m = pd.concat([all_runs, augment_only], ignore_index=True)
        return {
            "all_runs": all_runs,
            "augment": augment_only,
            "corpus_50m": corpus_50m,
            "corpus_200m": corpus_200m,
        }

    def test_populations_have_expected_sizes(self) -> None:
        s = self._synthetic()
        populations = mtt.assemble_populations(
            s["all_runs"],
            s["augment"],
            s["corpus_50m"],
            s["corpus_200m"],
        )
        assert len(populations["50m"]) == 4
        assert len(populations["100m"]) == len(s["all_runs"])
        # 200m = all_runs ∪ augment
        assert len(populations["200m"]) == len(s["all_runs"]) + len(s["augment"])

    def test_inclusion_violation_50m_raises(self) -> None:
        s = self._synthetic()
        # Inject a pair into the 50m corpus that isn't in all_runs.
        bad = s["corpus_50m"].copy()
        bad.iloc[0, bad.columns.get_loc("norad_id")] = 999_999
        with pytest.raises(SystemExit, match="50 m corpus contains pairs missing"):
            mtt.assemble_populations(s["all_runs"], s["augment"], bad, s["corpus_200m"])

    def test_inclusion_violation_200m_raises(self) -> None:
        s = self._synthetic()
        # Strip 200m corpus down to a subset of 100m — violates inclusion.
        bad_200 = s["all_runs"].iloc[:2].copy()
        with pytest.raises(SystemExit, match="100 m baseline corpus contains pairs missing"):
            mtt.assemble_populations(s["all_runs"], s["augment"], s["corpus_50m"], bad_200)


# --- Tests: bootstrap ------------------------------------------------------


class TestSatLevelBootstrap:
    def test_empty_returns_nans(self) -> None:
        empty = pd.DataFrame(columns=["norad_id", "dr_hifi_km"])
        med, lo, hi = mtt.sat_level_bootstrap_median_ci(empty)
        assert np.isnan(med) and np.isnan(lo) and np.isnan(hi)

    def test_point_estimate_is_median(self) -> None:
        pairs = pd.DataFrame(
            {
                "norad_id": [1, 1, 2, 2, 3, 3],
                "dr_hifi_km": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        )
        med, _, _ = mtt.sat_level_bootstrap_median_ci(pairs, n_draws=10)
        assert med == pytest.approx(3.5)  # median of [1,2,3,4,5,6]

    def test_ci_brackets_point_estimate_on_uniform_data(self) -> None:
        # 10 sats, 5 identical pairs each → every resample yields the same median.
        norad = np.repeat(np.arange(10), 5)
        values = np.full_like(norad, 2.5, dtype=float)
        pairs = pd.DataFrame({"norad_id": norad, "dr_hifi_km": values})
        med, lo, hi = mtt.sat_level_bootstrap_median_ci(pairs, n_draws=100)
        # No variability in the data → CI collapses to the point estimate.
        assert med == pytest.approx(2.5)
        assert lo == pytest.approx(2.5)
        assert hi == pytest.approx(2.5)

    def test_ci_widens_with_inter_sat_variability(self) -> None:
        # 10 sats, all-same within-sat values but very different between sats →
        # sat-level resample produces a wide CI.
        norad = np.repeat(np.arange(10), 3)
        values = np.repeat(np.arange(10, dtype=float), 3)
        pairs = pd.DataFrame({"norad_id": norad, "dr_hifi_km": values})
        med, lo, hi = mtt.sat_level_bootstrap_median_ci(pairs, n_draws=200, seed=0)
        # Median of [0..9 repeated 3×] is 4.5; bootstrap CI must straddle.
        assert med == pytest.approx(4.5)
        assert lo < med < hi
        # CI width should be at least a couple of units for this distribution.
        assert hi - lo > 1.0

    def test_seed_makes_bootstrap_deterministic(self) -> None:
        pairs = pd.DataFrame(
            {
                "norad_id": np.repeat(np.arange(5), 3),
                "dr_hifi_km": np.array([1.0, 2.0, 3.0] * 5) + np.arange(15) * 0.1,
            },
        )
        a = mtt.sat_level_bootstrap_median_ci(pairs, n_draws=50, seed=123)
        b = mtt.sat_level_bootstrap_median_ci(pairs, n_draws=50, seed=123)
        assert a == b


# --- Tests: per_cell_table + render ----------------------------------------


class TestPerCellTable:
    def _make_populations(
        self,
        baseline_medians: dict[tuple[str, int], float],
        m50_medians: dict[tuple[str, int], float],
        m200_medians: dict[tuple[str, int], float],
    ) -> dict[str, pd.DataFrame]:
        # Baseline frame populates every cell. 50m and 200m frames are
        # synthesized as standalone populations with the requested
        # medians, sharing the baseline's norad_ids for bootstrap structure.
        baseline = _runs(medians_km=baseline_medians, sats_per_cell=8, pairs_per_sat=4)
        # Build 50m by perturbing baseline's dr_hifi_km to the target cell median.
        pop_50m = baseline.copy()
        for cell, m in m50_medians.items():
            mask = (pop_50m["alt_shell"] == cell[0]) & (pop_50m["target_dt_sec"] == cell[1])
            base_med = baseline.loc[mask, "dr_hifi_km"].median()
            pop_50m.loc[mask, "dr_hifi_km"] = pop_50m.loc[mask, "dr_hifi_km"] - base_med + m
        pop_200m = baseline.copy()
        for cell, m in m200_medians.items():
            mask = (pop_200m["alt_shell"] == cell[0]) & (pop_200m["target_dt_sec"] == cell[1])
            base_med = baseline.loc[mask, "dr_hifi_km"].median()
            pop_200m.loc[mask, "dr_hifi_km"] = pop_200m.loc[mask, "dr_hifi_km"] - base_med + m
        return {"50m": pop_50m, "100m": baseline, "200m": pop_200m}

    def test_per_cell_columns(self) -> None:
        pop = self._make_populations(
            baseline_medians={("550", 21_600): 1.5, ("560", 86_400): 8.0},
            m50_medians={("550", 21_600): 1.45, ("560", 86_400): 7.5},
            m200_medians={("550", 21_600): 1.55, ("560", 86_400): 8.5},
        )
        per_cell = mtt.per_cell_table(pop, n_bootstrap=50)
        assert set(per_cell.columns) >= {
            "alt_shell",
            "target_dt_sec",
            "n_100m",
            "median_50m_km",
            "median_100m_km",
            "ci_lo_100m_km",
            "ci_hi_100m_km",
            "median_200m_km",
            "shift_50m",
            "shift_200m",
            "in_ci_50m",
            "in_ci_200m",
        }
        assert len(per_cell) == 2

    def test_per_cell_medians_match_input(self) -> None:
        pop = self._make_populations(
            baseline_medians={("550", 21_600): 1.5},
            m50_medians={("550", 21_600): 1.45},
            m200_medians={("550", 21_600): 1.55},
        )
        per_cell = mtt.per_cell_table(pop, n_bootstrap=20)
        row = per_cell.iloc[0]
        assert row["median_100m_km"] == pytest.approx(1.5, abs=1e-9)
        assert row["median_50m_km"] == pytest.approx(1.45, abs=1e-9)
        assert row["median_200m_km"] == pytest.approx(1.55, abs=1e-9)
        # Relative shifts
        assert row["shift_50m"] == pytest.approx((1.45 - 1.5) / 1.5, abs=1e-6)
        assert row["shift_200m"] == pytest.approx((1.55 - 1.5) / 1.5, abs=1e-6)

    def test_outside_ci_flag_when_shift_is_large(self) -> None:
        # A 50m population with median 10x larger than baseline must
        # land outside any reasonable bootstrap CI.
        pop = self._make_populations(
            baseline_medians={("550", 21_600): 1.5},
            m50_medians={("550", 21_600): 15.0},
            m200_medians={("550", 21_600): 1.5},
        )
        per_cell = mtt.per_cell_table(pop, n_bootstrap=100)
        assert bool(per_cell.iloc[0]["in_ci_50m"]) is False
        assert bool(per_cell.iloc[0]["in_ci_200m"]) is True


class TestRenderThresholdTable:
    def _per_cell(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "alt_shell": "550",
                    "target_dt_sec": 21_600,
                    "n_100m": 100,
                    "median_50m_km": 1.45,
                    "median_100m_km": 1.5,
                    "ci_lo_100m_km": 1.4,
                    "ci_hi_100m_km": 1.6,
                    "median_200m_km": 1.55,
                    "shift_50m": (1.45 - 1.5) / 1.5,
                    "shift_200m": (1.55 - 1.5) / 1.5,
                    "in_ci_50m": True,
                    "in_ci_200m": True,
                },
                {
                    "alt_shell": "560",
                    "target_dt_sec": 604_800,
                    "n_100m": 50,
                    "median_50m_km": 100.0,
                    "median_100m_km": 130.0,
                    "ci_lo_100m_km": 120.0,
                    "ci_hi_100m_km": 140.0,
                    "median_200m_km": 145.0,
                    "shift_50m": (100.0 - 130.0) / 130.0,
                    "shift_200m": (145.0 - 130.0) / 130.0,
                    "in_ci_50m": False,
                    "in_ci_200m": False,
                },
            ],
        )

    def test_table_renders_booktabs(self) -> None:
        latex = mtt.render_threshold_table(self._per_cell())
        assert "\\toprule" in latex
        assert "\\bottomrule" in latex
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex

    def test_baseline_median_with_ci_appears(self) -> None:
        latex = mtt.render_threshold_table(self._per_cell())
        # 550 × 6 h row: median 1.50 with 95% CI [1.40, 1.60]
        assert "1.50 [1.40, 1.60]" in latex
        # 560 × 7 d row: median 130.00 with CI [120.00, 140.00]
        assert "130.00 [120.00, 140.00]" in latex

    def test_shift_signs_correct(self) -> None:
        latex = mtt.render_threshold_table(self._per_cell())
        # 50m shift on 550×6h = (1.45-1.5)/1.5 = -3.3%
        assert "-3.3" in latex
        # 200m shift on 550×6h = (1.55-1.5)/1.5 = +3.3%
        assert "+3.3" in latex
        # 50m shift on 560×7d = (100-130)/130 = -23.1%
        assert "-23.1" in latex

    def test_outside_ci_cells_marked(self) -> None:
        latex = mtt.render_threshold_table(self._per_cell())
        # The 560×7d row's 50m and 200m shifts are outside the CI;
        # the cells must carry the dagger superscript.
        # Row layout: "560 & 7 h & ... & -23.1...\\dagger & ...\\dagger"
        assert "\\dagger" in latex
        # The 550×6h row's shifts are inside the CI; ensure dagger doesn't
        # appear in that row in a separate test for safety. Walk lines.
        for line in latex.splitlines():
            if line.startswith("550 & 6 h"):
                assert "\\dagger" not in line

    def test_midrule_between_shells(self) -> None:
        latex = mtt.render_threshold_table(self._per_cell())
        # The two rows in the fixture are different shells, so a midrule
        # must separate them: \midrule before data + \midrule between shells.
        assert latex.count("\\midrule") >= 2


class TestRenderRejectionsTable:
    def _rejection_counts(self) -> dict:
        return {
            "cells": [
                {
                    "alt_shell": "540",
                    "target_dt_sec": 21_600,
                    "n_candidates": 100,
                    "n_survivors": 90,
                    "n_rejected": 10,
                },
                {
                    "alt_shell": "550",
                    "target_dt_sec": 21_600,
                    "n_candidates": 200,
                    "n_survivors": 160,
                    "n_rejected": 40,
                },
            ],
            "totals": {"n_candidates": 300, "n_survivors": 250, "n_rejected": 50},
        }

    def test_rejections_table_includes_totals_row(self) -> None:
        latex = mtt.render_rejections_table(self._rejection_counts())
        assert "Total" in latex
        # Total candidates 300 should appear, rendered as 300 (no need for thousands sep here)
        assert "300" in latex
        # Rejection % for 550 × 6h = 40/200 = 20.0%
        assert "20.0\\%" in latex

    def test_rejections_table_renders_booktabs(self) -> None:
        latex = mtt.render_rejections_table(self._rejection_counts())
        assert "\\begin{tabular}" in latex
        assert "\\bottomrule" in latex


# --- Tests: build (end-to-end smoke) ---------------------------------------


class TestBuild:
    @pytest.fixture
    def synthetic_inputs(self, tmp_path: Path) -> dict[str, Path]:
        baseline_medians = {("550", 21_600): 1.5, ("560", 86_400): 8.0}
        all_runs = _runs(medians_km=baseline_medians, sats_per_cell=6, pairs_per_sat=3)
        augment = _runs(
            medians_km=baseline_medians,
            sats_per_cell=2,
            pairs_per_sat=2,
            run_id_offset=10_000,
            epoch_offset_h=999,
        )
        # 50m corpus = first half of all_runs (subset).
        corpus_50m = all_runs.iloc[: len(all_runs) // 2].copy()
        # 200m corpus = all_runs ∪ augment.
        corpus_200m = pd.concat([all_runs, augment], ignore_index=True)
        rejection_counts = {
            "sma_threshold_km": 0.1,
            "cells": [
                {
                    "alt_shell": "550",
                    "target_dt_sec": 21_600,
                    "n_candidates": 100,
                    "n_survivors": 90,
                    "n_rejected": 10,
                },
            ],
            "totals": {"n_candidates": 100, "n_survivors": 90, "n_rejected": 10},
        }

        paths = {
            "all_runs": tmp_path / "all_runs.parquet",
            "augment": tmp_path / "augment.parquet",
            "corpus_50m": tmp_path / "tles_cache_50m.parquet",
            "corpus_200m": tmp_path / "tles_cache_200m.parquet",
            "rejection_counts": tmp_path / "rejection_counts.json",
        }
        all_runs.to_parquet(paths["all_runs"], index=False)
        augment.to_parquet(paths["augment"], index=False)
        corpus_50m.to_parquet(paths["corpus_50m"], index=False)
        corpus_200m.to_parquet(paths["corpus_200m"], index=False)
        paths["rejection_counts"].write_text(json.dumps(rejection_counts))
        return paths

    def test_build_returns_two_tables_and_summary(
        self,
        synthetic_inputs: dict[str, Path],
    ) -> None:
        rejections_tex, threshold_tex, summary = mtt.build(
            synthetic_inputs["all_runs"],
            synthetic_inputs["augment"],
            synthetic_inputs["corpus_50m"],
            synthetic_inputs["corpus_200m"],
            synthetic_inputs["rejection_counts"],
            n_bootstrap=20,
        )
        assert "\\begin{tabular}" in rejections_tex
        assert "\\begin{tabular}" in threshold_tex
        assert summary["n_cells"] == 2
        assert summary["n_pairs"]["100m"] > summary["n_pairs"]["50m"]
        assert summary["n_pairs"]["200m"] > summary["n_pairs"]["100m"]
        assert (
            summary["n_pairs"]["augment"] == summary["n_pairs"]["200m"] - summary["n_pairs"]["100m"]
        )

    def test_summary_pooled_medians_present(
        self,
        synthetic_inputs: dict[str, Path],
    ) -> None:
        _, _, summary = mtt.build(
            synthetic_inputs["all_runs"],
            synthetic_inputs["augment"],
            synthetic_inputs["corpus_50m"],
            synthetic_inputs["corpus_200m"],
            synthetic_inputs["rejection_counts"],
            n_bootstrap=20,
        )
        for threshold in ("50m", "100m", "200m"):
            assert threshold in summary["pooled_median_km"]
            assert np.isfinite(summary["pooled_median_km"][threshold])


class TestFmtShift:
    def test_plus_sign_present(self) -> None:
        assert mtt._fmt_shift(0.123) == "+12.3\\%"

    def test_minus_sign_present(self) -> None:
        assert mtt._fmt_shift(-0.123) == "-12.3\\%"

    def test_nan_renders_dash(self) -> None:
        assert mtt._fmt_shift(float("nan")) == "---"
