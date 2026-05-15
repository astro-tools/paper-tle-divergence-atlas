"""Unit tests for sweep.sensitivity_subset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sweep.sensitivity_subset import (
    DEFAULT_SEED,
    DEFAULT_TARGET_SIZE,
    STRATA_COLUMNS,
    _proportional_allocation,
    main,
    select_subset,
    write_pair_ids,
)

_BASE_EPOCH = pd.Timestamp("2026-04-01T00:00:00Z")


def _synthetic_corpus(
    cell_sizes: dict[tuple[str, str, int], int],
    *,
    base_epoch: pd.Timestamp = _BASE_EPOCH,
) -> pd.DataFrame:
    """Build a corpus DataFrame with `cell_sizes` rows per stratum cell.

    Row index is reset so it matches the run_id convention.
    """
    rows: list[dict] = []
    norad_counter = 50000
    for (shell, gen, target_dt), n in cell_sizes.items():
        for k in range(n):
            rows.append(
                {
                    "alt_shell": shell,
                    "generation": gen,
                    "target_dt_sec": int(target_dt),
                    "norad_id": norad_counter + k,
                    "epoch_i": base_epoch + pd.Timedelta(hours=k),
                }
            )
        norad_counter += 1000  # keep norad_ids distinct across cells
    return pd.DataFrame(rows).reset_index(drop=True)


def test_proportional_allocation_sums_to_target():
    counts = pd.Series({"a": 7000, "b": 2000, "c": 1000})
    quotas = _proportional_allocation(counts, target_size=100)
    assert int(quotas.sum()) == 100


def test_proportional_allocation_respects_cell_headroom():
    # Cell "small" has only 3 rows but a naive proportional share at
    # target=100 would request 7. The allocator must clip and
    # redistribute.
    counts = pd.Series({"big": 60, "small": 3})
    quotas = _proportional_allocation(counts, target_size=50)
    assert quotas["small"] <= 3
    assert int(quotas.sum()) == 50


def test_proportional_allocation_zero_cells_get_zero():
    counts = pd.Series({"a": 100, "empty": 0, "b": 50})
    quotas = _proportional_allocation(counts, target_size=30)
    assert quotas["empty"] == 0
    assert int(quotas.sum()) == 30


def test_proportional_allocation_target_clipped_to_total():
    counts = pd.Series({"a": 5, "b": 3})
    quotas = _proportional_allocation(counts, target_size=100)
    assert int(quotas.sum()) == 8
    assert quotas["a"] == 5
    assert quotas["b"] == 3


def test_select_subset_returns_target_size_when_corpus_is_large_enough():
    corpus = _synthetic_corpus(
        {
            ("540", "v1.5", 21600): 200,
            ("550", "v1.5", 21600): 150,
            ("550", "v2-mini", 21600): 100,
            ("560", "v1.5", 21600): 100,
            ("560", "v2-mini", 21600): 50,
        }
    )
    run_ids = select_subset(corpus, target_size=100, seed=DEFAULT_SEED)
    assert len(run_ids) == 100


def test_select_subset_is_sorted_ascending():
    corpus = _synthetic_corpus({("550", "v1.5", 21600): 200})
    run_ids = select_subset(corpus, target_size=50, seed=DEFAULT_SEED)
    assert list(run_ids) == sorted(run_ids)


def test_select_subset_is_deterministic_under_fixed_seed():
    corpus = _synthetic_corpus(
        {
            ("540", "v1.5", 21600): 80,
            ("550", "v1.5", 21600): 80,
            ("560", "v2-mini", 86400): 80,
        }
    )
    a = select_subset(corpus, target_size=60, seed=DEFAULT_SEED)
    b = select_subset(corpus, target_size=60, seed=DEFAULT_SEED)
    assert np.array_equal(a, b)


def test_select_subset_changes_with_seed():
    corpus = _synthetic_corpus({("550", "v1.5", 21600): 200})
    a = select_subset(corpus, target_size=50, seed=1)
    b = select_subset(corpus, target_size=50, seed=2)
    assert not np.array_equal(a, b)


def test_select_subset_run_ids_are_valid_corpus_indices():
    corpus = _synthetic_corpus(
        {
            ("540", "v1.5", 21600): 100,
            ("550", "v2-mini", 86400): 100,
        }
    )
    run_ids = select_subset(corpus, target_size=80, seed=DEFAULT_SEED)
    assert run_ids.min() >= 0
    assert run_ids.max() < len(corpus)
    # All run_ids are unique (sampling without replacement, across cells).
    assert len(np.unique(run_ids)) == len(run_ids)


def test_select_subset_cell_quotas_match_strata():
    corpus = _synthetic_corpus(
        {
            ("540", "v1.5", 21600): 400,
            ("550", "v1.5", 21600): 400,
            ("560", "v2-mini", 86400): 200,
        }
    )
    run_ids = select_subset(corpus, target_size=100, seed=DEFAULT_SEED)
    chosen_strata = (
        corpus.loc[run_ids, list(STRATA_COLUMNS)].apply(tuple, axis=1).value_counts().sort_index()
    )
    # Each cell receives at least floor(proportional share); the
    # exact split is governed by the largest-remainder allocator and
    # tested separately above. Here we just confirm that every
    # populated cell is represented and counts sum to target_size.
    assert chosen_strata.sum() == 100
    assert set(chosen_strata.index) == {
        ("540", "v1.5", 21600),
        ("550", "v1.5", 21600),
        ("560", "v2-mini", 86400),
    }


def test_select_subset_rejects_non_range_index():
    corpus = _synthetic_corpus({("550", "v1.5", 21600): 50})
    corpus.index = corpus.index + 1000
    with pytest.raises(ValueError, match="0-based RangeIndex"):
        select_subset(corpus, target_size=10)


def test_select_subset_rejects_missing_strata_columns():
    corpus = _synthetic_corpus({("550", "v1.5", 21600): 50}).drop(columns=["generation"])
    with pytest.raises(KeyError, match="generation"):
        select_subset(corpus, target_size=10)


def test_write_pair_ids_round_trip(tmp_path: Path):
    ids = np.array([1, 7, 42, 100], dtype=np.int64)
    out = tmp_path / "subdir" / "ids.txt"
    write_pair_ids(ids, out)
    assert out.read_text() == "1\n7\n42\n100\n"


def test_main_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus = _synthetic_corpus(
        {
            ("540", "v1.5", 21600): 100,
            ("550", "v1.5", 21600): 100,
            ("560", "v2-mini", 86400): 100,
        }
    )
    tles_path = tmp_path / "tles_cache.parquet"
    corpus.to_parquet(tles_path, index=False)

    out_path = tmp_path / "outputs" / "sensitivity_subset_pair_ids.txt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sensitivity_subset",
            "--tles",
            str(tles_path),
            "--out",
            str(out_path),
            "--target-size",
            "60",
            "--seed",
            str(DEFAULT_SEED),
        ],
    )
    assert main() == 0

    written = [int(x) for x in out_path.read_text().splitlines()]
    assert len(written) == 60
    assert written == sorted(written)
    assert len(set(written)) == 60
    assert min(written) >= 0
    assert max(written) < len(corpus)


def test_defaults_match_issue_30_spec():
    # Locked-in by issue #30; #28 and #31 reuse this exact subset.
    assert DEFAULT_SEED == 20260513
    assert DEFAULT_TARGET_SIZE == 1_000


def test_default_target_size_against_real_corpus_shape():
    # The real corpus has 24,641 pairs; target_size=1,000 is the
    # contract with #28 and #31. This synthetic case mirrors the corpus
    # shape at lower magnitude to confirm the allocator does not
    # under-fill when the corpus dwarfs the target.
    corpus = _synthetic_corpus(
        {
            ("540", "v1.5", 21600): 7479,
            ("550", "v1.5", 21600): 2586,
            ("550", "v2-mini", 21600): 6908,
            ("560", "v1.5", 21600): 4629,
            ("560", "v2-mini", 21600): 2937,
        }
    )
    run_ids = select_subset(corpus, target_size=1000, seed=DEFAULT_SEED)
    assert len(run_ids) == 1000
