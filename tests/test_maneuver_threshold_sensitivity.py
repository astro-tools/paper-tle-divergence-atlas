"""Unit tests for sweep.maneuver_threshold_sensitivity.

GMAT-touching code paths are exercised end-to-end through `make
maneuver-threshold-sensitivity`, not pytest. Tested here:

- `compute_augment` computes the 200 m \\ 100 m set difference by the
  ``(norad_id, epoch_i, epoch_j)`` key and preserves the high-threshold
  corpus's full schema (including the GCAT-derived spacecraft-property
  columns the GMAT driver consumes).
- `_assign_augment_run_ids` returns the augment frame with run_ids
  starting at ``AUGMENT_RUN_ID_OFFSET`` and deterministic sort order, so
  reruns against the same corpora produce the same outputs.
"""

from __future__ import annotations

import pandas as pd

from sweep.maneuver_threshold_sensitivity import (
    AUGMENT_RUN_ID_OFFSET,
    CARRIED_COLUMNS,
    PAIR_KEY,
    _assign_augment_run_ids,
    compute_augment,
)


def _row(
    norad_id: int,
    epoch_i: str,
    target_dt_sec: int = 86_400,
    alt_shell: str = "550",
    generation: str = "v1.5",
) -> dict:
    """One synthetic corpus row with all columns the driver needs.

    `epoch_j = epoch_i + target_dt_sec` keeps the pair internally
    consistent. The GCAT-derived spacecraft-property columns get
    realistic v1.5 defaults; tests overriding generation also override
    the drag/SRP areas accordingly.
    """
    epoch_i_ts = pd.Timestamp(epoch_i, tz="UTC")
    epoch_j_ts = epoch_i_ts + pd.Timedelta(seconds=target_dt_sec)
    drag = 4.5 if generation == "v2-mini" else 2.0
    srp = 10.0 if generation == "v2-mini" else 5.0
    return {
        "norad_id": norad_id,
        "target_dt_sec": target_dt_sec,
        "epoch_i": epoch_i_ts,
        "epoch_j": epoch_j_ts,
        "actual_dt_sec": float(target_dt_sec),
        "line1_i": "1",
        "line2_i": "2",
        "line1_j": "1",
        "line2_j": "2",
        "sma_i_km": 6928.0,
        "sma_j_km": 6928.0,
        "alt_shell": alt_shell,
        "dry_mass_kg": 260.0 if generation == "v1.5" else 800.0,
        "span_m": 9.0,
        "generation": generation,
        "drag_area_m2": drag,
        "srp_area_m2": srp,
        "gcat_pl_name": "Starlink",
    }


class TestComputeAugment:
    def test_empty_when_corpora_equal(self) -> None:
        rows = [
            _row(101, "2026-04-01T00:00:00Z"),
            _row(102, "2026-04-02T00:00:00Z"),
        ]
        same = pd.DataFrame(rows)
        aug = compute_augment(same, same.copy())
        assert aug.empty

    def test_returns_pairs_only_in_high_threshold(self) -> None:
        base = pd.DataFrame([_row(101, "2026-04-01T00:00:00Z")])
        high = pd.DataFrame(
            [
                _row(101, "2026-04-01T00:00:00Z"),  # in both
                _row(102, "2026-04-02T00:00:00Z"),  # only in high
                _row(103, "2026-04-03T00:00:00Z"),  # only in high
            ],
        )
        aug = compute_augment(base, high)
        assert set(aug["norad_id"]) == {102, 103}

    def test_match_key_is_norad_and_both_epochs(self) -> None:
        # Same norad_id and epoch_i but different epoch_j → different pair.
        base = pd.DataFrame([_row(101, "2026-04-01T00:00:00Z", target_dt_sec=86_400)])
        high = pd.DataFrame(
            [
                _row(101, "2026-04-01T00:00:00Z", target_dt_sec=86_400),  # exact dup
                _row(101, "2026-04-01T00:00:00Z", target_dt_sec=604_800),  # different epoch_j
            ],
        )
        aug = compute_augment(base, high)
        assert len(aug) == 1
        assert aug.iloc[0]["target_dt_sec"] == 604_800

    def test_preserves_corpus_schema(self) -> None:
        # All GCAT-derived columns must survive — the GMAT driver reads
        # `drag_area_m2`, `srp_area_m2`, `dry_mass_kg`, and `generation`.
        base = pd.DataFrame([_row(101, "2026-04-01T00:00:00Z")])
        high = pd.DataFrame(
            [
                _row(101, "2026-04-01T00:00:00Z"),
                _row(102, "2026-04-02T00:00:00Z", generation="v2-mini"),
            ],
        )
        aug = compute_augment(base, high)
        for col in CARRIED_COLUMNS:
            assert col in aug.columns
        assert aug.iloc[0]["generation"] == "v2-mini"
        assert aug.iloc[0]["drag_area_m2"] == 4.5

    def test_empty_baseline_returns_full_high(self) -> None:
        base = pd.DataFrame(columns=list(PAIR_KEY))
        high = pd.DataFrame(
            [
                _row(101, "2026-04-01T00:00:00Z"),
                _row(102, "2026-04-02T00:00:00Z"),
            ],
        )
        aug = compute_augment(base, high)
        assert len(aug) == 2


class TestAssignAugmentRunIds:
    def test_run_ids_start_at_offset(self) -> None:
        rows = [
            _row(102, "2026-04-02T00:00:00Z"),
            _row(101, "2026-04-01T00:00:00Z"),
            _row(103, "2026-04-03T00:00:00Z"),
        ]
        out = _assign_augment_run_ids(pd.DataFrame(rows))
        assert out.index[0] == AUGMENT_RUN_ID_OFFSET
        assert out.index[-1] == AUGMENT_RUN_ID_OFFSET + len(rows) - 1
        # Index is contiguous and integer-typed.
        assert (out.index == range(AUGMENT_RUN_ID_OFFSET, AUGMENT_RUN_ID_OFFSET + len(rows))).all()

    def test_sort_order_is_deterministic(self) -> None:
        rows_a = [
            _row(103, "2026-04-03T00:00:00Z"),
            _row(101, "2026-04-01T00:00:00Z"),
            _row(102, "2026-04-02T00:00:00Z"),
        ]
        rows_b = [
            _row(101, "2026-04-01T00:00:00Z"),
            _row(102, "2026-04-02T00:00:00Z"),
            _row(103, "2026-04-03T00:00:00Z"),
        ]
        out_a = _assign_augment_run_ids(pd.DataFrame(rows_a))
        out_b = _assign_augment_run_ids(pd.DataFrame(rows_b))
        # Sort order should align regardless of input order.
        assert list(out_a["norad_id"]) == list(out_b["norad_id"])
        assert list(out_a.index) == list(out_b.index)

    def test_sort_breaks_ties_by_target_dt(self) -> None:
        # Same sat and same starting epoch but different Δt buckets → must
        # land at distinct run_ids in a stable order.
        rows = [
            _row(101, "2026-04-01T00:00:00Z", target_dt_sec=604_800),
            _row(101, "2026-04-01T00:00:00Z", target_dt_sec=86_400),
            _row(101, "2026-04-01T00:00:00Z", target_dt_sec=21_600),
        ]
        out = _assign_augment_run_ids(pd.DataFrame(rows))
        assert list(out["target_dt_sec"]) == [21_600, 86_400, 604_800]

    def test_empty_input_returns_empty(self) -> None:
        # Defensive: an empty augment must not blow up the assigner.
        cols = list(_row(0, "2026-04-01T00:00:00Z").keys())
        empty = pd.DataFrame(columns=cols)
        out = _assign_augment_run_ids(empty)
        assert out.empty
        assert out.index.start == AUGMENT_RUN_ID_OFFSET
        assert out.index.stop == AUGMENT_RUN_ID_OFFSET

    def test_carried_columns_preserved(self) -> None:
        # The aggregator reads `CARRIED_COLUMNS` off the run_id-indexed
        # frame — those columns must survive the sort+reindex step.
        rows = [_row(101, "2026-04-01T00:00:00Z", generation="v2-mini")]
        out = _assign_augment_run_ids(pd.DataFrame(rows))
        for col in CARRIED_COLUMNS:
            assert col in out.columns
        assert out.iloc[0]["generation"] == "v2-mini"


class TestModuleConstants:
    def test_augment_offset_is_above_main_sweep_ids(self) -> None:
        # The main sweep has ≤ 24,641 run_ids; the augment offset must
        # sit well above that to guarantee no collision after concat.
        assert AUGMENT_RUN_ID_OFFSET >= 100_000

    def test_carried_columns_match_aggregate_contract(self) -> None:
        # The aggregator's CARRIED_COLUMNS must include `generation`
        # (read by the table emitter for per-(shell × gen) bootstrap)
        # plus the spacecraft-property columns the figure scripts
        # consume.
        assert "generation" in CARRIED_COLUMNS
        for col in ("dry_mass_kg", "drag_area_m2", "srp_area_m2"):
            assert col in CARRIED_COLUMNS

    def test_pair_key_is_norad_and_both_epochs(self) -> None:
        # The set-difference logic depends on this triple. If the key
        # changes, augment computation against the locked corpora must
        # be re-validated.
        assert PAIR_KEY == ("norad_id", "epoch_i", "epoch_j")
