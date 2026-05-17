"""Tests for src/scripts/_propagator_wins.py.

The module computes binary "hi-fid beats SGP4" fractions per cell and
bootstraps them at the satellite level. Meaningful coverage:

  - Estimator semantics: a hand-built frame where the ground-truth win
    fraction is known returns that fraction, and the along-track
    estimator uses absolute values (sign cancellation across pairs
    cannot drive the metric).
  - Schema: the JSON carries both ``pooled_by_dt`` and ``by_cell``
    blocks, each with the documented per-row keys; ``by_cell`` rows
    carry the ``gens_present`` annotation.
  - Plumbing: every cell defined in ``ALT_SHELL_ORDER`` ×
    ``BUCKET_SECONDS`` is represented in ``by_cell`` (empty cells
    return ``n_pairs == 0`` rather than being dropped silently).
  - Table fragment: per-cell rows below the ``MIN_CELL_PAIRS`` floor are
    suppressed from the booktabs output; pooled-per-Δt rows always
    appear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _propagator_wins import (  # noqa: E402
    MIN_CELL_PAIRS,
    _wins,
    compute,
    render_table,
)
from _style import ALT_SHELL_ORDER, BUCKET_SECONDS


def _frame(
    n_sats: int = 6,
    *,
    bucket_sec: int = 21600,
    alt_shell: str = "550",
    generation: str = "v1.5",
    hifi_factor: float = 0.5,
) -> pd.DataFrame:
    """Synthetic run frame with a known hi-fid-vs-SGP4 outcome.

    Every pair has ``dr_hifi_km = hifi_factor * dr_sgp4_km``: with
    ``hifi_factor < 1`` every pair is a hi-fid win, with
    ``hifi_factor > 1`` every pair is an SGP4 win. The along-track
    components mirror the L2 scaling but carry opposing signs across
    pairs so the absolute-value comparison is non-trivial.
    """
    rng = np.random.default_rng(0)
    rows = []
    for sat in range(n_sats):
        for k in range(5):  # 5 pairs per sat
            sgp4 = float(rng.uniform(0.5, 10.0))
            sign = 1.0 if (k + sat) % 2 == 0 else -1.0
            rows.append(
                {
                    "norad_id": 50000 + sat,
                    "alt_shell": alt_shell,
                    "generation": generation,
                    "target_dt_sec": bucket_sec,
                    "dr_sgp4_km": sgp4,
                    "dr_hifi_km": hifi_factor * sgp4,
                    # Magnitudes mirror the L2 scaling, signs flip pair-to-pair.
                    "dr_sgp4_along_km": sign * 0.95 * sgp4,
                    "dr_hifi_along_km": -sign * 0.95 * hifi_factor * sgp4,
                }
            )
    return pd.DataFrame(rows)


def test_estimator_recovers_known_win_fractions():
    """All-win and all-lose synthetic frames anchor the estimator."""
    all_win = _frame(hifi_factor=0.5)
    all_lose = _frame(hifi_factor=2.0)
    assert _wins(all_win) == {"hifi_wins_l2": 1.0, "hifi_wins_along": 1.0}
    assert _wins(all_lose) == {"hifi_wins_l2": 0.0, "hifi_wins_along": 0.0}


def test_along_track_uses_absolute_values():
    """A frame with sign-symmetric along-track components must not produce
    a non-trivial along-track win fraction by sign cancellation alone."""
    df = _frame(hifi_factor=0.5)
    # Sanity: signs do oscillate (the test would be vacuous if the
    # frame happened to have constant-sign components).
    assert df["dr_sgp4_along_km"].pipe(lambda s: (s > 0).any() and (s < 0).any())
    # Hi-fid magnitude is strictly smaller than SGP4 magnitude on every
    # pair by construction, so the absolute-value comparison must
    # report a 100% win fraction regardless of sign pattern.
    assert _wins(df)["hifi_wins_along"] == 1.0


def test_compute_emits_full_schema_and_grid():
    """`compute` returns the documented blocks and covers the full grid."""
    cells = []
    for shell in ALT_SHELL_ORDER:
        for bucket in BUCKET_SECONDS:
            cells.append(
                _frame(
                    bucket_sec=bucket,
                    alt_shell=shell,
                    generation="v1.5",
                    hifi_factor=0.7,
                )
            )
    payload = compute(pd.concat(cells, ignore_index=True))

    assert payload["n_resamples"] > 0
    assert "hifi_wins_l2" in payload["metric_definitions"]
    assert "hifi_wins_along" in payload["metric_definitions"]
    assert len(payload["pooled_by_dt"]) == len(BUCKET_SECONDS)
    assert len(payload["by_cell"]) == len(ALT_SHELL_ORDER) * len(BUCKET_SECONDS)

    for row in payload["pooled_by_dt"]:
        assert row["n_pairs"] > 0
        assert row["hifi_wins_l2"]["point"] == 1.0  # hifi_factor=0.7 everywhere
        # CI bounds well-defined on >= MIN_CELL_PAIRS / 2 resamples.
        lo, hi = row["hifi_wins_l2"]["ci_95"]
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    for row in payload["by_cell"]:
        assert row["alt_shell"] in ALT_SHELL_ORDER
        assert row["target_dt_sec"] in BUCKET_SECONDS
        # gens_present is the per-cell annotation; pooled rows lack it.
        assert "gens_present" in row


def test_empty_cells_appear_with_zero_pairs():
    """A cell with no pairs must serialise as ``n_pairs=0`` rather than
    being dropped from ``by_cell`` (the §4.2 prose depends on the
    full grid being present)."""
    cells = []
    # Populate only one cell; leave the rest empty.
    cells.append(_frame(bucket_sec=21600, alt_shell="540", generation="v1.5"))
    payload = compute(pd.concat(cells, ignore_index=True))
    empty_cells = [r for r in payload["by_cell"] if r["n_pairs"] == 0]
    populated_cells = [r for r in payload["by_cell"] if r["n_pairs"] > 0]
    assert len(populated_cells) == 1
    assert len(empty_cells) == len(ALT_SHELL_ORDER) * len(BUCKET_SECONDS) - 1
    # Empty rows still carry None payloads, not raw fractions.
    for row in empty_cells:
        assert row["hifi_wins_l2"] is None
        assert row["hifi_wins_along"] is None


def test_render_table_drops_thin_cells_but_keeps_pooled():
    """Cells with fewer than MIN_CELL_PAIRS pairs are suppressed; pooled
    rows always appear."""
    thin = _frame(n_sats=2, bucket_sec=21600, alt_shell="540")  # 10 pairs
    fat = _frame(n_sats=12, bucket_sec=86400, alt_shell="540")  # 60 pairs
    assert len(thin) < MIN_CELL_PAIRS
    assert len(fat) >= MIN_CELL_PAIRS
    payload = compute(pd.concat([thin, fat], ignore_index=True))
    table = render_table(payload)
    # Fat cell appears as a data row (one row matches the 540 / 1d cell).
    assert "540 & 1d & 60 &" in table
    # Thin cell's per-cell row is suppressed.
    assert "540 & 6h & 10 &" not in table
    # Pooled rows always appear under \textit{pooled}.
    assert r"\textit{pooled}" in table
