"""Tests for Stage 1 OPEX data preparation and panel construction."""

from __future__ import annotations

import pandas as pd

from opex_forecast.data_prep.prepare_opex_panel import build_monthly_opex_panel
from opex_forecast.io import load_pharma_table


def test_prepare_opex_panel_tie_out_and_guardrails(tmp_path) -> None:
    res = build_monthly_opex_panel(output_dir=tmp_path)
    panel = pd.read_parquet(tmp_path / "opex_monthly_panel.parquet")
    benchmarks = pd.read_parquet(tmp_path / "plan_benchmarks.parquet")
    specials = pd.read_parquet(tmp_path / "special_items_schedule.parquet")

    assert len(panel) == len(res["opex_monthly_panel"]) == 196 * 48
    assert not panel.duplicated(subset=["RYEAR", "POPER", "RCNTR", "RACCT"]).any()

    # Forbidden leakage/metadata columns must not appear in modeling panel
    for forbidden in ("COST_SCALE", "VERSION", "ANOMALY_TAG"):
        assert forbidden not in panel.columns

    # Rolling forecasts (VERSION=2) live only in plan_benchmarks.parquet
    assert set(benchmarks["VERSION"].unique()) == {1, 2}
    assert len(specials) > 0

    # Exact dollar tie-out against raw acdoca_universal_journal
    uj = load_pharma_table("acdoca_universal_journal")
    gl = load_pharma_table("dim_gl_account")
    opex_gls = set(gl.loc[gl["ACCOUNT_TYPE"] == "OPEX", "HKONT"])
    raw_clean = uj[
        (uj["RACCT"].isin(opex_gls))
        & (uj["POPER"] <= 12)
        & (uj["SPECIAL_ITEM_FLAG"] == "NONE")
    ]
    assert abs(float(panel["KSL"].sum()) - float(raw_clean["KSL"].sum())) < 0.01
    assert abs(float(panel["KSL_BUDGET_FX"].sum()) - float(raw_clean["KSL_BUDGET_FX"].sum())) < 0.01

    # Single-snapshot COOI PO check on a sample month (2024-06 -> snapshot 2024-05-31)
    po = load_pharma_table("fact_po_commitments")
    po_2024_06 = (
        po[(po["SNAPSHOT_DATE"] == "2024-05-31") & (po["EXPECTED_GR_MONTH"] == "2024-06")]
        .groupby(["KOSTL", "HKONT"], as_index=False)["OPEN_COMMITMENT_USD"]
        .sum()
    )
    panel_2024_06 = panel[(panel["RYEAR"] == 2024) & (panel["POPER"] == 6)]
    merged = panel_2024_06.merge(
        po_2024_06, left_on=["RCNTR", "RACCT"], right_on=["KOSTL", "HKONT"], how="inner"
    )
    assert len(merged) > 0
    assert (merged["PO_OPEN_COMMIT_T_MINUS_1"] - merged["OPEN_COMMITMENT_USD"]).abs().max() < 1e-6

    # Causal lag/YTD guard: POPER == 1 has NaN YTD utilization and 2022-01 has NaN LAG_1M
    assert panel.loc[panel["POPER"] == 1, "YTD_BUDGET_UTILIZATION_LAG1"].isna().all()
    assert panel.loc[(panel["RYEAR"] == 2022) & (panel["POPER"] == 1), "LAG_1M"].isna().all()
