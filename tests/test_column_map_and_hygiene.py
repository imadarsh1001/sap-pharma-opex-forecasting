"""Tests for column_map, io readable loading, PIT cutoff, and hygiene diagnostics."""

from __future__ import annotations

import pandas as pd

from opex_forecast.column_map import build_column_crosswalk, rename_to_sap
from opex_forecast.data_prep.prepare_opex_panel import (
    audit_universal_journal,
    build_monthly_opex_panel,
    extract_hygiene_diagnostics,
)
from opex_forecast.io import load_pharma_table


def test_column_map_round_trip_and_ingestion() -> None:
    cw = build_column_crosswalk()
    assert len(cw) == 206

    for tbl in cw["table_name"].unique():
        raw = load_pharma_table(tbl, readable=False)
        readable = load_pharma_table(tbl, readable=True)
        assert not readable.columns.has_duplicates
        back = rename_to_sap(readable, tbl, strict=True)
        assert list(back.columns) == list(raw.columns)

    # Signed vs. unsigned USD column clarity
    act_r = load_pharma_table("acdoca_actuals", readable=True)
    uj_r = load_pharma_table("acdoca_universal_journal", readable=True)
    assert "AMOUNT_USD_REPORTED_UNSIGNED" in act_r.columns
    assert "AMOUNT_USD_REPORTED_SIGNED" in uj_r.columns

    # Table-scoped WBS_ELEMENT vs. PS_PSP_PNR reverse mapping + strict=False pass-through
    drv_back = rename_to_sap(load_pharma_table("fact_opex_drivers", readable=True), "fact_opex_drivers", strict=True)
    assert "WBS_ELEMENT" in drv_back.columns and "PS_PSP_PNR" not in drv_back.columns

    extra_back = rename_to_sap(uj_r.assign(EXTRA_FEATURE=1.0), "acdoca_universal_journal", strict=False)
    assert "EXTRA_FEATURE" in extra_back.columns and "PS_PSP_PNR" in extra_back.columns

    # Leading zeros & blank strings preserved on readable=True + DOCLN/BUZEI join
    assert uj_r["DOCUMENT_NUMBER"].str.startswith("0").all()
    assert uj_r["LINE_ITEM_NUMBER"].isin(["000001", "000002"]).all()
    for col in ("COST_CENTER_ID", "FUNCTIONAL_AREA", "REVERSAL_DOCUMENT_NUMBER", "PO_NUMBER"):
        assert not uj_r[col].isna().any()
        assert (uj_r[col] == "").any()

    eval_r = load_pharma_table("eval_anomaly_labels", readable=True)
    matched = eval_r.merge(
        uj_r.assign(LINE_ITEM_INT=uj_r["LINE_ITEM_NUMBER"].astype(int)),
        left_on=["COMPANY_CODE", "FISCAL_YEAR", "DOCUMENT_NUMBER", "LINE_ITEM_NUMBER"],
        right_on=["COMPANY_CODE", "FISCAL_YEAR", "DOCUMENT_NUMBER", "LINE_ITEM_INT"],
        how="inner",
    )
    assert len(matched) == len(eval_r)


def test_pit_cutoff_prevents_future_leakage_and_hygiene_diagnostics(tmp_path) -> None:
    cutoff = "2025-06-10"
    res = build_monthly_opex_panel(cutoff_date=cutoff, output_dir=tmp_path)
    panel = res["opex_monthly_panel"]
    benchmarks = res["plan_benchmarks"]
    specials = res["special_items_schedule"]

    # No post-cutoff calendar months or drivers in panel
    assert panel.loc[panel["RYEAR"] == 2025, "POPER"].max() == 6
    assert (benchmarks["SNAPSHOT_DATE"] <= cutoff).all()
    assert (specials["CPUDT"] <= cutoff).all()

    jun10 = audit_universal_journal(cutoff_date=cutoff)["clean_opex"]
    jun30 = audit_universal_journal(cutoff_date="2025-06-30")["clean_opex"]
    assert len(jun30) > len(jun10)

    diag = extract_hygiene_diagnostics()
    late = diag["late_close_by_blart"]
    assert not late.empty and (late["LATE_CLOSE_LINES"] >= 0).all()
    assert len(diag["prepaid_lines"]) > 0
    fx = diag["monthly_fx_variance"]
    assert (fx["FX_TRANSLATION_USD"] - (fx["KSL"] - fx["KSL_BUDGET_FX"])).abs().max() < 1e-6
