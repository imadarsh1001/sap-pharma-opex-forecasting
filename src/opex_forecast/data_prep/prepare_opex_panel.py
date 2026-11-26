"""Stage 1: OPEX Data Preparation & Point-in-Time Feature Panel Builder.

Enforces double-entry balance checks, the canonical 4-condition recurring OPEX
filter, single-snapshot COOI PO commitment joins, and strictly causal lag/YTD
features using canonical SAP column names throughout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from opex_forecast.config import PROCESSED_DATA_DIR
from opex_forecast.io import load_pharma_table

WATERFALL_AMOUNT_COLS = ("HSL", "KSL", "KSL_BUDGET_FX")


def _summarize_step(step_num: int, label: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "STEP": step_num,
        "FILTER_STAGE": label,
        "LINE_COUNT": int(len(df)),
        "DOC_COUNT": int(df[["RBUKRS", "RYEAR", "BELNR"]].drop_duplicates().shape[0]),
        "HSL_SUM": round(float(df["HSL"].sum()), 2),
        "KSL_SUM": round(float(df["KSL"].sum()), 2),
        "KSL_BUDGET_FX_SUM": round(float(df["KSL_BUDGET_FX"].sum()), 2),
    }


def _assert_double_entry_balance(uj_df: pd.DataFrame, tol: float = 0.01) -> None:
    doc_sums = uj_df.groupby(["RBUKRS", "RYEAR", "BELNR"], as_index=False)[list(WATERFALL_AMOUNT_COLS)].sum()
    for col in WATERFALL_AMOUNT_COLS:
        max_abs = float(doc_sums[col].abs().max())
        if max_abs >= tol:
            raise ValueError(f"Double-entry balance violated on {col}: max_abs={max_abs:.4f} >= {tol}")


def audit_universal_journal(
    cutoff_date: str | None = None,
    uj_df: pd.DataFrame | None = None,
    dim_gl: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Audit double-entry balance and execute the 4-condition recurring OPEX waterfall."""
    uj = load_pharma_table("acdoca_universal_journal") if uj_df is None else uj_df.copy()
    gl = load_pharma_table("dim_gl_account") if dim_gl is None else dim_gl

    _assert_double_entry_balance(uj)
    opex_gls = set(gl.loc[gl["ACCOUNT_TYPE"] == "OPEX", "HKONT"])

    step1_opex = uj[uj["RACCT"].isin(opex_gls)]
    step2_reg = step1_opex[step1_opex["POPER"] <= 12]
    step3_clean = step2_reg[step2_reg["SPECIAL_ITEM_FLAG"] == "NONE"]
    if cutoff_date is not None:
        step4_pit = step3_clean[step3_clean["CPUDT"] <= cutoff_date]
        pit_label = f"4. CPUDT <= {cutoff_date}"
    else:
        step4_pit = step3_clean
        pit_label = "4. CPUDT <= None (Full History)"

    waterfall = pd.DataFrame(
        [
            _summarize_step(0, "0. All Journal Lines (OPEX + BS)", uj),
            _summarize_step(1, "1. ACCOUNT_TYPE == 'OPEX'", step1_opex),
            _summarize_step(2, "2. POPER <= 12 (Regular Periods)", step2_reg),
            _summarize_step(3, "3. SPECIAL_ITEM_FLAG == 'NONE'", step3_clean),
            _summarize_step(4, pit_label, step4_pit),
        ]
    )

    special_periods = step1_opex[step1_opex["POPER"] > 12].copy()
    if cutoff_date is not None:
        special_periods = special_periods[special_periods["CPUDT"] <= cutoff_date]
    special_periods_summary = (
        special_periods.groupby(["RYEAR", "POPER", "RBUKRS", "BLART"], as_index=False)
        .agg(
            LINE_COUNT=("BELNR", "count"),
            DOC_COUNT=("BELNR", "nunique"),
            KSL=("KSL", "sum"),
            KSL_BUDGET_FX=("KSL_BUDGET_FX", "sum"),
        )
        .sort_values(["RYEAR", "POPER", "RBUKRS"])
        .reset_index(drop=True)
    )

    gaap_specials = step2_reg[step2_reg["SPECIAL_ITEM_FLAG"] != "NONE"]
    if cutoff_date is not None:
        gaap_specials = gaap_specials[gaap_specials["CPUDT"] <= cutoff_date]
    gaap_specials = gaap_specials.assign(SCHEDULE_BUCKET="GAAP_SPECIAL_ITEM")
    audit_periods = special_periods.assign(SCHEDULE_BUCKET="SPECIAL_PERIOD_13_16")
    special_items_schedule = (
        pd.concat([gaap_specials, audit_periods], ignore_index=True)
        .sort_values(["RYEAR", "POPER", "RBUKRS", "BELNR", "DOCLN"])
        .reset_index(drop=True)
    )

    return {
        "waterfall": waterfall,
        "special_periods_summary": special_periods_summary,
        "special_items_schedule": special_items_schedule,
        "clean_opex": step4_pit.reset_index(drop=True),
    }


def extract_hygiene_diagnostics(
    uj_df: pd.DataFrame | None = None,
    dim_gl: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute late-close lag, multi-month prepaid spans, and monthly FX translation deltas."""
    uj = load_pharma_table("acdoca_universal_journal") if uj_df is None else uj_df
    gl = load_pharma_table("dim_gl_account") if dim_gl is None else dim_gl
    opex_gls = set(gl.loc[gl["ACCOUNT_TYPE"] == "OPEX", "HKONT"])

    opex_l1 = uj[uj["RACCT"].isin(opex_gls)].copy()
    budat_dt = pd.to_datetime(opex_l1["BUDAT"])
    cpudt_dt = pd.to_datetime(opex_l1["CPUDT"])
    opex_l1["IS_LATE_CLOSE"] = cpudt_dt.dt.to_period("M") > budat_dt.dt.to_period("M")
    opex_l1["LATE_CLOSE_KSL"] = np.where(opex_l1["IS_LATE_CLOSE"], opex_l1["KSL"], 0.0)

    late_close_by_blart = (
        opex_l1.groupby("BLART", as_index=False)
        .agg(
            TOTAL_LINES=("BELNR", "count"),
            LATE_CLOSE_LINES=("IS_LATE_CLOSE", "sum"),
            LATE_CLOSE_KSL=("LATE_CLOSE_KSL", "sum"),
        )
        .assign(LATE_CLOSE_SHARE=lambda d: (d["LATE_CLOSE_LINES"] / d["TOTAL_LINES"]).round(4))
        .sort_values("LATE_CLOSE_LINES", ascending=False)
        .reset_index(drop=True)
    )

    svc_days = (pd.to_datetime(opex_l1["SERVICE_END_DATE"]) - pd.to_datetime(opex_l1["SERVICE_START_DATE"])).dt.days
    prepaid_lines = (
        opex_l1[svc_days > 35]
        .assign(SERVICE_SPAN_DAYS=svc_days[svc_days > 35])
        .reset_index(drop=True)
    )

    clean_reg = opex_l1[(opex_l1["POPER"] <= 12) & (opex_l1["SPECIAL_ITEM_FLAG"] == "NONE")]
    monthly_fx_variance = (
        clean_reg.groupby(["RYEAR", "POPER"], as_index=False)[["HSL", "KSL", "KSL_BUDGET_FX"]]
        .sum()
        .assign(FX_TRANSLATION_USD=lambda d: d["KSL"] - d["KSL_BUDGET_FX"])
        .sort_values(["RYEAR", "POPER"])
        .reset_index(drop=True)
    )

    return {
        "late_close_by_blart": late_close_by_blart,
        "prepaid_lines": prepaid_lines,
        "monthly_fx_variance": monthly_fx_variance,
    }


def _aggregate_single_snapshot_po(
    po_df: pd.DataFrame,
    grid_months: pd.DataFrame,
    lag_months: int,
    out_col: str,
) -> pd.DataFrame:
    """Aggregate OPEN_COMMITMENT_USD from exactly one SNAPSHOT_DATE per target month."""
    month_map = grid_months[["RYEAR", "POPER"]].drop_duplicates().copy()
    month_starts = pd.to_datetime(
        dict(year=month_map["RYEAR"], month=month_map["POPER"], day=1)
    )
    month_map["EXPECTED_GR_MONTH"] = month_starts.dt.strftime("%Y-%m")
    month_map["TARGET_SNAPSHOT_DATE"] = (
        month_starts - pd.offsets.MonthEnd(lag_months)
    ).dt.strftime("%Y-%m-%d")

    if month_map["TARGET_SNAPSHOT_DATE"].nunique() != len(month_map):
        raise ValueError(f"Non-1:1 snapshot mapping detected for lag_months={lag_months}")

    scoped = po_df.merge(
        month_map,
        left_on=["SNAPSHOT_DATE", "EXPECTED_GR_MONTH"],
        right_on=["TARGET_SNAPSHOT_DATE", "EXPECTED_GR_MONTH"],
        how="inner",
    )
    agg = (
        scoped.groupby(["RYEAR", "POPER", "KOSTL", "HKONT"], as_index=False)["OPEN_COMMITMENT_USD"]
        .sum()
        .rename(columns={"KOSTL": "RCNTR", "HKONT": "RACCT", "OPEN_COMMITMENT_USD": out_col})
    )
    if agg.duplicated(subset=["RYEAR", "POPER", "RCNTR", "RACCT"]).any():
        raise ValueError(f"Duplicate PO commitment keys detected for {out_col}")
    return agg


def _add_causal_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add strictly shifted target/driver lags and calendar features per (RCNTR, RACCT)."""
    df = panel.sort_values(["RCNTR", "RACCT", "RYEAR", "POPER"]).reset_index(drop=True)
    grp_target = df.groupby(["RCNTR", "RACCT"], sort=False)["KSL_BUDGET_FX"]

    for lag in (1, 2, 3, 12):
        df[f"LAG_{lag}M"] = grp_target.shift(lag)

    df["_SHIFTED_TARGET"] = grp_target.shift(1)
    roll_grp = df.groupby(["RCNTR", "RACCT"], sort=False)["_SHIFTED_TARGET"]
    df["ROLL_MEAN_3M"] = roll_grp.rolling(3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df["ROLL_MEAN_6M"] = roll_grp.rolling(6, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df["ROLL_STD_6M"] = roll_grp.rolling(6, min_periods=2).std().reset_index(level=[0, 1], drop=True)
    df["ROLL_MEAN_12M"] = roll_grp.rolling(12, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df = df.drop(columns=["_SHIFTED_TARGET"])

    driver_lags = {
        "ACTIVE_FTE": "ACTIVE_FTE_LAG1",
        "ENROLLED_PATIENTS": "ENROLLED_PATIENTS_LAG1",
        "ACTIVE_SITES": "ACTIVE_SITES_LAG1",
        "HCP_EVENTS_COUNT": "HCP_EVENTS_LAG1",
        "WAGE_INDEX": "WAGE_INDEX_LAG1",
        "CPI_INDEX": "CPI_INDEX_LAG1",
    }
    grp = df.groupby(["RCNTR", "RACCT"], sort=False)
    for src_col, dst_col in driver_lags.items():
        df[dst_col] = grp[src_col].shift(1)

    df["QUARTER"] = ((df["POPER"] - 1) // 3) + 1
    df["IS_Q4"] = (df["QUARTER"] == 4).astype("int64")
    df["IS_YEAR_END"] = (df["POPER"] == 12).astype("int64")
    angle = 2.0 * np.pi * (df["POPER"] - 1) / 12.0
    df["MONTH_SIN"] = np.sin(angle)
    df["MONTH_COS"] = np.cos(angle)
    return df


def build_monthly_opex_panel(
    cutoff_date: str | None = None,
    output_dir: Path | str | None = None,
    uj_df: pd.DataFrame | None = None,
    dim_gl: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build the 48-month recurring OPEX panel, benchmarks, and special-items schedule."""
    gl = load_pharma_table("dim_gl_account") if dim_gl is None else dim_gl
    dim_cc = load_pharma_table("dim_cost_center")
    drivers = load_pharma_table("fact_opex_drivers")
    po = load_pharma_table("fact_po_commitments")
    plan_versions = load_pharma_table("plan_versions")
    plan_drivers = load_pharma_table("plan_drivers")

    audit_res = audit_universal_journal(cutoff_date=cutoff_date, uj_df=uj_df, dim_gl=gl)
    clean_opex = audit_res["clean_opex"]

    actual_agg = (
        clean_opex.groupby(["RYEAR", "POPER", "RBUKRS", "WERKS", "RCNTR", "RACCT"], as_index=False)
        .agg(
            HSL=("HSL", "sum"),
            KSL=("KSL", "sum"),
            KSL_BUDGET_FX=("KSL_BUDGET_FX", "sum"),
            DOC_COUNT=("BELNR", "nunique"),
            LINE_COUNT=("BELNR", "count"),
        )
    )

    series_keys = actual_agg[["RBUKRS", "WERKS", "RCNTR", "RACCT"]].drop_duplicates().sort_values(
        ["RCNTR", "RACCT"]
    )
    years = sorted(clean_opex["RYEAR"].unique().tolist())
    months = list(range(1, 13))
    cal_grid = pd.MultiIndex.from_product([years, months], names=["RYEAR", "POPER"]).to_frame(index=False)

    if cutoff_date is not None:
        cutoff_ts = pd.Timestamp(cutoff_date)
        cal_grid = cal_grid[
            (cal_grid["RYEAR"] < cutoff_ts.year)
            | ((cal_grid["RYEAR"] == cutoff_ts.year) & (cal_grid["POPER"] <= cutoff_ts.month))
        ].reset_index(drop=True)
        drivers = drivers[
            (drivers["GJAHR"] < cutoff_ts.year)
            | ((drivers["GJAHR"] == cutoff_ts.year) & (drivers["MONAT"] <= cutoff_ts.month))
        ]
        plan_versions = plan_versions[plan_versions["SNAPSHOT_DATE"] <= cutoff_date].reset_index(drop=True)
        plan_drivers = plan_drivers[
            (plan_drivers["GJAHR"] < cutoff_ts.year)
            | ((plan_drivers["GJAHR"] == cutoff_ts.year) & (plan_drivers["MONAT"] <= cutoff_ts.month))
        ]

    panel = series_keys.merge(cal_grid, how="cross").merge(
        actual_agg,
        on=["RYEAR", "POPER", "RBUKRS", "WERKS", "RCNTR", "RACCT"],
        how="left",
    )
    for zero_col in ("HSL", "KSL", "KSL_BUDGET_FX"):
        panel[zero_col] = panel[zero_col].fillna(0.0)
    for cnt_col in ("DOC_COUNT", "LINE_COUNT"):
        panel[cnt_col] = panel[cnt_col].fillna(0).astype("int64")

    cc_cols = ["KOSTL", "SEGMENT", "FRANCHISE_TA", "REGION", "RFAREA", "DEPARTMENT"]
    gl_cols = ["HKONT", "GL_NAME", "COST_BEHAVIOR", "OPEX_GROUP", "USES_PO"]
    panel = panel.merge(
        dim_cc[cc_cols], left_on="RCNTR", right_on="KOSTL", how="left"
    ).drop(columns=["KOSTL"])
    panel = panel.merge(
        gl[gl_cols], left_on="RACCT", right_on="HKONT", how="left"
    ).drop(columns=["HKONT"])

    driver_feature_cols = [
        "GJAHR",
        "MONAT",
        "KOSTL",
        "WBS_ELEMENT",
        "ACTIVE_FTE",
        "PLANNED_HIRES",
        "FIELD_SALES_REPS",
        "MERIT_CYCLE_FLAG",
        "TRIAL_PHASE",
        "ACTIVE_SITES",
        "ENROLLED_PATIENTS",
        "MILESTONE_EVENT",
        "PRODUCT_LAUNCH_FLAG",
        "LOE_FLAG",
        "HCP_EVENTS_COUNT",
        "SURGICAL_PROCEDURE_INDEX",
        "CPI_INDEX",
        "WAGE_INDEX",
        "WAGE_INFLATION_RATE",
    ]
    panel = panel.merge(
        drivers[driver_feature_cols],
        left_on=["RYEAR", "POPER", "RCNTR"],
        right_on=["GJAHR", "MONAT", "KOSTL"],
        how="left",
    ).drop(columns=["GJAHR", "MONAT", "KOSTL"])

    panel = _add_causal_lag_features(panel)

    po_t1 = _aggregate_single_snapshot_po(po, cal_grid, lag_months=1, out_col="PO_OPEN_COMMIT_T_MINUS_1")
    po_t2 = _aggregate_single_snapshot_po(po, cal_grid, lag_months=2, out_col="PO_OPEN_COMMIT_T_MINUS_2")
    panel = panel.merge(po_t1, on=["RYEAR", "POPER", "RCNTR", "RACCT"], how="left")
    panel = panel.merge(po_t2, on=["RYEAR", "POPER", "RCNTR", "RACCT"], how="left")
    panel["PO_OPEN_COMMIT_T_MINUS_1"] = panel["PO_OPEN_COMMIT_T_MINUS_1"].fillna(0.0)
    panel["PO_OPEN_COMMIT_T_MINUS_2"] = panel["PO_OPEN_COMMIT_T_MINUS_2"].fillna(0.0)

    bp_versions = (
        plan_versions[plan_versions["VERSION"] == 1][
            ["GJAHR", "MONAT", "KOSTL", "HKONT", "KSL_BUDGET", "DMBTR_PLAN"]
        ]
        .rename(columns={"KSL_BUDGET": "BP_KSL_BUDGET", "DMBTR_PLAN": "BP_DMBTR_PLAN"})
    )
    panel = panel.merge(
        bp_versions,
        left_on=["RYEAR", "POPER", "RCNTR", "RACCT"],
        right_on=["GJAHR", "MONAT", "KOSTL", "HKONT"],
        how="left",
    ).drop(columns=["GJAHR", "MONAT", "KOSTL", "HKONT"])

    bp_drv = plan_drivers[
        ["GJAHR", "MONAT", "KOSTL", "PLAN_FTE", "PLAN_ENROLLED_PATIENTS", "PLAN_HCP_EVENTS", "PLAN_MILESTONE_EVENT"]
    ]
    panel = panel.merge(
        bp_drv,
        left_on=["RYEAR", "POPER", "RCNTR"],
        right_on=["GJAHR", "MONAT", "KOSTL"],
        how="left",
    ).drop(columns=["GJAHR", "MONAT", "KOSTL"])

    panel = panel.sort_values(["RYEAR", "RCNTR", "RACCT", "POPER"]).reset_index(drop=True)
    ytd_grp = panel.groupby(["RYEAR", "RCNTR", "RACCT"], sort=False)
    ytd_act_lag1 = ytd_grp["KSL_BUDGET_FX"].cumsum().groupby([panel["RYEAR"], panel["RCNTR"], panel["RACCT"]]).shift(1)
    ytd_bp_lag1 = ytd_grp["BP_KSL_BUDGET"].cumsum().groupby([panel["RYEAR"], panel["RCNTR"], panel["RACCT"]]).shift(1)
    panel["YTD_BUDGET_UTILIZATION_LAG1"] = np.where(
        (panel["POPER"] > 1) & (ytd_bp_lag1 > 0),
        ytd_act_lag1 / ytd_bp_lag1,
        np.nan,
    )

    forbidden_cols = {"COST_SCALE", "ANOMALY_TAG", "VERSION"}
    leaked = forbidden_cols.intersection(panel.columns)
    if leaked:
        raise ValueError(f"Forbidden leakage/metadata columns present in panel: {sorted(leaked)}")
    if panel.duplicated(subset=["RYEAR", "POPER", "RCNTR", "RACCT"]).any():
        raise ValueError("Panel primary key (RYEAR, POPER, RCNTR, RACCT) is not unique")

    panel = panel.sort_values(["RYEAR", "POPER", "RCNTR", "RACCT"]).reset_index(drop=True)

    out_dir = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_dir / "opex_monthly_panel.parquet", index=False)
    plan_versions.to_parquet(out_dir / "plan_benchmarks.parquet", index=False)
    audit_res["special_items_schedule"].to_parquet(out_dir / "special_items_schedule.parquet", index=False)

    return {
        "opex_monthly_panel": panel,
        "plan_benchmarks": plan_versions,
        "special_items_schedule": audit_res["special_items_schedule"],
        "waterfall": audit_res["waterfall"],
        "special_periods_summary": audit_res["special_periods_summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 1 OPEX monthly panel and audit schedules")
    parser.add_argument("--cutoff-date", default=None, help="Optional PIT CPUDT cutoff (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default=str(PROCESSED_DATA_DIR), help="Output directory for Parquet files")
    args = parser.parse_args()

    res = build_monthly_opex_panel(cutoff_date=args.cutoff_date, output_dir=args.output_dir)
    panel = res["opex_monthly_panel"]
    print("=== Accounting & Ingestion Waterfall ===")
    print(res["waterfall"].to_string(index=False))
    print(
        f"\nWrote 3 Parquet artifacts to {args.output_dir}:\n"
        f"  - opex_monthly_panel.parquet:     {len(panel):,} rows x {panel.shape[1]} cols\n"
        f"  - plan_benchmarks.parquet:        {len(res['plan_benchmarks']):,} rows\n"
        f"  - special_items_schedule.parquet: {len(res['special_items_schedule']):,} rows"
    )


if __name__ == "__main__":
    main()
