#!/usr/bin/env python3
"""
34-Check Acceptance Suite for SAP-Inspired Analytical Pharma OPEX Dataset (v2.2)
================================================================================
Fails with exit code 1 if any accounting, anti-leakage, chronological BELNR,
PO-to-invoice reconciliation, SAP FI metadata, causal, or scenario check fails.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd


def median_within_group_corr(df: pd.DataFrame, grp_col: str, x_col: str, y_col: str) -> float:
    corrs = [
        float(g[x_col].corr(g[y_col]))
        for _, g in df.groupby(grp_col)
        if len(g) >= 3 and g[x_col].std() > 0 and g[y_col].std() > 0
    ]
    return float(np.nanmedian(corrs)) if corrs else float("nan")


def run_suite(data_dir: str) -> int:
    required_files = [
        "dim_company_code.csv",
        "dim_plant.csv",
        "dim_cost_center.csv",
        "dim_profit_center.csv",
        "dim_gl_account.csv",
        "dim_fx_rates.csv",
        "fact_opex_drivers.csv",
        "plan_drivers.csv",
        "fact_po_commitments.csv",
        "acdoca_actuals.csv",
        "acdoca_universal_journal.csv",
        "plan_versions.csv",
        "scenario_assumptions.csv",
        "eval_anomaly_labels.csv",
    ]

    checks: List[Tuple[int, str, bool, str]] = []

    def record(num: int, name: str, cond: bool, detail: str) -> None:
        checks.append((num, name, bool(cond), detail))

    # 1. File completeness
    missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    record(1, "All 14 CSV tables present", len(missing) == 0, f"missing={missing}")
    if missing:
        _print_report(checks)
        return 1

    str_cols = {
        "BELNR": str, "STBLG": str, "HKONT": str, "RACCT": str, "DOCLN": str,
        "KOSTL": str, "RCNTR": str, "RFAREA": str, "EBELN": str, "EBELP": str,
        "WERKS": str, "BUKRS": str, "RBUKRS": str, "BSCHL": str, "USNAM": str,
    }
    load = lambda f: pd.read_csv(os.path.join(data_dir, f), dtype=str_cols, low_memory=False)

    dim_comp = load("dim_company_code.csv")
    dim_plant = load("dim_plant.csv")
    dim_cc = load("dim_cost_center.csv")
    dim_pc = load("dim_profit_center.csv")
    dim_gl = load("dim_gl_account.csv")
    dim_fx = load("dim_fx_rates.csv")
    drivers = load("fact_opex_drivers.csv")
    plan_drv = load("plan_drivers.csv")
    po = load("fact_po_commitments.csv")
    act = load("acdoca_actuals.csv")
    uj = load("acdoca_universal_journal.csv")
    plan = load("plan_versions.csv")
    scen = load("scenario_assumptions.csv")
    anom = load("eval_anomaly_labels.csv")

    for df_tmp, cols in [
        (act, ["STBLG", "KOSTL", "RFAREA", "EBELN", "WBS_ELEMENT"]),
        (uj, ["STBLG", "RCNTR", "RFAREA", "EBELN", "PS_PSP_PNR"]),
    ]:
        for c in cols:
            df_tmp[c] = df_tmp[c].fillna("")

    # 2. Double-entry balance in local currency (HSL)
    bal_hsl = float(uj.groupby(["RBUKRS", "RYEAR", "BELNR"])["HSL"].sum().abs().max())
    record(2, "Double-entry balance (HSL == 0 per doc)", bal_hsl < 0.01, f"max_imbalance={bal_hsl:.4f}")

    # 3. Double-entry balance in Group USD (KSL & KSL_BUDGET_FX)
    bal_ksl = float(uj.groupby(["RBUKRS", "RYEAR", "BELNR"])["KSL"].sum().abs().max())
    bal_cc = float(uj.groupby(["RBUKRS", "RYEAR", "BELNR"])["KSL_BUDGET_FX"].sum().abs().max())
    record(3, "Double-entry balance (KSL & Constant FX)", bal_ksl < 0.01 and bal_cc < 0.01, f"ksl={bal_ksl:.4f}, const_fx={bal_cc:.4f}")

    # 4. Row-aligned ECC <-> S/4HANA parity + 6-char DOCLN + Journal PK uniqueness
    sign = np.where(act["SHKZG"] == "S", 1.0, -1.0)
    pk_act_unique = not act.duplicated(subset=["BUKRS", "GJAHR", "BELNR", "BUZEI"]).any()
    pk_uj_unique = not uj.duplicated(subset=["RBUKRS", "RYEAR", "BELNR", "DOCLN"]).any()
    docln_6char = bool(uj["DOCLN"].str.len().eq(6).all())
    keys_aligned = (
        len(act) == len(uj)
        and (act["BUKRS"].values == uj["RBUKRS"].values).all()
        and (act["WERKS"].values == uj["WERKS"].values).all()
        and (act["GJAHR"].values == uj["RYEAR"].values).all()
        and (act["BELNR"].values == uj["BELNR"].values).all()
        and (act["BUZEI"].astype(int).values == uj["DOCLN"].astype(int).values).all()
        and (act["HKONT"].values == uj["RACCT"].values).all()
        and (act["KOSTL"].values == uj["RCNTR"].values).all()
        and (act["RFAREA"].values == uj["RFAREA"].values).all()
        and (act["USNAM"].values == uj["USNAM"].values).all()
        and (act["BSCHL"].values == uj["BSCHL"].values).all()
    )
    amts_aligned = (
        np.allclose(act["DMBTR"] * sign, uj["HSL"], atol=1e-4)
        and np.allclose(act["KSL"] * sign, uj["KSL"], atol=1e-4)
        and np.allclose(act["KSL_BUDGET_FX"] * sign, uj["KSL_BUDGET_FX"], atol=1e-4)
    )
    record(4, "Row-aligned ECC<->S/4 parity, 6-char DOCLN & PKs", pk_act_unique and pk_uj_unique and docln_6char and keys_aligned and amts_aligned, f"rows={len(act):,}, docln6={docln_6char}, pk_ok={pk_act_unique and pk_uj_unique}")

    # 5. Date chronology: BLDAT <= BUDAT <= CPUDT
    bldat = pd.to_datetime(act["BLDAT"])
    budat = pd.to_datetime(act["BUDAT"])
    cpudt = pd.to_datetime(act["CPUDT"])
    chron_viol = int((~((bldat <= budat) & (budat <= cpudt))).sum())
    record(5, "Date chronology (BLDAT <= BUDAT <= CPUDT)", chron_viol == 0, f"violations={chron_viol}")

    # 6. Close-window CPUDT lag present
    late_close = int(((cpudt.dt.year * 12 + cpudt.dt.month) > (budat.dt.year * 12 + budat.dt.month)).sum())
    record(6, "Close-window CPUDT lag present", late_close > 500, f"late_close_lines={late_close:,}")

    # 7. Service dates valid & prepaid amortizations (140000) carry >35d span
    s_start = pd.to_datetime(act["SERVICE_START_DATE"])
    s_end = pd.to_datetime(act["SERVICE_END_DATE"])
    prep_lines = act[act["HKONT"] == "140000"]
    prep_span_ok = len(prep_lines) > 100 and bool(
        ((pd.to_datetime(prep_lines["SERVICE_END_DATE"]) - pd.to_datetime(prep_lines["SERVICE_START_DATE"])).dt.days > 35).all()
    )
    record(7, "Service dates valid & 140000 prepaids (>35d)", bool((s_start <= s_end).all()) and prep_span_ok, f"prepaid_release_lines={len(prep_lines):,}")

    # 8. Hardened ZA <-> AB Accruals (after chronological BELNR remapping!)
    za_l1 = act[(act["BLART"] == "ZA") & (act["BUZEI"] == 1)].copy()
    za_l2 = act[(act["BLART"] == "ZA") & (act["BUZEI"] == 2)].copy()
    ab_l1 = act[(act["BLART"] == "AB") & (act["BUZEI"] == 1)].copy()
    ab_l2 = act[(act["BLART"] == "AB") & (act["BUZEI"] == 2)].copy()

    za_off_ok = len(za_l1) == len(za_l2) and bool((za_l2["HKONT"] == "215000").all())
    ab_off_ok = len(ab_l1) == len(ab_l2) and bool((ab_l2["HKONT"] == "215000").all())
    signs_ok = bool((za_l1["SHKZG"] == "S").all()) and bool((ab_l1["SHKZG"] == "H").all())

    za_paired = za_l1[za_l1["STBLG"] != ""]
    no_orphan_ab = len(za_paired) == len(ab_l1)
    ab_map = {
        (r.BUKRS, int(r.GJAHR), r.BELNR): (r.STBLG, int(r.STJAH), float(r.DMBTR), r.SHKZG)
        for r in ab_l1.itertuples(index=False)
    }
    broken_ptr, lc_mismatch = 0, 0
    for r in za_paired.itertuples(index=False):
        target = ab_map.get((r.BUKRS, int(r.STJAH), r.STBLG))
        if target is None or target[0] != r.BELNR or target[1] != int(r.GJAHR) or target[3] != "H":
            broken_ptr += 1
        elif abs(target[2] - float(r.DMBTR)) > 0.005:
            lc_mismatch += 1

    za_unpaired = za_l1[za_l1["STBLG"] == ""]
    dec25_open = za_unpaired[(za_unpaired["GJAHR"] == 2025) & (za_unpaired["MONAT"] == 12) & (za_unpaired["BUDAT"] == "2025-12-31")]
    pre_dec25_unrev = za_unpaired[~((za_unpaired["GJAHR"] == 2025) & (za_unpaired["MONAT"] == 12))]
    unrev_lbl = anom[anom["ANOMALY_TAG"] == "UNREVERSED_ACCRUAL"]
    pre_dec25_keys = set(zip(pre_dec25_unrev["BUKRS"], pre_dec25_unrev["GJAHR"], pre_dec25_unrev["BELNR"]))
    lbl_unrev_keys = set(zip(unrev_lbl["BUKRS"], unrev_lbl["GJAHR"], unrev_lbl["BELNR"]))
    exact_unrev_match = (len(lbl_unrev_keys) >= 20) and (pre_dec25_keys == lbl_unrev_keys)

    record(
        8,
        "ZA<->AB accruals: 215000, signs, LC clear, no orphans",
        za_off_ok and ab_off_ok and signs_ok and no_orphan_ab and len(za_paired) > 50 and broken_ptr == 0 and lc_mismatch == 0 and len(dec25_open) >= 5 and exact_unrev_match,
        f"pairs={len(za_paired)}/{len(ab_l1)}, broken={broken_ptr}, lc_diff={lc_mismatch}, dec25_open={len(dec25_open)}, exact_unrev={exact_unrev_match}({len(lbl_unrev_keys)})",
    )

    # 9. Special periods 13-16 present
    sp_periods = sorted(uj.loc[uj["POPER"] > 12, "POPER"].unique().tolist())
    record(9, "Special periods 13-16 populated", sp_periods == [13, 14, 15, 16], f"periods={sp_periods}")

    # 10. Multi-currency FX accuracy & zero NaN joins
    uj_fx = uj[["RYEAR", "POPER", "RWCUR", "HSL", "KSL", "KSL_BUDGET_FX"]].copy()
    uj_fx["FX_M"] = np.minimum(uj_fx["POPER"], 12)
    fx_act = dim_fx[dim_fx["RATE_TYPE"] == "ACTUAL_AVG"][["GJAHR", "MONAT", "FROM_CURR", "FX_RATE"]].rename(columns={"FX_RATE": "ACT_FX"})
    fx_bud = dim_fx[dim_fx["RATE_TYPE"] == "BUDGET_PLAN"][["GJAHR", "MONAT", "FROM_CURR", "FX_RATE"]].rename(columns={"FX_RATE": "BUD_FX"})
    uj_fx = uj_fx.merge(fx_act, left_on=["RYEAR", "FX_M", "RWCUR"], right_on=["GJAHR", "MONAT", "FROM_CURR"], how="left")
    uj_fx = uj_fx.merge(fx_bud, left_on=["RYEAR", "FX_M", "RWCUR"], right_on=["GJAHR", "MONAT", "FROM_CURR"], how="left")
    fx_no_nulls = not uj_fx[["ACT_FX", "BUD_FX"]].isna().any().any()
    err_act = float((uj_fx["HSL"] * uj_fx["ACT_FX"] - uj_fx["KSL"]).abs().max()) if fx_no_nulls else float("inf")
    err_bud = float((uj_fx["HSL"] * uj_fx["BUD_FX"] - uj_fx["KSL_BUDGET_FX"]).abs().max()) if fx_no_nulls else float("inf")
    currencies = sorted(uj["RWCUR"].unique().tolist())
    record(10, "Multi-currency FX accuracy & zero NaN joins", fx_no_nulls and currencies == ["EUR", "INR", "USD"] and err_act <= 0.02 and err_bud <= 0.02, f"no_nan={fx_no_nulls}, currs={currencies}, err_act={err_act:.4f}, err_bud={err_bud:.4f}")

    # 11. GAAP Special Items present
    sp_flags = set(uj["SPECIAL_ITEM_FLAG"].unique())
    req_sp = {"NONE", "ACQUIRED_IPRD", "RESTRUCTURING", "MA_INTEGRATION", "LITIGATION"}
    record(11, "GAAP Special Item flags populated", req_sp.issubset(sp_flags), f"flags={sorted(sp_flags)}")

    # 12. Zero orphan master-data keys + all table PK uniqueness
    opex_set = set(dim_gl.loc[dim_gl["ACCOUNT_TYPE"] == "OPEX", "HKONT"])
    opex_rows = uj[uj["RACCT"].isin(opex_set)]
    orphan_cc = set(opex_rows["RCNTR"]) - set(dim_cc["KOSTL"])
    orphan_gl = set(uj["RACCT"]) - set(dim_gl["HKONT"])
    orphan_pc = set(uj["PRCTR"]) - set(dim_pc["PRCTR"])
    orphan_co = set(uj["RBUKRS"]) - set(dim_comp["BUKRS"])
    pks_ok = (
        not plan.duplicated(subset=["GJAHR", "MONAT", "KOSTL", "HKONT", "CYCLE_NAME"]).any()
        and not drivers.duplicated(subset=["GJAHR", "MONAT", "KOSTL"]).any()
        and not plan_drv.duplicated(subset=["GJAHR", "MONAT", "KOSTL"]).any()
        and not dim_fx.duplicated(subset=["GJAHR", "MONAT", "FROM_CURR", "RATE_TYPE"]).any()
        and not scen.duplicated(subset=["SCENARIO_ID", "GJAHR", "MONAT", "KOSTL"]).any()
    )
    record(12, "Zero orphan master keys & all table PKs unique", not (orphan_cc or orphan_gl or orphan_pc or orphan_co) and pks_ok, f"orphans=({len(orphan_cc)},{len(orphan_gl)},{len(orphan_pc)},{len(orphan_co)}), all_pks_ok={pks_ok}")

    # 13. J&J hierarchy & cost-behavior non-null
    cc_nulls = int(dim_cc[["SEGMENT", "FRANCHISE_TA", "REGION", "RFAREA", "WERKS", "DEPARTMENT"]].isna().sum().sum())
    gl_nulls = int(dim_gl[["COST_BEHAVIOR", "OPEX_GROUP"]].isna().sum().sum())
    record(13, "J&J hierarchy & cost-behavior non-null", cc_nulls == 0 and gl_nulls == 0, f"cc_nulls={cc_nulls}, gl_nulls={gl_nulls}")

    # 14. Dense forecastable series count (120-350)
    n_series = opex_rows.groupby(["RCNTR", "RACCT"]).ngroups
    record(14, "Dense forecastable series count (120-350)", 120 <= n_series <= 350, f"active_cc_gl_series={n_series}")

    clean_opex = opex_rows[(opex_rows["POPER"] <= 12) & (opex_rows["SPECIAL_ITEM_FLAG"] == "NONE")]
    monthly_cc_gl = (
        clean_opex.groupby(["RYEAR", "POPER", "RCNTR", "RACCT"], as_index=False)[["HSL", "KSL", "KSL_BUDGET_FX"]]
        .sum()
        .rename(columns={"RYEAR": "GJAHR", "POPER": "MONAT", "RCNTR": "KOSTL", "RACCT": "HKONT"})
    )

    # 15. Causal Driver Check #1: ACTIVE_FTE -> Base Salaries
    sal = monthly_cc_gl[monthly_cc_gl["HKONT"] == "610010"].merge(
        drivers[["GJAHR", "MONAT", "KOSTL", "ACTIVE_FTE"]], on=["GJAHR", "MONAT", "KOSTL"], how="inner"
    )
    r_fte_pool = float(sal["KSL_BUDGET_FX"].corr(sal["ACTIVE_FTE"]))
    r_fte_within = median_within_group_corr(sal, "KOSTL", "KSL_BUDGET_FX", "ACTIVE_FTE")
    record(15, "Causal: ACTIVE_FTE -> Salaries (pool & within-CC)", r_fte_pool > 0.70 and r_fte_within > 0.65, f"pooled_r={r_fte_pool:.3f}, within_cc_med_r={r_fte_within:.3f}")

    # 16. Causal Driver Check #2: Lagged ENROLLED_PATIENTS -> CRO Spend
    drv_lag = drivers[["GJAHR", "MONAT", "KOSTL", "ENROLLED_PATIENTS"]].copy()
    ym_idx = drv_lag["GJAHR"] * 12 + (drv_lag["MONAT"] - 1) + 1
    drv_lag["GJAHR"] = ym_idx // 12
    drv_lag["MONAT"] = (ym_idx % 12) + 1
    cro = monthly_cc_gl[monthly_cc_gl["HKONT"] == "620010"].merge(drv_lag, on=["GJAHR", "MONAT", "KOSTL"], how="inner")
    r_pat_pool = float(cro["KSL_BUDGET_FX"].corr(cro["ENROLLED_PATIENTS"]))
    r_pat_within = median_within_group_corr(cro, "KOSTL", "KSL_BUDGET_FX", "ENROLLED_PATIENTS")
    record(16, "Causal: Lagged PATIENTS -> CRO (pool & within-CC)", r_pat_pool > 0.60 and r_pat_within > 0.55, f"pooled_r={r_pat_pool:.3f}, within_cc_med_r={r_pat_within:.3f}")

    # 17. Leading Indicator Check: Point-in-time COOI Open POs
    po_eval_df = po.copy()
    po_eval_df["GR_YR"] = po_eval_df["EXPECTED_GR_MONTH"].str.slice(0, 4).astype(int)
    po_eval_df["GR_M"] = po_eval_df["EXPECTED_GR_MONTH"].str.slice(5, 7).astype(int)
    snap_dt = pd.to_datetime(po_eval_df["SNAPSHOT_DATE"])
    po_pit = po_eval_df[(po_eval_df["GR_YR"] * 12 + po_eval_df["GR_M"]) - (snap_dt.dt.year * 12 + snap_dt.dt.month) == 1]
    po_m = po_pit.groupby(["GR_YR", "GR_M", "KOSTL", "HKONT"], as_index=False)["OPEN_COMMITMENT_USD"].sum()
    po_mrg = monthly_cc_gl.merge(
        po_m, left_on=["GJAHR", "MONAT", "KOSTL", "HKONT"], right_on=["GR_YR", "GR_M", "KOSTL", "HKONT"], how="inner"
    )
    r_po = float(po_mrg["KSL"].corr(po_mrg["OPEN_COMMITMENT_USD"]))
    record(17, "Leading signal: PIT COOI Open POs -> GL (r > 0.70)", r_po > 0.70, f"pit_pearson_r={r_po:.3f}")

    # 18. Zero Leakage in BP (blind to unexpected trial hold)
    bp = plan[plan["CYCLE_NAME"] == "BP"][["GJAHR", "MONAT", "KOSTL", "HKONT", "KSL_BUDGET", "SNAPSHOT_DATE"]]
    bp_pause = bp[(bp["KOSTL"] == "CC1010") & (bp["HKONT"] == "620010") & (bp["GJAHR"] == 2024) & (bp["MONAT"].isin([4, 5]))].sort_values("MONAT")
    act_pause = monthly_cc_gl[(monthly_cc_gl["KOSTL"] == "CC1010") & (monthly_cc_gl["HKONT"] == "620010") & (monthly_cc_gl["GJAHR"] == 2024) & (monthly_cc_gl["MONAT"].isin([4, 5]))].sort_values("MONAT")
    bp_ratio = float(bp_pause["KSL_BUDGET"].iloc[1] / bp_pause["KSL_BUDGET"].iloc[0])
    act_ratio = float(act_pause["KSL_BUDGET_FX"].iloc[1] / act_pause["KSL_BUDGET_FX"].iloc[0])
    record(18, "Zero leakage in BP (blind to trial hold)", bp_ratio > 0.92 and act_ratio < 0.60, f"BP_May/Apr={bp_ratio:.2f}, Actual_May/Apr={act_ratio:.2f}")

    # 19. J&J planning cadence complete
    cycles = set(plan["CYCLE_NAME"].unique())
    record(19, "J&J planning cadence (BP/Q1_RF/JU/Q3_RF/NU)", cycles == {"BP", "Q1_RF", "JU", "Q3_RF", "NU"}, f"cycles={sorted(cycles)}")

    # 20. Open-Month Forecast Coherence
    def open_wape(cycle: str, months: List[int]) -> float:
        sub = plan[(plan["CYCLE_NAME"] == cycle) & (plan["GJAHR"].isin([2023, 2024, 2025])) & (plan["MONAT"].isin(months))]
        mrg = monthly_cc_gl.merge(sub[["GJAHR", "MONAT", "KOSTL", "HKONT", "KSL_BUDGET"]], on=["GJAHR", "MONAT", "KOSTL", "HKONT"], how="inner")
        return float((mrg["KSL_BUDGET_FX"] - mrg["KSL_BUDGET"]).abs().sum() / mrg["KSL_BUDGET_FX"].abs().sum())

    beat_bp_own = all(
        open_wape(cyc, list(range(cut + 1, 13))) < open_wape("BP", list(range(cut + 1, 13)))
        for cyc, cut in [("Q1_RF", 3), ("JU", 5), ("Q3_RF", 9), ("NU", 10)]
    )
    w_bp_nd, w_ju_nd, w_nu_nd = open_wape("BP", [11, 12]), open_wape("JU", [11, 12]), open_wape("NU", [11, 12])
    record(20, "Open-month WAPE coherence (NU < JU < BP on M11-12)", beat_bp_own and (w_nu_nd < w_ju_nd < w_bp_nd), f"all_beat_BP={beat_bp_own}, M11-12: BP={w_bp_nd:.2%}, JU={w_ju_nd:.2%}, NU={w_nu_nd:.2%}")

    # 21. Physical separation & expanded count of anomaly labels
    sep_ok = ("ANOMALY_TAG" not in act.columns) and ("ANOMALY_TAG" not in uj.columns) and ("ANOMALY_TAG" in anom.columns)
    record(21, "Anomalies isolated from ACDOCA (95-130 rows)", sep_ok and (95 <= len(anom) <= 130), f"labeled_anomalies={len(anom)}")

    # 22. All 4 anomaly classes linked to ACDOCA, metadata & cross-month duplicates
    req_tags = {"DUPLICATE_INVOICE", "UNREVERSED_ACCRUAL", "Q4_BUDGET_FLUSH", "TRIAL_PAUSE_DROP"}
    act_keys = set(zip(act["BUKRS"], act["GJAHR"], act["BELNR"], act["BUZEI"]))
    anom_keys = set(zip(anom["BUKRS"], anom["GJAHR"], anom["BELNR"], anom["BUZEI"]))
    meta_ok = {"GRANULARITY", "ESTIMATED_IMPACT_USD"}.issubset(anom.columns) and not anom[["GRANULARITY", "ESTIMATED_IMPACT_USD"]].isna().any().any()
    cross_dups = int(anom["DESCRIPTION"].str.contains("Cross-month", case=False, na=False).sum())
    record(22, "4 anomaly classes, metadata & cross-month dups", set(anom["ANOMALY_TAG"].unique()) == req_tags and anom_keys.issubset(act_keys) and meta_ok and cross_dups >= 5, f"cross_month_dups={cross_dups}, counts={anom['ANOMALY_TAG'].value_counts().to_dict()}")

    # 23. FY2026 What-If Scenario Coverage
    req_sc = {"BASE", "HIRING_FREEZE_Q2", "PHASE3_DELAY_1Q", "USD_STRENGTHEN_10PCT", "LAUNCH_ACCEL"}
    record(23, "FY2026 What-If scenarios (5 cases)", set(scen["SCENARIO_ID"].unique()) == req_sc and set(scen["GJAHR"].unique()) == {2026}, f"scenarios={sorted(scen['SCENARIO_ID'].unique())}")

    # 24. Domain sanity & FY2022-FY2025 actual coverage
    sanity_ok = (
        (drivers["ACTIVE_FTE"] >= 0).all()
        and (drivers["ENROLLED_PATIENTS"] >= 0).all()
        and (dim_fx["FX_RATE"] > 0).all()
        and sorted(uj["RYEAR"].unique().tolist()) == [2022, 2023, 2024, 2025]
        and sorted(drivers["GJAHR"].unique().tolist()) == [2022, 2023, 2024, 2025]
    )
    record(24, "Domain sanity & FY22-25 actual coverage", sanity_ok, f"driver_rows={len(drivers):,}, plan_driver_rows={len(plan_drv):,}")

    # 25. Snapshot Cutoff Anti-Leakage (reads plan_versions SNAPSHOT_DATE + both sides of ms_decoupled)
    bp_snap_ok = all(set(bp.loc[bp["GJAHR"] == yr, "SNAPSHOT_DATE"].unique()) == {f"{yr - 1}-10-15"} for yr in [2022, 2023, 2024, 2025, 2026])
    rf_cut_ok = True
    for yr in [2022, 2023, 2024, 2025]:
        for cyc, cut_m, expected_md in [("Q1_RF", 3, "04-10"), ("JU", 5, "06-15"), ("Q3_RF", 9, "10-10"), ("NU", 10, "11-15")]:
            sub_snaps = set(plan.loc[(plan["GJAHR"] == yr) & (plan["CYCLE_NAME"] == cyc), "SNAPSHOT_DATE"].unique())
            if sub_snaps != {f"{yr}-{expected_md}"}:
                rf_cut_ok = False
                continue
            snap_from_plan = next(iter(sub_snaps))
            closed = uj[(uj["RYEAR"] == yr) & (uj["POPER"] <= cut_m)]
            if len(closed) > 0 and closed["CPUDT"].max() > snap_from_plan:
                rf_cut_ok = False

    bp_m9 = float(bp[(bp["KOSTL"] == "CC1010") & (bp["HKONT"] == "620030") & (bp["GJAHR"] == 2025) & (bp["MONAT"] == 9)]["KSL_BUDGET"].sum())
    bp_m10 = float(bp[(bp["KOSTL"] == "CC1010") & (bp["HKONT"] == "620030") & (bp["GJAHR"] == 2025) & (bp["MONAT"] == 10)]["KSL_BUDGET"].sum())
    act_m9 = float(monthly_cc_gl[(monthly_cc_gl["KOSTL"] == "CC1010") & (monthly_cc_gl["HKONT"] == "620030") & (monthly_cc_gl["GJAHR"] == 2025) & (monthly_cc_gl["MONAT"] == 9)]["KSL_BUDGET_FX"].sum())
    act_m10 = float(monthly_cc_gl[(monthly_cc_gl["KOSTL"] == "CC1010") & (monthly_cc_gl["HKONT"] == "620030") & (monthly_cc_gl["GJAHR"] == 2025) & (monthly_cc_gl["MONAT"] == 10)]["KSL_BUDGET_FX"].sum())
    ms_decoupled = (bp_m9 > 500_000.0) and (bp_m10 == 0.0) and (act_m9 == 0.0) and (act_m10 > 500_000.0)

    dec25_fte = drivers[(drivers["GJAHR"] == 2025) & (drivers["MONAT"] == 12)].set_index("KOSTL")["ACTIVE_FTE"]
    jan26_fte = scen[(scen["SCENARIO_ID"] == "BASE") & (scen["MONAT"] == 1)].set_index("KOSTL")["ACTIVE_FTE"]
    fte_jump = float((jan26_fte - dec25_fte).abs().max())
    scen_snap_ok = set(scen["SNAPSHOT_DATE"].unique()) == {"2025-12-31"} and fte_jump < 5.0
    record(25, "Snapshot anti-leakage (plan dates, CPUDT, milestones)", bp_snap_ok and rf_cut_ok and ms_decoupled and scen_snap_ok, f"bp_snap={bp_snap_ok}, rf_cut={rf_cut_ok}, ms_both_sides={ms_decoupled}, fte_step={fte_jump:.1f}")

    # 26. SAP Universal Journal Structure (BS selected via dim_gl.ACCOUNT_TYPE == "BS")
    has_both_yrs = ("GJAHR" in uj.columns) and ("RYEAR" in uj.columns) and bool((uj["GJAHR"] == uj["RYEAR"]).all())
    uj_budat = pd.to_datetime(uj["BUDAT"])
    p1_12, p13_16 = uj["POPER"] <= 12, uj["POPER"] > 12
    budat_1_12_ok = bool(((uj_budat[p1_12].dt.year == uj.loc[p1_12, "GJAHR"]) & (uj_budat[p1_12].dt.month == uj.loc[p1_12, "POPER"])).all())
    budat_13_16_ok = bool((uj.loc[p13_16, "BUDAT"] == uj.loc[p13_16, "GJAHR"].astype(str) + "-12-31").all())
    bs_gl_set = set(dim_gl.loc[dim_gl["ACCOUNT_TYPE"] == "BS", "HKONT"])
    bs_lines = uj[uj["RACCT"].isin(bs_gl_set)]
    bs_clean = len(bs_lines) > 0 and bool(((bs_lines["RFAREA"] == "") & (bs_lines["RCNTR"] == "")).all())
    record(26, "UJ structure (GJAHR+RYEAR, BUDAT, BS via GL type)", has_both_yrs and budat_1_12_ok and budat_13_16_ok and bs_clean, f"yrs={has_both_yrs}, budat={budat_1_12_ok and budat_13_16_ok}, bs_lines={len(bs_lines):,}, bs_clean={bs_clean}")

    # 27. Functional Posting Matrix + Cost-Center RFAREA Agreement
    gl_allow = dict(zip(dim_gl["HKONT"], dim_gl["ALLOWED_RFAREAS"].str.split("|")))
    cc_rfarea = dict(zip(dim_cc["KOSTL"], dim_cc["RFAREA"]))
    opex_allow_viol = sum(1 for r in opex_rows[["RACCT", "RFAREA"]].itertuples(index=False) if r.RFAREA not in gl_allow.get(r.RACCT, []))
    opex_cc_rf_viol = int((opex_rows["RCNTR"].map(cc_rfarea) != opex_rows["RFAREA"]).sum())
    record(27, "Functional matrix (ALLOWED_RFAREAS & CC RFAREA)", opex_allow_viol == 0 and opex_cc_rf_viol == 0, f"gl_rf_viol={opex_allow_viol}, cc_rf_viol={opex_cc_rf_viol}")

    # 28. PO Lifecycle PK, Inequalities, Status Agreement & 2025 Cap
    po_create_dt = pd.to_datetime(po["PO_CREATE_DATE"])
    gr_first_dt = pd.to_datetime(po["EXPECTED_GR_MONTH"] + "-01")
    snap_po_dt = pd.to_datetime(po["SNAPSHOT_DATE"])
    po_pk_ok = not po.duplicated(subset=["EBELN", "EBELP", "SNAPSHOT_DATE"]).any()
    po_chron = bool((po_create_dt < gr_first_dt).all() and (snap_po_dt >= po_create_dt).all())
    po_cap = bool((po_create_dt <= pd.Timestamp("2025-12-31")).all() and (snap_po_dt <= pd.Timestamp("2025-12-31")).all())
    po_ineq = bool(
        (po["INVOICED_AMOUNT_USD"] >= -1e-6).all()
        and (po["INVOICED_AMOUNT_USD"] <= po["GR_AMOUNT_USD"] + 1e-6).all()
        and (po["GR_AMOUNT_USD"] <= po["ORIGINAL_COMMITMENT_USD"] + 1e-6).all()
        and (po["OPEN_COMMITMENT_USD"] >= -1e-6).all()
    )
    po_ident = float(((po["ORIGINAL_COMMITMENT_USD"] - po["GR_AMOUNT_USD"]) - po["OPEN_COMMITMENT_USD"]).abs().max()) < 0.01
    st_closed, st_open, st_part = po[po["PO_STATUS"] == "CLOSED"], po[po["PO_STATUS"] == "OPEN"], po[po["PO_STATUS"] == "PARTIALLY_RECEIVED"]
    status_agree = (
        len(st_closed) > 0 and len(st_open) > 0 and len(st_part) > 0
        and bool((st_closed["OPEN_COMMITMENT_USD"].abs() < 0.01).all())
        and bool((st_open["GR_AMOUNT_USD"].abs() < 0.01).all())
        and bool(((st_part["GR_AMOUNT_USD"] > 0) & (st_part["GR_AMOUNT_USD"] < st_part["ORIGINAL_COMMITMENT_USD"])).all())
    )
    open_2026 = int(((po["SNAPSHOT_DATE"] == "2025-12-31") & (po["EXPECTED_GR_MONTH"].isin(["2026-01", "2026-02"])) & (po["OPEN_COMMITMENT_USD"] > 0)).sum())
    record(28, "PO lifecycle PK, inequalities, statuses & 2025 cap", po_pk_ok and po_chron and po_cap and po_ineq and po_ident and status_agree and open_2026 > 50, f"pk={po_pk_ok}, chron={po_chron}, ineq={po_ineq}, status_ok={status_agree}, open_fy26={open_2026}")

    # 29. Compounded WAGE_INDEX & April-Only Merit Monotonicity
    w_all = pd.concat(
        [
            drivers[["GJAHR", "MONAT", "MERIT_CYCLE_FLAG", "WAGE_INDEX"]].drop_duplicates(),
            scen[scen["SCENARIO_ID"] == "BASE"][["GJAHR", "MONAT", "MERIT_CYCLE_FLAG", "WAGE_INDEX"]].drop_duplicates(),
        ],
        ignore_index=True,
    ).sort_values(["GJAHR", "MONAT"]).reset_index(drop=True)
    diffs = w_all["WAGE_INDEX"].diff().fillna(0.0)
    non_dec = bool((diffs >= -1e-9).all())
    apr_only_steps = bool(((diffs > 1e-9) == (w_all["MERIT_CYCLE_FLAG"] == 1)).all())
    record(29, "Compounded WAGE_INDEX strictly steps only in April", non_dec and apr_only_steps, f"start={w_all['WAGE_INDEX'].iloc[0]:.2f}, end={w_all['WAGE_INDEX'].iloc[-1]:.2f}")

    # 30. Plan FX Math, Oct-Locked Zero FX Leakage & Closed-Month HSL
    fx_plan_err = float((plan["DMBTR_PLAN"] * plan["PLAN_FX_RATE"] - plan["KSL_BUDGET"]).abs().max())
    bud_fx_const = bool((dim_fx[dim_fx["RATE_TYPE"] == "BUDGET_PLAN"].groupby(["GJAHR", "FROM_CURR"])["FX_RATE"].nunique() == 1).all())
    fx_oct_act = dim_fx[(dim_fx["RATE_TYPE"] == "ACTUAL_AVG") & (dim_fx["MONAT"] == 10)].set_index(["GJAHR", "FROM_CURR"])["FX_RATE"]
    fx_bp_m1 = dim_fx[(dim_fx["RATE_TYPE"] == "BUDGET_PLAN") & (dim_fx["MONAT"] == 1)].set_index(["GJAHR", "FROM_CURR"])["FX_RATE"]
    fx_zero_leak = all(
        abs(float(fx_bp_m1.loc[(yr, curr)]) - float(fx_oct_act.loc[(yr - 1, curr)])) <= 1e-6
        for yr in [2023, 2024, 2025, 2026]
        for curr in ["USD", "EUR", "INR"]
    )
    closed_errs = []
    for cyc, cut_m in [("Q1_RF", 3), ("JU", 5), ("Q3_RF", 9), ("NU", 10)]:
        sub = plan[(plan["CYCLE_NAME"] == cyc) & (plan["MONAT"] <= cut_m)]
        mrg = sub.merge(monthly_cc_gl[["GJAHR", "MONAT", "KOSTL", "HKONT", "HSL"]], on=["GJAHR", "MONAT", "KOSTL", "HKONT"], how="left")
        mrg["HSL"] = mrg["HSL"].fillna(0.0)
        closed_errs.append(float((mrg["DMBTR_PLAN"] - mrg["HSL"]).abs().max()))
    max_closed_err = max(closed_errs)
    record(30, "Plan FX math, Oct-locked zero FX leak & closed HSL", fx_plan_err <= 0.02 and bud_fx_const and fx_zero_leak and max_closed_err <= 0.02, f"fx_err={fx_plan_err:.4f}, zero_fx_leak={fx_zero_leak}, closed_hsl_err={max_closed_err:.4f}")

    # 31. Plant (WERKS) & BUKRS FK Integrity across ALL tables
    valid_plants = set(dim_plant["WERKS"])
    plant_to_co = dict(zip(dim_plant["WERKS"], dim_plant["BUKRS"]))
    tables_to_check = [
        (dim_cc, "BUKRS"), (po, "BUKRS"), (act, "BUKRS"), (uj, "RBUKRS"),
        (drivers, "BUKRS"), (plan_drv, "BUKRS"), (plan, "BUKRS"), (scen, "BUKRS"),
    ]
    werks_ok = all(
        set(df_t["WERKS"]).issubset(valid_plants) and bool((df_t["WERKS"].map(plant_to_co) == df_t[co_col]).all())
        for df_t, co_col in tables_to_check
    )
    record(31, "Plant (WERKS) & BUKRS FK integrity (all tables)", werks_ok, f"tables_checked={len(tables_to_check)}, plants={len(valid_plants)}")

    # 32. Scenario Invariants & All-4 Anomaly Signatures
    schema_parity = set(drivers.columns).issubset(set(scen.columns))
    sc_base = scen[scen["SCENARIO_ID"] == "BASE"].sort_values(["KOSTL", "MONAT"]).reset_index(drop=True)
    sc_hf = scen[scen["SCENARIO_ID"] == "HIRING_FREEZE_Q2"].sort_values(["KOSTL", "MONAT"]).reset_index(drop=True)
    sc_p3 = scen[scen["SCENARIO_ID"] == "PHASE3_DELAY_1Q"].sort_values(["KOSTL", "MONAT"]).reset_index(drop=True)
    sc_fx = scen[scen["SCENARIO_ID"] == "USD_STRENGTHEN_10PCT"].sort_values(["KOSTL", "MONAT"]).reset_index(drop=True)
    sc_la = scen[scen["SCENARIO_ID"] == "LAUNCH_ACCEL"].sort_values(["KOSTL", "MONAT"]).reset_index(drop=True)

    hf_zero_hires = bool((sc_hf.loc[sc_hf["MONAT"] >= 4, "PLANNED_HIRES"] == 0).all())
    hf_diffs = sc_hf.groupby("KOSTL")["ACTIVE_FTE"].diff()
    hf_non_inc = bool((hf_diffs[sc_hf["MONAT"] >= 4] <= 1e-9).all())
    m4_drop = float(1.0 - sc_hf.loc[sc_hf["MONAT"] == 4, "ACTIVE_FTE"].sum() / sc_base.loc[sc_base["MONAT"] == 4, "ACTIVE_FTE"].sum())
    m12_drop = float(1.0 - sc_hf.loc[sc_hf["MONAT"] == 12, "ACTIVE_FTE"].sum() / sc_base.loc[sc_base["MONAT"] == 12, "ACTIVE_FTE"].sum())
    hf_ok = hf_zero_hires and hf_non_inc and (0.005 < m4_drop < 0.025) and (m12_drop > 0.05)

    p3_only = sc_p3[sc_p3["TRIAL_PHASE"] == "P3"].sort_values(["KOSTL", "MONAT"])
    p3_pct_chg = p3_only.groupby("KOSTL")["ENROLLED_PATIENTS"].pct_change()
    p3_no_april_drop = bool((p3_pct_chg[p3_only["MONAT"] == 4] >= -1e-6).all())
    p3_smooth_m4_12 = bool((p3_pct_chg[p3_only["MONAT"] >= 4].abs() < 0.05).all())
    p3_q23_drop = float(p3_only.loc[p3_only["MONAT"].between(4, 9), "ENROLLED_PATIENTS"].sum()) < float(
        sc_base.loc[(sc_base["TRIAL_PHASE"] == "P3") & (sc_base["MONAT"].between(4, 9)), "ENROLLED_PATIENTS"].sum()
    )
    ms_lookup = {(r.KOSTL, int(r.MONAT)): r.MILESTONE_EVENT for r in p3_only.itertuples(index=False)}
    p3_ms_ok = (
        ms_lookup[("CC1010", 8)] == "DB_LOCK"
        and ms_lookup[("CC1030", 10)] == "FDA_FILING"
        and ms_lookup[("CC2010", 9)] == "FPI"
        and ms_lookup[("CC3010", 10)] == "NONE"
    )
    p3_ok = p3_no_april_drop and p3_smooth_m4_12 and p3_q23_drop and p3_ms_ok

    fx_ops_same = bool((sc_fx["ACTIVE_FTE"] == sc_base["ACTIVE_FTE"]).all() and np.allclose(sc_fx["FX_EUR_USD"], sc_base["FX_EUR_USD"] * 0.90, atol=1e-5))
    la_hcp_up = float(sc_la.loc[sc_la["MONAT"].between(3, 9), "HCP_EVENTS_COUNT"].sum()) > float(sc_base.loc[sc_base["MONAT"].between(3, 9), "HCP_EVENTS_COUNT"].sum())

    kr_opex = act[(act["BLART"] == "KR") & (act["BUZEI"] == 1)]
    dup_sig_counts = kr_opex.groupby(["BUKRS", "LIFNR", "BLDAT", "DMBTR"]).size()
    dup_keys_gt1 = set(dup_sig_counts[dup_sig_counts >= 2].index)
    dup_anoms = anom[anom["ANOMALY_TAG"] == "DUPLICATE_INVOICE"].merge(
        kr_opex[["BUKRS", "GJAHR", "BELNR", "BUZEI", "LIFNR", "BLDAT", "DMBTR"]],
        on=["BUKRS", "GJAHR", "BELNR", "BUZEI"], how="left",
    )
    dup_sig_ok = all((r.BUKRS, r.LIFNR, r.BLDAT, r.DMBTR) in dup_keys_gt1 for r in dup_anoms.itertuples(index=False))
    q4_sig_ok = bool(anom.loc[anom["ANOMALY_TAG"] == "Q4_BUDGET_FLUSH", "MONAT"].isin([11, 12]).all() and (anom.loc[anom["ANOMALY_TAG"] == "Q4_BUDGET_FLUSH", "ESTIMATED_IMPACT_USD"] > 0).all())
    tp_sig_ok = bool(anom.loc[anom["ANOMALY_TAG"] == "TRIAL_PAUSE_DROP", "HKONT"].isin(["620010", "620020"]).all() and (anom.loc[anom["ANOMALY_TAG"] == "TRIAL_PAUSE_DROP", "ESTIMATED_IMPACT_USD"] < 0).all())
    unrev_sig_ok = bool((anom.loc[anom["ANOMALY_TAG"] == "UNREVERSED_ACCRUAL", "ESTIMATED_IMPACT_USD"] > 0).all())

    record(32, "Scenario invariants & all-4 anomaly signatures", schema_parity and hf_ok and p3_ok and fx_ops_same and la_hcp_up and dup_sig_ok and q4_sig_ok and tp_sig_ok and unrev_sig_ok, f"hf={hf_ok}({m4_drop:.1%}->{m12_drop:.1%}), p3={p3_ok}, fx={fx_ops_same}, la={la_hcp_up}, anoms={dup_sig_ok and q4_sig_ok and tp_sig_ok and unrev_sig_ok}")

    # --- NEW v2.2 CHECKS (Checks 33 & 34) ---

    # 33. Chronological Ledger Sort, Sequential BELNR & Zero Anomaly Position Clustering
    act_sorted_ok = act[["BUKRS", "GJAHR", "MONAT", "BUDAT", "BELNR", "BUZEI"]].equals(
        act[["BUKRS", "GJAHR", "MONAT", "BUDAT", "BELNR", "BUZEI"]].sort_values(["BUKRS", "GJAHR", "MONAT", "BUDAT", "BELNR", "BUZEI"]).reset_index(drop=True)
    )
    doc_l1 = act[act["BUZEI"] == 1].copy()
    doc_l1["BELNR_INT"] = doc_l1["BELNR"].astype(int)
    belnr_mono_ok = True
    for _, grp in doc_l1.groupby(["BUKRS", "GJAHR"]):
        expected_seq = np.arange(100000001, 100000001 + len(grp))
        if not np.array_equal(grp["BELNR_INT"].values, expected_seq):
            belnr_mono_ok = False
            break
    # Verify anomalies are distributed across the fiscal year BELNR range (not clustered at tail > 0.95)
    doc_l1["RANK_PCT"] = doc_l1.groupby(["BUKRS", "GJAHR"])["BELNR_INT"].rank(pct=True)
    anom_ranks = anom.merge(doc_l1[["BUKRS", "GJAHR", "BELNR", "RANK_PCT"]], on=["BUKRS", "GJAHR", "BELNR"], how="left")["RANK_PCT"]
    med_rank, min_rank = float(anom_ranks.median()), float(anom_ranks.min())
    not_clustered = (0.25 <= med_rank <= 0.85) and (min_rank < 0.25)
    record(33, "Chronological BELNR order & unclustered anomalies", act_sorted_ok and belnr_mono_ok and not_clustered, f"sorted={act_sorted_ok}, seq_belnr={belnr_mono_ok}, anom_rank_min/med={min_rank:.2f}/{med_rank:.2f}")

    # 34. 100% Closed PO-to-ACDOCA EBELN Reconciliation + SAP FI Metadata & Traffic Realism
    closed_pos = set(po.loc[po["PO_STATUS"] == "CLOSED", "EBELN"].unique())
    invoiced_pos = set(act.loc[(act["BLART"] == "KR") & (act["EBELN"] != ""), "EBELN"].unique())
    orphan_closed_pos = closed_pos - invoiced_pos
    po_reconciled = len(closed_pos) > 1000 and len(orphan_closed_pos) == 0

    pr_l2 = act[(act["BLART"] == "PR") & (act["BUZEI"] == 2)]
    pr_clearing_ok = len(pr_l2) > 100 and bool((pr_l2["HKONT"] == "210000").all()) and bool((pr_l2["USNAM"] == "BATCH_PAYROLL").all())
    ab_batch_ok = bool((act.loc[act["BLART"] == "AB", "USNAM"] == "BATCH_REV").all())
    bschl_ok = bool(((act["SHKZG"] == "S") == (act["BSCHL"] == "40")).all() and ((act["SHKZG"] == "H") == (act["BSCHL"] == "50")).all())

    kr_dates = pd.to_datetime(kr_opex["BUDAT"])
    wknd_share = float((kr_dates.dt.weekday >= 5).mean())
    days_1_3_present = int((kr_dates.dt.day <= 3).sum()) > 100
    traffic_ok = (wknd_share < 0.08) and days_1_3_present

    record(
        34,
        "100% Closed PO->ACDOCA EBELN, 210000 PR, USNAM/BSCHL",
        po_reconciled and pr_clearing_ok and ab_batch_ok and bschl_ok and traffic_ok,
        f"orphan_closed_POs={len(orphan_closed_pos)}/{len(closed_pos)}, pr210000={pr_clearing_ok}, bschl={bschl_ok}, wknd_KR={wknd_share:.1%}",
    )

    # 35. ERP Traffic & Distribution Realism (Month-End Surge, PO Right-Skew & GL Heterogeneity)
    kr_reg = kr_opex[kr_opex["SPECIAL_ITEM_FLAG"] == "NONE"].copy()
    kr_reg_dt = pd.to_datetime(kr_reg["BUDAT"])
    me_share = float((kr_reg_dt.dt.day > (kr_reg_dt.dt.days_in_month - 3)).mean())

    po_gls = set(dim_gl.loc[dim_gl["USES_PO"].astype(bool), "HKONT"])
    kr_po = kr_reg[kr_reg["HKONT"].isin(po_gls)]
    po_skew_ratios = kr_po.groupby("HKONT")["KSL"].agg(lambda s: float(s.mean() / s.median()))
    med_po_skew = float(po_skew_ratios.median())

    opex_reg_l1 = act[
        (act["BUZEI"] == 1)
        & (act["HKONT"].isin(opex_set))
        & (act["SPECIAL_ITEM_FLAG"] == "NONE")
        & (act["MONAT"] <= 12)
    ]
    gl_medians = opex_reg_l1.groupby("HKONT")["KSL"].median()
    gl_het_ratio = float(gl_medians.max() / gl_medians.min())

    record(
        35,
        "ERP traffic & skew (ME 15-60%, skew 1.25-2.6, het>5x)",
        (0.15 <= me_share <= 0.60) and (1.25 <= med_po_skew <= 2.60) and (gl_het_ratio > 5.0),
        f"me_kr={me_share:.1%}, med_po_skew={med_po_skew:.2f}, gl_het={gl_het_ratio:.1f}x",
    )

    # 36. Per-Year Period Completeness (48 + 16), Bounded BP WAPE & Pharma Functional Mix
    reg_pairs = uj.loc[uj["POPER"] <= 12, ["RYEAR", "POPER"]].drop_duplicates().shape[0]
    sp_pairs = uj.loc[uj["POPER"] > 12, ["RYEAR", "POPER"]].drop_duplicates().shape[0]

    bp_fy = plan[(plan["CYCLE_NAME"] == "BP") & (plan["GJAHR"].isin([2022, 2023, 2024, 2025]))]
    bp_mrg = monthly_cc_gl.merge(
        bp_fy[["GJAHR", "MONAT", "KOSTL", "HKONT", "KSL_BUDGET"]],
        on=["GJAHR", "MONAT", "KOSTL", "HKONT"],
        how="inner",
    )
    bp_fy_wape = float((bp_mrg["KSL_BUDGET_FX"] - bp_mrg["KSL_BUDGET"]).abs().sum() / bp_mrg["KSL_BUDGET_FX"].abs().sum())

    ann_rf = clean_opex.groupby(["RYEAR", "RFAREA"])["KSL"].sum().unstack(fill_value=0.0)
    ann_tot = ann_rf.sum(axis=1).sort_index()
    rf_shares = ann_rf.div(ann_tot, axis=0)
    rd_min = float(rf_shares["R&D"].min())
    comm_min = float(rf_shares["Commercial_SGA"].min())
    yoy_growth = ann_tot.pct_change().dropna()
    smooth_growth = bool((ann_tot > 0).all() and (yoy_growth > 0.0).all() and (yoy_growth < 0.25).all())

    record(
        36,
        "48+16 periods, BP WAPE [3-30%] & pharma RFAREA mix",
        (reg_pairs == 48) and (sp_pairs == 16) and (0.03 <= bp_fy_wape <= 0.30) and (rd_min > 0.20) and (comm_min > 0.20) and smooth_growth,
        f"reg/sp={reg_pairs}/{sp_pairs}, bp_wape={bp_fy_wape:.2%}, min_RD/Comm={rd_min:.1%}/{comm_min:.1%}, yoy={yoy_growth.min():.1%}..{yoy_growth.max():.1%}",
    )

    _print_report(checks)
    failed = sum(1 for _, _, ok, _ in checks if not ok)
    return 0 if failed == 0 else 1


def _print_report(checks: List[Tuple[int, str, bool, str]]) -> None:
    print("\n" + "=" * 110)
    print(f"{'#':<4} {'STATUS':<7} {'CHECK NAME':<55} DETAILS")
    print("-" * 110)
    for num, name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"{num:<4} [{status}]  {name:<55} {detail}")
    print("=" * 110)
    passed = sum(1 for _, _, ok, _ in checks if ok)
    print(f"RESULT: {passed}/{len(checks)} checks passed.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SAP-Inspired Analytical Pharma OPEX Dataset (v2.2)")
    parser.add_argument("data_dir", nargs="?", default="./data/raw/pharma_v22", help="Dataset directory")
    args = parser.parse_args()
    sys.exit(run_suite(args.data_dir))


if __name__ == "__main__":
    main()