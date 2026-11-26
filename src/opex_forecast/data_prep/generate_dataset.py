#!/usr/bin/env python3
"""
SAP-Inspired Analytical Pharmaceutical OPEX Generator (v2.2, Final Freeze)
==========================================================================
Driver-based synthetic OPEX generator producing an SAP-inspired analytical
Universal Journal dataset (inspired by SAP S/4HANA ACDOCA, ECC BKPF/BSEG,
and COOI/EKPO commitments; KSL_BUDGET_FX is an analytical constant-FX
extension and dim_fx_rates abstracts TCURR/TCURF/TCURX).

Planning Cadence (J&J Aligned):
  - BP    : Annual Business Plan (0+12, locked Oct {yr-1} with Nov-Dec bridge)
  - Q1_RF : Q1 Latest Thinking (3+9, CPUDT <= {yr}-04-10)
  - JU    : June Update (5+7, CPUDT <= {yr}-06-15)
  - Q3_RF : Q3 Latest Thinking (9+3, CPUDT <= {yr}-10-10)
  - NU    : November Update (10+2, CPUDT <= {yr}-11-15)
"""

from __future__ import annotations

import argparse
import calendar
import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DRIVER_FEATURE_COLS = [
    "GJAHR", "MONAT", "BUKRS", "WERKS", "KOSTL", "WBS_ELEMENT",
    "ACTIVE_FTE", "PLANNED_HIRES", "FIELD_SALES_REPS", "MERIT_CYCLE_FLAG",
    "TRIAL_PHASE", "ACTIVE_SITES", "ENROLLED_PATIENTS", "MILESTONE_EVENT",
    "PRODUCT_LAUNCH_FLAG", "LOE_FLAG", "HCP_EVENTS_COUNT",
    "SURGICAL_PROCEDURE_INDEX", "CPI_INDEX", "WAGE_INDEX", "WAGE_INFLATION_RATE",
]


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    actual_years: Tuple[int, ...] = (2022, 2023, 2024, 2025)
    all_years: Tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)
    ar1_rho: float = 0.45
    po_vendor_poisson_lam: float = 6.5
    po_dirichlet_alpha: float = 0.68
    non_po_var_poisson_lam: float = 3.5
    non_po_var_dirichlet_alpha: float = 0.90
    weekend_day_weight: float = 0.08
    month_end_day_mult: float = 5.0
    accrual_prob: float = 0.16
    n_unreversed_accruals: int = 28
    n_duplicate_invoices: int = 36
    n_cross_month_duplicates: int = 9
    po_lead1_share: float = 0.65
    po_partial_gr_prob: float = 0.35
    po_partial_gr_frac: float = 0.25
    n_human_users: int = 39
    max_cutoff_date: date = date(2025, 12, 31)


def month_end_date(yr: int, m: int) -> date:
    return date(yr, m, calendar.monthrange(yr, m)[1])


def add_months(yr: int, m: int, delta: int) -> Tuple[int, int]:
    idx = (yr * 12 + (m - 1)) + delta
    return idx // 12, (idx % 12) + 1


def _day_weights(yr: int, m: int, cfg: GeneratorConfig) -> np.ndarray:
    """Realistic SAP posting-day weights: days 1..last_day, low weekends, 5x month-end surge."""
    last_day = calendar.monthrange(yr, m)[1]
    weights = np.ones(last_day, dtype=float)
    for d in range(1, last_day + 1):
        if date(yr, m, d).weekday() >= 5:
            weights[d - 1] *= cfg.weekend_day_weight
        if d > last_day - 3:
            weights[d - 1] *= cfg.month_end_day_mult
    return weights / weights.sum()


def _split_amount_dirichlet(total: float, n_docs: int, alpha: float, rng: np.random.Generator) -> List[float]:
    """Splits total across n_docs with Dirichlet right-skew while guaranteeing every doc >= $15."""
    if n_docs <= 1 or total < 40.0:
        return [round(total, 2)]
    raw_w = rng.dirichlet(np.full(n_docs, alpha))
    # Guarantee a small floor so no split line collapses to zero
    min_share = min(0.02, 15.0 / total)
    w = raw_w * (1.0 - n_docs * min_share) + min_share
    amts = [round(total * float(x), 2) for x in w]
    amts[-1] = round(total - sum(amts[:-1]), 2)
    return amts


# ---------------------------------------------------------------------------
# Stage 1: Master Dimensions (incl. dim_plant & 210000 Payroll Clearing) + FX
# ---------------------------------------------------------------------------
def build_dimensions(cfg: GeneratorConfig, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    dim_company = pd.DataFrame(
        [
            {"BUKRS": "US10", "BUTXT": "J&J Pharma US Inc.", "WAERS": "USD", "REGION": "US"},
            {"BUKRS": "EU20", "BUTXT": "J&J Pharma EMEA B.V.", "WAERS": "EUR", "REGION": "EMEA"},
            {"BUKRS": "AP30", "BUTXT": "J&J Pharma APAC Pvt Ltd", "WAERS": "INR", "REGION": "APAC"},
        ]
    )

    dim_plant = pd.DataFrame(
        [
            {"WERKS": "US01", "PLANT_NAME": "Titusville Commercial HQ", "BUKRS": "US10", "REGION": "US", "CITY": "Titusville, NJ", "PLANT_TYPE": "Commercial_HQ"},
            {"WERKS": "US02", "PLANT_NAME": "Raritan R&D Campus", "BUKRS": "US10", "REGION": "US", "CITY": "Raritan, NJ", "PLANT_TYPE": "RD_Campus"},
            {"WERKS": "US03", "PLANT_NAME": "Spring House R&D Campus", "BUKRS": "US10", "REGION": "US", "CITY": "Spring House, PA", "PLANT_TYPE": "RD_Campus"},
            {"WERKS": "BE01", "PLANT_NAME": "Beerse EMEA R&D Hub", "BUKRS": "EU20", "REGION": "EMEA", "CITY": "Beerse, Belgium", "PLANT_TYPE": "RD_Campus"},
            {"WERKS": "NL01", "PLANT_NAME": "Leiden EMEA Commercial Hub", "BUKRS": "EU20", "REGION": "EMEA", "CITY": "Leiden, Netherlands", "PLANT_TYPE": "Commercial_HQ"},
            {"WERKS": "IN01", "PLANT_NAME": "Mumbai/Hyderabad APAC Hub", "BUKRS": "AP30", "REGION": "APAC", "CITY": "Mumbai / Hyderabad, India", "PLANT_TYPE": "Regional_Hub"},
        ]
    )

    cc_specs = [
        ("US10", "US02", "CC1010", "US Oncology Clinical Ops", "Clinical_Development", "RND", "Innovative_Medicine", "Oncology", "US", "R&D", "PC_ONC_US", "WBS-ONC-P3-001", 1.35),
        ("US10", "US03", "CC1020", "US Oncology Discovery Labs", "Discovery_Research", "RND", "Innovative_Medicine", "Oncology", "US", "R&D", "PC_ONC_US", "WBS-ONC-P1-002", 0.95),
        ("US10", "US03", "CC1030", "US Immunology Clinical Ops", "Clinical_Development", "RND", "Innovative_Medicine", "Immunology", "US", "R&D", "PC_IMM_US", "WBS-IMM-P3-003", 1.25),
        ("US10", "US02", "CC1040", "US Neuroscience R&D", "Clinical_Development", "RND", "Innovative_Medicine", "Neuroscience", "US", "R&D", "PC_NEU_US", "WBS-NEU-P2-004", 0.85),
        ("US10", "US01", "CC1050", "US Oncology Commercial", "Brand_Commercial", "SALES", "Innovative_Medicine", "Oncology", "US", "Commercial_SGA", "PC_ONC_US", "NONE", 1.40),
        ("US10", "US01", "CC1060", "US Immunology Commercial", "Brand_Commercial", "SALES", "Innovative_Medicine", "Immunology", "US", "Commercial_SGA", "PC_IMM_US", "NONE", 1.30),
        ("US10", "US01", "CC1070", "US Medical Affairs IM", "Medical_Science", "RND", "Innovative_Medicine", "Oncology", "US", "Medical_Affairs", "PC_ONC_US", "NONE", 0.80),
        ("US10", "US01", "CC1080", "US MedTech Surgery Comm", "MedTech_Commercial", "SALES", "MedTech", "Surgery", "US", "Commercial_SGA", "PC_SUR_US", "NONE", 1.15),
        ("US10", "US02", "CC1090", "US MedTech Cardio R&D", "MedTech_RD", "RND", "MedTech", "Cardiovascular", "US", "R&D", "PC_CAR_US", "WBS-CAR-P2-005", 0.90),
        ("US10", "US01", "CC1100", "US Global IT & Digital", "Enterprise_IT", "ADMIN", "Enterprise_Corporate", "Corporate", "US", "G&A", "PC_CORP_GLB", "NONE", 1.20),
        ("US10", "US01", "CC1110", "US Corporate Finance & Legal", "Finance_Legal", "ADMIN", "Enterprise_Corporate", "Corporate", "US", "G&A", "PC_CORP_GLB", "NONE", 1.00),
        ("EU20", "BE01", "CC2010", "EMEA Oncology Clinical Ops", "Clinical_Development", "RND", "Innovative_Medicine", "Oncology", "EMEA", "R&D", "PC_ONC_EU", "WBS-ONC-P3-006", 1.05),
        ("EU20", "BE01", "CC2020", "EMEA Immunology Clinical", "Clinical_Development", "RND", "Innovative_Medicine", "Immunology", "EMEA", "R&D", "PC_IMM_EU", "WBS-IMM-P2-007", 0.85),
        ("EU20", "BE01", "CC2030", "EMEA Regulatory Affairs", "Regulatory_Affairs", "RND", "Innovative_Medicine", "Neuroscience", "EMEA", "R&D", "PC_NEU_EU", "WBS-NEU-FIL-008", 0.70),
        ("EU20", "NL01", "CC2040", "EMEA Oncology Commercial", "Brand_Commercial", "SALES", "Innovative_Medicine", "Oncology", "EMEA", "Commercial_SGA", "PC_ONC_EU", "NONE", 1.05),
        ("EU20", "NL01", "CC2050", "EMEA MedTech Vision Comm", "MedTech_Commercial", "SALES", "MedTech", "Vision", "EMEA", "Commercial_SGA", "PC_VIS_EU", "NONE", 0.80),
        ("EU20", "NL01", "CC2060", "EMEA MedTech Surgery Comm", "MedTech_Commercial", "SALES", "MedTech", "Surgery", "EMEA", "Commercial_SGA", "PC_SUR_EU", "NONE", 0.90),
        ("EU20", "BE01", "CC2070", "EMEA Medical Affairs", "Medical_Science", "RND", "Innovative_Medicine", "Immunology", "EMEA", "Medical_Affairs", "PC_IMM_EU", "NONE", 0.65),
        ("EU20", "NL01", "CC2080", "EMEA Corporate G&A", "Finance_Legal", "ADMIN", "Enterprise_Corporate", "Corporate", "EMEA", "G&A", "PC_CORP_EU", "NONE", 0.75),
        ("AP30", "IN01", "CC3010", "APAC Clinical Trial Hub", "Clinical_Development", "RND", "Innovative_Medicine", "Oncology", "APAC", "R&D", "PC_ONC_AP", "WBS-ONC-P3-009", 0.65),
        ("AP30", "IN01", "CC3020", "APAC Immunology Commercial", "Brand_Commercial", "SALES", "Innovative_Medicine", "Immunology", "APAC", "Commercial_SGA", "PC_IMM_AP", "NONE", 0.60),
        ("AP30", "IN01", "CC3030", "APAC MedTech Surgery Comm", "MedTech_Commercial", "SALES", "MedTech", "Surgery", "APAC", "Commercial_SGA", "PC_SUR_AP", "NONE", 0.55),
        ("AP30", "IN01", "CC3040", "APAC Medical Affairs", "Medical_Science", "RND", "Innovative_Medicine", "Neuroscience", "APAC", "Medical_Affairs", "PC_NEU_AP", "NONE", 0.45),
        ("AP30", "IN01", "CC3050", "APAC Global Services G&A", "Global_Shared_Services", "ADMIN", "Enterprise_Corporate", "Corporate", "APAC", "G&A", "PC_CORP_AP", "NONE", 0.70),
    ]
    dim_cc = pd.DataFrame(
        cc_specs,
        columns=[
            "BUKRS", "WERKS", "KOSTL", "KOSTL_NAME", "DEPARTMENT", "KOSTL_GROUP",
            "SEGMENT", "FRANCHISE_TA", "REGION", "RFAREA", "PRCTR", "DEFAULT_WBS", "COST_SCALE",
        ],
    )

    dim_pc = (
        dim_cc[["PRCTR", "SEGMENT", "FRANCHISE_TA", "REGION"]]
        .drop_duplicates()
        .assign(PRCTR_NAME=lambda df: df["SEGMENT"] + " - " + df["FRANCHISE_TA"] + " (" + df["REGION"] + ")")
        .reset_index(drop=True)
    )

    # 22 Life-Sciences OPEX GLs + 4 Balance Sheet Offset GLs (incl. 210000 Payroll Clearing)
    gl_specs = [
        ("610010", "Base Salaries & Wages", "OPEX", "STEP_FIXED", "WORKFORCE_CORE", "R&D|Commercial_SGA|Medical_Affairs|G&A", False),
        ("610020", "Employee Benefits & Payroll Tax", "OPEX", "STEP_FIXED", "WORKFORCE_CORE", "R&D|Commercial_SGA|Medical_Affairs|G&A", False),
        ("610030", "Annual Bonus & Equity Accrual", "OPEX", "VARIABLE", "WORKFORCE_CORE", "R&D|Commercial_SGA|Medical_Affairs|G&A", False),
        ("620010", "Clinical Trial CRO Services", "OPEX", "VARIABLE", "R&D_CLINICAL", "R&D", True),
        ("620020", "Investigator Site & Patient Fees", "OPEX", "VARIABLE", "R&D_CLINICAL", "R&D", True),
        ("620030", "Clinical Milestone Payments", "OPEX", "MILESTONE", "R&D_CLINICAL", "R&D", False),
        ("620040", "Lab Consumables & Reagents", "OPEX", "VARIABLE", "R&D_DISCOVERY", "R&D", True),
        ("620050", "Preclinical & Toxicology Studies", "OPEX", "VARIABLE", "R&D_DISCOVERY", "R&D", True),
        ("620060", "FDA/EMA Regulatory Filing Fees", "OPEX", "MILESTONE", "R&D_REGULATORY", "R&D", False),
        ("620070", "Lab Facility Rent & Utilities", "OPEX", "FIXED", "R&D_DISCOVERY", "R&D", False),
        ("630010", "Field Sales Incentives & Comm", "OPEX", "VARIABLE", "COMMERCIAL_FIELD", "Commercial_SGA", False),
        ("630020", "HCP Speaker Programs & Congress", "OPEX", "DISCRETIONARY", "COMMERCIAL_MARKETING", "Commercial_SGA", True),
        ("630030", "DTC & Digital Brand Marketing", "OPEX", "DISCRETIONARY", "COMMERCIAL_MARKETING", "Commercial_SGA", True),
        ("630040", "Market Access & HEOR Studies", "OPEX", "DISCRETIONARY", "COMMERCIAL_MARKETING", "Commercial_SGA", True),
        ("630050", "Field Travel, Fleet & Samples", "OPEX", "VARIABLE", "COMMERCIAL_FIELD", "Commercial_SGA", False),
        ("640010", "Medical Science Liaison Travel", "OPEX", "VARIABLE", "MEDICAL_AFFAIRS", "Medical_Affairs", False),
        ("640020", "Investigator Initiated Grants", "OPEX", "DISCRETIONARY", "MEDICAL_AFFAIRS", "Medical_Affairs", True),
        ("640030", "Scientific Publications & MedEd", "OPEX", "DISCRETIONARY", "MEDICAL_AFFAIRS", "Medical_Affairs", True),
        ("650010", "Enterprise Cloud & SaaS Licenses", "OPEX", "STEP_FIXED", "G&A_IT", "G&A", True),
        ("650020", "IT Consulting & ERP Support", "OPEX", "DISCRETIONARY", "G&A_IT", "G&A", True),
        ("650030", "Corporate Real Estate & Leases", "OPEX", "FIXED", "G&A_CORP", "G&A", False),
        ("650040", "External Audit, Legal & Advisory", "OPEX", "DISCRETIONARY", "G&A_CORP", "G&A", True),
        ("113100", "Cash & Bank Clearing", "BS", "NONE", "BALANCE_SHEET", "ALL", False),
        ("140000", "Prepaid Expenses", "BS", "NONE", "BALANCE_SHEET", "ALL", False),
        ("210000", "Payroll Clearing Liability", "BS", "NONE", "BALANCE_SHEET", "ALL", False),
        ("215000", "Accounts Payable & Accruals", "BS", "NONE", "BALANCE_SHEET", "ALL", False),
    ]
    dim_gl = pd.DataFrame(
        gl_specs,
        columns=["HKONT", "GL_NAME", "ACCOUNT_TYPE", "COST_BEHAVIOR", "OPEX_GROUP", "ALLOWED_RFAREAS", "USES_PO"],
    )
    dim_gl["RACCT"] = dim_gl["HKONT"]

    # Analytical FX Rate Table (BUDGET_PLAN(yr) locked strictly to Oct yr-1 ACTUAL_AVG)
    fx_rows = []
    eur_rate = 1.1200
    inr_rate = 0.01330
    oct_eur_prev = 1.1350
    oct_inr_prev = 0.01340

    for yr in cfg.all_years:
        eur_budget = round(oct_eur_prev, 6)
        inr_budget = round(oct_inr_prev, 6)
        for m in range(1, 13):
            if yr in cfg.actual_years:
                if yr == 2022:
                    eur_drift, inr_drift = -0.0045, -0.00008
                elif yr == 2023:
                    eur_drift, inr_drift = 0.0022, -0.00003
                elif yr == 2024:
                    eur_drift, inr_drift = -0.0012, -0.00002
                else:
                    eur_drift, inr_drift = 0.0010, -0.00002
                eur_rate = float(np.clip(eur_rate + eur_drift + rng.normal(0.0, 0.0075), 0.96, 1.22))
                inr_rate = float(np.clip(inr_rate + inr_drift + rng.normal(0.0, 0.00005), 0.0108, 0.0140))
                if m == 10:
                    oct_eur_prev = round(eur_rate, 6)
                    oct_inr_prev = round(inr_rate, 6)
                eur_fwd = round(eur_rate * 1.0015, 6)
                inr_fwd = round(inr_rate * 0.9990, 6)
            else:
                eur_rate = round(eur_rate * 1.0008, 6)
                inr_rate = round(inr_rate * 0.9992, 6)
                eur_fwd, inr_fwd = eur_rate, inr_rate

            for curr, act, bud, fwd in [
                ("USD", 1.0, 1.0, 1.0),
                ("EUR", round(eur_rate, 6), eur_budget, eur_fwd),
                ("INR", round(inr_rate, 6), inr_budget, inr_fwd),
            ]:
                if yr in cfg.actual_years:
                    fx_rows.append({"GJAHR": yr, "MONAT": m, "FROM_CURR": curr, "TO_CURR": "USD", "RATE_TYPE": "ACTUAL_AVG", "FX_RATE": act})
                fx_rows.append({"GJAHR": yr, "MONAT": m, "FROM_CURR": curr, "TO_CURR": "USD", "RATE_TYPE": "BUDGET_PLAN", "FX_RATE": bud})
                fx_rows.append({"GJAHR": yr, "MONAT": m, "FROM_CURR": curr, "TO_CURR": "USD", "RATE_TYPE": "FORWARD_CURVE", "FX_RATE": fwd})

    return {
        "dim_company_code": dim_company,
        "dim_plant": dim_plant,
        "dim_cost_center": dim_cc,
        "dim_profit_center": dim_pc,
        "dim_gl_account": dim_gl,
        "dim_fx_rates": pd.DataFrame(fx_rows),
    }


# ---------------------------------------------------------------------------
# Stage 2: Causal Operational Drivers (FY22-25) & Nov-Dec Bridged BP Drivers
# ---------------------------------------------------------------------------
def build_macro_indices(
    cfg: GeneratorConfig, rng: np.random.Generator
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    cpi_map, wage_idx_map, wage_rate_map = {}, {}, {}
    ann_cpi_tbl = {2022: 0.062, 2023: 0.039, 2024: 0.029, 2025: 0.025, 2026: 0.024}
    ann_wage_tbl = {2022: 0.048, 2023: 0.042, 2024: 0.036, 2025: 0.034, 2026: 0.033}

    cpi, wage_idx = 100.0, 100.0
    for yr in cfg.all_years:
        ann_cpi, ann_wage = ann_cpi_tbl[yr], ann_wage_tbl[yr]
        for m in range(1, 13):
            if m == 4:
                wage_idx = round(wage_idx * (1.0 + ann_wage), 4)
            noise = rng.normal(0.0, 0.0007) if yr in cfg.actual_years else 0.0
            cpi = round(cpi * (1.0 + ann_cpi / 12.0 + noise), 3)
            cpi_map[(yr, m)] = cpi
            wage_idx_map[(yr, m)] = wage_idx
            wage_rate_map[(yr, m)] = ann_wage
    return cpi_map, wage_idx_map, wage_rate_map


def build_opex_drivers(
    dim_cc: pd.DataFrame, cfg: GeneratorConfig, rng: np.random.Generator
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, dict],
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], float],
]:
    cpi_map, wage_idx_map, wage_rate_map = build_macro_indices(cfg, rng)

    planned_milestones = {
        ("CC1010", 2022, 10): "FPI",
        ("CC1010", 2024, 9): "DB_LOCK",
        ("CC1010", 2025, 9): "FDA_FILING",
        ("CC1010", 2026, 5): "DB_LOCK",
        ("CC1030", 2023, 6): "FPI",
        ("CC1030", 2025, 6): "DB_LOCK",
        ("CC1030", 2026, 7): "FDA_FILING",
        ("CC2010", 2023, 10): "DB_LOCK",
        ("CC2010", 2025, 7): "FDA_FILING",
        ("CC2010", 2026, 6): "FPI",
        ("CC2030", 2024, 4): "FDA_FILING",
        ("CC2030", 2025, 11): "FDA_FILING",
        ("CC3010", 2024, 6): "FPI",
        ("CC3010", 2026, 10): "DB_LOCK",
    }
    realized_milestones = {
        ("CC1010", 2022, 11): "FPI",
        ("CC1010", 2024, 9): "DB_LOCK",
        ("CC1010", 2025, 10): "FDA_FILING",
        ("CC1030", 2023, 6): "FPI",
        ("CC1030", 2025, 5): "DB_LOCK",
        ("CC2010", 2023, 10): "DB_LOCK",
        ("CC2010", 2025, 8): "FDA_FILING",
        ("CC2030", 2024, 4): "FDA_FILING",
        ("CC2030", 2025, 11): "FDA_FILING",
        ("CC3010", 2024, 8): "FPI",
    }

    driver_rows, plan_driver_rows = [], []
    dec2025_state: Dict[str, dict] = {}

    for _, cc in dim_cc.iterrows():
        bukrs, werks, kostl = cc["BUKRS"], cc["WERKS"], cc["KOSTL"]
        rfarea, wbs, scale = cc["RFAREA"], cc["DEFAULT_WBS"], float(cc["COST_SCALE"])
        segment, ta = cc["SEGMENT"], cc["FRANCHISE_TA"]

        base_fte = 52.0 * scale if rfarea in ("R&D", "Commercial_SGA") else 36.0 * scale
        curr_fte = oct_fte_anchor = base_fte

        is_clinical = wbs.startswith("WBS-")
        if "-P3-" in wbs:
            trial_phase_init, sites_base, pat_base = "P3", 52.0 * scale, 290.0 * scale
        elif "-P2-" in wbs:
            trial_phase_init, sites_base, pat_base = "P2", 22.0 * scale, 105.0 * scale
        elif "-P1-" in wbs:
            trial_phase_init, sites_base, pat_base = "P1", 8.0 * scale, 32.0 * scale
        elif "-FIL-" in wbs:
            trial_phase_init, sites_base, pat_base = "FILING", 5.0 * scale, 16.0 * scale
        else:
            trial_phase_init, sites_base, pat_base = "NONE", 0.0, 0.0

        curr_sites = oct_sites_anchor = sites_base
        curr_patients = oct_patients_anchor = pat_base

        for yr in cfg.all_years:
            if yr > 2022:
                yr_plan_fte = oct_fte_anchor + 2.0 * 0.42 * scale
                yr_plan_sites = (oct_sites_anchor * (1.008 ** 2)) if is_clinical else 0.0
                yr_plan_patients = (oct_patients_anchor * (1.013 ** 2)) if is_clinical else 0.0
            else:
                yr_plan_fte = base_fte
                yr_plan_sites = sites_base if is_clinical else 0.0
                yr_plan_patients = pat_base if is_clinical else 0.0

            yr_plan_hcp_base = (18.0 * scale) if rfarea == "Commercial_SGA" else 0.0
            bp_wage_rate = 0.036
            bp_cpi_base = cpi_map.get((yr - 1, 10), 99.0)
            bp_wage_idx_base = wage_idx_map.get((yr - 1, 10), 100.0)

            for m in range(1, 13):
                merit_flag = 1 if m == 4 else 0
                congress_bump = 1.45 if m in (5, 6, 9, 10, 11) else 0.85

                yr_plan_fte = max(12.0, yr_plan_fte + 0.42 * scale)
                if is_clinical:
                    yr_plan_sites = max(3.0, yr_plan_sites * 1.008)
                    yr_plan_patients = max(10.0, yr_plan_patients * 1.013)
                else:
                    yr_plan_sites, yr_plan_patients = 0.0, 0.0

                plan_hcp = max(1, int(round(yr_plan_hcp_base * congress_bump))) if rfarea == "Commercial_SGA" else 0
                plan_ms = planned_milestones.get((kostl, yr, m), "NONE")
                bp_wage_idx = round(bp_wage_idx_base * (1.0 + bp_wage_rate) if m >= 4 else bp_wage_idx_base, 4)
                bp_cpi = round(bp_cpi_base * (1.030 ** ((m + 2) / 12.0)), 3)

                plan_driver_rows.append(
                    {
                        "GJAHR": yr, "MONAT": m, "BUKRS": bukrs, "WERKS": werks,
                        "KOSTL": kostl, "WBS_ELEMENT": wbs,
                        "PLAN_FTE": round(yr_plan_fte, 1),
                        "PLAN_ACTIVE_SITES": round(yr_plan_sites, 1),
                        "PLAN_ENROLLED_PATIENTS": round(yr_plan_patients, 1),
                        "PLAN_HCP_EVENTS": plan_hcp,
                        "PLAN_MILESTONE_EVENT": plan_ms,
                        "PLAN_CPI_INDEX": bp_cpi,
                        "PLAN_WAGE_INDEX": bp_wage_idx,
                    }
                )

                if yr not in cfg.actual_years:
                    continue

                hire_lam = 1.45 * scale if m in (1, 2, 4, 7, 9, 10) else 0.55 * scale
                planned_hires = max(0, int(rng.poisson(hire_lam)))
                attrition = curr_fte * rng.uniform(0.005, 0.011)
                net_hire = planned_hires * 0.85 - attrition
                if yr == 2024 and m in (7, 8, 9):
                    net_hire -= 0.35 * scale
                curr_fte = max(12.0, curr_fte + net_hire)
                field_reps = round(curr_fte * 0.68, 1) if rfarea == "Commercial_SGA" else 0.0

                trial_phase = trial_phase_init
                if is_clinical:
                    if trial_phase_init == "P2" and yr >= 2024:
                        trial_phase = "P3"
                    elif trial_phase_init == "P1" and yr >= 2025:
                        trial_phase = "P2"
                    ramp_p = 1.0 + 0.014 * math.sin(2 * math.pi * m / 12.0) + (0.017 if trial_phase == "P3" else 0.011)
                    curr_sites = max(3.0, curr_sites * 1.0085 + rng.normal(0.0, 0.55))
                    curr_patients = max(10.0, curr_patients * ramp_p + rng.normal(0.0, 3.2 * scale))
                else:
                    curr_sites, curr_patients = 0.0, 0.0

                milestone_event = realized_milestones.get((kostl, yr, m), "NONE")

                launch_flag = 0
                loe_flag = 0
                if rfarea == "Commercial_SGA":
                    if ta == "Oncology" and yr == 2023 and m in (4, 5, 6, 7, 8, 9, 10):
                        launch_flag = 1
                    if ta == "Immunology" and yr == 2025 and m in (2, 3, 4, 5, 6, 7):
                        launch_flag = 1
                    if ta == "Immunology" and yr == 2024 and m >= 7:
                        loe_flag = 1
                    hcp_events = max(
                        1,
                        int(round(18.0 * scale * congress_bump * (1.35 if launch_flag else 1.0) * (0.86 if loe_flag else 1.0) + rng.normal(0, 1.8))),
                    )
                else:
                    hcp_events = 0

                if segment == "MedTech":
                    surg_season = {1: 96, 2: 98, 3: 103, 4: 102, 5: 104, 6: 103, 7: 97, 8: 93, 9: 102, 10: 106, 11: 108, 12: 110}[m]
                    surg_idx = round(surg_season * (1.0 + 0.025 * (yr - 2022)) + rng.normal(0.0, 1.3), 2)
                else:
                    surg_idx = 100.0

                rec = {
                    "GJAHR": yr, "MONAT": m, "BUKRS": bukrs, "WERKS": werks,
                    "KOSTL": kostl, "WBS_ELEMENT": wbs,
                    "ACTIVE_FTE": round(curr_fte, 1), "PLANNED_HIRES": planned_hires,
                    "FIELD_SALES_REPS": field_reps, "MERIT_CYCLE_FLAG": merit_flag,
                    "TRIAL_PHASE": trial_phase, "ACTIVE_SITES": round(curr_sites, 1),
                    "ENROLLED_PATIENTS": round(curr_patients, 1), "MILESTONE_EVENT": milestone_event,
                    "PRODUCT_LAUNCH_FLAG": launch_flag, "LOE_FLAG": loe_flag,
                    "HCP_EVENTS_COUNT": hcp_events, "SURGICAL_PROCEDURE_INDEX": surg_idx,
                    "CPI_INDEX": cpi_map[(yr, m)], "WAGE_INDEX": wage_idx_map[(yr, m)],
                    "WAGE_INFLATION_RATE": wage_rate_map[(yr, m)],
                }
                driver_rows.append(rec)

                if m == 10:
                    oct_fte_anchor, oct_sites_anchor, oct_patients_anchor = curr_fte, curr_sites, curr_patients

                if yr == 2025 and m == 12:
                    dec2025_state[kostl] = {
                        **rec, "COST_SCALE": scale, "SEGMENT": segment,
                        "FRANCHISE_TA": ta, "RFAREA": rfarea, "REGION": cc["REGION"],
                    }

    return (
        pd.DataFrame(driver_rows)[DRIVER_FEATURE_COLS],
        pd.DataFrame(plan_driver_rows),
        dec2025_state,
        cpi_map,
        wage_idx_map,
        wage_rate_map,
    )


# ---------------------------------------------------------------------------
# Stage 3: Deterministic FY2026 What-If Scenarios (Strictly from Dec 2025 State)
# ---------------------------------------------------------------------------
def build_scenario_assumptions(
    dec2025_state: Dict[str, dict],
    dim_fx: pd.DataFrame,
    cpi_map: Dict[Tuple[int, int], float],
    wage_idx_map: Dict[Tuple[int, int], float],
    wage_rate_map: Dict[Tuple[int, int], float],
) -> pd.DataFrame:
    fwd_fx = (
        dim_fx[(dim_fx["GJAHR"] == 2026) & (dim_fx["RATE_TYPE"] == "FORWARD_CURVE")]
        .pivot(index=["GJAHR", "MONAT"], columns="FROM_CURR", values="FX_RATE")
        .reset_index()
        .rename(columns={"EUR": "FX_EUR_USD", "INR": "FX_INR_USD"})
    )
    fx26 = {int(r.MONAT): (float(r.FX_EUR_USD), float(r.FX_INR_USD)) for r in fwd_fx.itertuples(index=False)}
    base_milestones_26 = {
        ("CC1010", 5): "DB_LOCK",
        ("CC1030", 7): "FDA_FILING",
        ("CC2010", 6): "FPI",
        ("CC3010", 10): "DB_LOCK",
    }

    base_rows = []
    for kostl, st in sorted(dec2025_state.items()):
        scale, rfarea, segment = float(st["COST_SCALE"]), st["RFAREA"], st["SEGMENT"]
        trial_phase, is_clinical = st["TRIAL_PHASE"], st["WBS_ELEMENT"].startswith("WBS-")
        fte, sites, patients = float(st["ACTIVE_FTE"]), float(st["ACTIVE_SITES"]), float(st["ENROLLED_PATIENTS"])

        for m in range(1, 13):
            merit_flag = 1 if m == 4 else 0
            planned_hires = max(1, int(round(1.4 * scale))) if m in (1, 2, 4, 7, 9, 10) else max(0, int(round(0.5 * scale)))
            fte = max(12.0, fte + planned_hires * 0.85 - fte * 0.009)
            reps = round(fte * 0.68, 1) if rfarea == "Commercial_SGA" else 0.0

            if is_clinical:
                ramp_p = 1.0 + 0.012 * math.sin(2 * math.pi * m / 12.0) + (0.015 if trial_phase == "P3" else 0.010)
                sites = max(3.0, sites * 1.008)
                patients = max(10.0, patients * ramp_p)
            else:
                sites, patients = 0.0, 0.0

            congress_bump = 1.45 if m in (5, 6, 9, 10, 11) else 0.85
            hcp = max(1, int(round(18.0 * scale * congress_bump))) if rfarea == "Commercial_SGA" else 0
            if segment == "MedTech":
                surg_season = {1: 96, 2: 98, 3: 103, 4: 102, 5: 104, 6: 103, 7: 97, 8: 93, 9: 102, 10: 106, 11: 108, 12: 110}[m]
                surg_idx = round(surg_season * (1.0 + 0.025 * 4), 2)
            else:
                surg_idx = 100.0

            fx_eur, fx_inr = fx26[m]
            base_rows.append(
                {
                    "SCENARIO_ID": "BASE", "SNAPSHOT_DATE": "2025-12-31",
                    "GJAHR": 2026, "MONAT": m, "BUKRS": st["BUKRS"], "WERKS": st["WERKS"],
                    "KOSTL": kostl, "WBS_ELEMENT": st["WBS_ELEMENT"],
                    "ACTIVE_FTE": round(fte, 1), "PLANNED_HIRES": planned_hires,
                    "FIELD_SALES_REPS": reps, "MERIT_CYCLE_FLAG": merit_flag,
                    "TRIAL_PHASE": trial_phase, "ACTIVE_SITES": round(sites, 1),
                    "ENROLLED_PATIENTS": round(patients, 1),
                    "MILESTONE_EVENT": base_milestones_26.get((kostl, m), "NONE"),
                    "PRODUCT_LAUNCH_FLAG": 0, "LOE_FLAG": 0, "HCP_EVENTS_COUNT": hcp,
                    "SURGICAL_PROCEDURE_INDEX": surg_idx,
                    "CPI_INDEX": cpi_map[(2026, m)], "WAGE_INDEX": wage_idx_map[(2026, m)],
                    "WAGE_INFLATION_RATE": wage_rate_map[(2026, m)],
                    "FX_EUR_USD": fx_eur, "FX_INR_USD": fx_inr,
                    "_RFAREA": rfarea, "_FRANCHISE_TA": st["FRANCHISE_TA"],
                }
            )

    base_df = pd.DataFrame(base_rows)
    all_scens = [base_df.copy()]

    # HIRING_FREEZE_Q2: zero hires M4-M12 + smooth 0.9%/mo natural attrition
    hf = base_df.copy()
    hf["SCENARIO_ID"] = "HIRING_FREEZE_Q2"
    for _, grp in hf.groupby("KOSTL"):
        idxs = grp.sort_values("MONAT").index.tolist()
        rfarea = grp["_RFAREA"].iloc[0]
        curr_fte = float(hf.loc[idxs[2], "ACTIVE_FTE"])
        for pos in range(3, 12):
            ix = idxs[pos]
            curr_fte = max(12.0, curr_fte * (1.0 - 0.009))
            hf.loc[ix, "PLANNED_HIRES"] = 0
            hf.loc[ix, "ACTIVE_FTE"] = round(curr_fte, 1)
            hf.loc[ix, "FIELD_SALES_REPS"] = round(curr_fte * 0.68, 1) if rfarea == "Commercial_SGA" else 0.0
    all_scens.append(hf)

    # PHASE3_DELAY_1Q: smooth Q2 enrollment pause near M3 + shifted ramp M7-M12 + shifted milestones
    p3 = base_df.copy()
    p3["SCENARIO_ID"] = "PHASE3_DELAY_1Q"
    for _, grp in p3[p3["TRIAL_PHASE"] == "P3"].groupby("KOSTL"):
        grp_s = grp.sort_values("MONAT")
        idxs = grp_s.index.tolist()
        base_pat, base_sit, base_ms = grp_s["ENROLLED_PATIENTS"].tolist(), grp_s["ACTIVE_SITES"].tolist(), grp_s["MILESTONE_EVENT"].tolist()
        new_ms = base_ms[:3] + ["NONE"] * 9
        for m_zero in range(3, 12):
            if base_ms[m_zero] != "NONE" and m_zero + 3 < 12:
                new_ms[m_zero + 3] = base_ms[m_zero]
        for pos in range(3, 12):
            ix = idxs[pos]
            if pos < 6:
                p3.loc[ix, "ENROLLED_PATIENTS"] = round(base_pat[2] * (1.002 ** (pos - 2)), 1)
                p3.loc[ix, "ACTIVE_SITES"] = round(base_sit[2] * (1.001 ** (pos - 2)), 1)
            else:
                p3.loc[ix, "ENROLLED_PATIENTS"] = round(base_pat[pos - 3], 1)
                p3.loc[ix, "ACTIVE_SITES"] = round(base_sit[pos - 3], 1)
            p3.loc[ix, "MILESTONE_EVENT"] = new_ms[pos]
    all_scens.append(p3)

    # USD_STRENGTHEN_10PCT
    fx_sc = base_df.copy()
    fx_sc["SCENARIO_ID"] = "USD_STRENGTHEN_10PCT"
    fx_sc["FX_EUR_USD"] = np.round(fx_sc["FX_EUR_USD"] * 0.90, 6)
    fx_sc["FX_INR_USD"] = np.round(fx_sc["FX_INR_USD"] * 0.90, 6)
    all_scens.append(fx_sc)

    # LAUNCH_ACCEL
    la = base_df.copy()
    la["SCENARIO_ID"] = "LAUNCH_ACCEL"
    mask = (la["_FRANCHISE_TA"] == "Immunology") & (la["_RFAREA"] == "Commercial_SGA") & (la["MONAT"].between(3, 9))
    la.loc[mask, "PRODUCT_LAUNCH_FLAG"] = 1
    la.loc[mask, "HCP_EVENTS_COUNT"] = np.round(la.loc[mask, "HCP_EVENTS_COUNT"] * 1.30).astype(int)
    all_scens.append(la)

    final_cols = ["SCENARIO_ID", "SNAPSHOT_DATE"] + DRIVER_FEATURE_COLS + ["FX_EUR_USD", "FX_INR_USD"]
    return pd.concat(all_scens, ignore_index=True)[final_cols]


# ---------------------------------------------------------------------------
# Structural Cost Equation (USD Run-Rate from Drivers)
# ---------------------------------------------------------------------------
def compute_cell_runrate_usd(
    hkont: str, drv: Dict, prev_drv: Dict, cost_scale: float, region: str
) -> Tuple[float, float]:
    labor_reg = {"US": 1.00, "EMEA": 0.84, "APAC": 0.46}[region]
    vendor_reg = {"US": 1.00, "EMEA": 0.90, "APAC": 0.68}[region]

    fte = float(drv["ACTIVE_FTE"])
    reps = float(drv["FIELD_SALES_REPS"])
    cpi_mult = float(drv["CPI_INDEX"]) / 100.0
    wage_mult = float(drv["WAGE_INDEX"]) / 100.0
    sites_lag = float(prev_drv["ACTIVE_SITES"])
    pat_lag = float(prev_drv["ENROLLED_PATIENTS"])
    hcp = float(drv["HCP_EVENTS_COUNT"])
    launch = int(drv["PRODUCT_LAUNCH_FLAG"])
    loe = int(drv["LOE_FLAG"])
    surg_mult = float(drv["SURGICAL_PROCEDURE_INDEX"]) / 100.0
    milestone = str(drv["MILESTONE_EVENT"])
    m, yr = int(drv["MONAT"]), int(drv["GJAHR"])
    q4_bump = 1.12 if m in (11, 12) else 1.00
    tiers_25 = math.ceil(max(1.0, fte) / 25.0)

    if hkont == "610010":
        usd, sigma = (fte * 10_900.0 + tiers_25 * 18_500.0) * labor_reg * wage_mult, 0.018
    elif hkont == "610020":
        usd, sigma = (fte * 2_950.0 + tiers_25 * 4_500.0) * labor_reg * wage_mult * (1.06 if m in (1, 2) else 1.0), 0.025
    elif hkont == "610030":
        usd, sigma = fte * 2_250.0 * labor_reg * wage_mult * (1.18 if m == 12 else 1.0), 0.048
    elif hkont == "620010":
        usd, sigma = (sites_lag * 16_000.0 + pat_lag * 1_850.0) * vendor_reg * cpi_mult, 0.052
    elif hkont == "620020":
        usd, sigma = (sites_lag * 5_600.0 + pat_lag * 700.0) * vendor_reg * cpi_mult, 0.058
    elif hkont == "620030":
        ms_val = {"NONE": 0.0, "FPI": 650_000.0, "DB_LOCK": 950_000.0, "FDA_FILING": 1_450_000.0}[milestone]
        usd, sigma = ms_val * vendor_reg, 0.025
    elif hkont == "620040":
        usd, sigma = (fte * 1_450.0 + sites_lag * 1_100.0) * vendor_reg * cpi_mult, 0.050
    elif hkont == "620050":
        usd, sigma = (65_000.0 * cost_scale + fte * 650.0) * vendor_reg * cpi_mult * (1.15 if m in (3, 6, 9, 12) else 1.0), 0.068
    elif hkont == "620060":
        usd, sigma = (850_000.0 if milestone == "FDA_FILING" else 18_000.0 * cost_scale) * cpi_mult, 0.035
    elif hkont == "620070":
        usd, sigma = 145_000.0 * cost_scale * vendor_reg * (1.0 + 0.03 * (yr - 2022)) * (1.06 if m in (1, 7, 8) else 1.0), 0.012
    elif hkont == "630010":
        usd, sigma = reps * 4_200.0 * labor_reg * surg_mult * (1.20 if launch else 1.0) * (1.15 if m in (3, 6, 9, 12) else 1.0), 0.052
    elif hkont == "630020":
        usd, sigma = (hcp * 14_500.0) * vendor_reg * cpi_mult * q4_bump, 0.062
    elif hkont == "630030":
        usd, sigma = (180_000.0 * cost_scale + hcp * 6_200.0) * vendor_reg * cpi_mult * (1.55 if launch else 1.0) * (0.82 if loe else 1.0) * q4_bump, 0.068
    elif hkont == "630040":
        usd, sigma = (75_000.0 * cost_scale + hcp * 1_800.0) * vendor_reg * cpi_mult * (1.25 if launch else 1.0), 0.072
    elif hkont == "630050":
        usd, sigma = (reps * 1_850.0 + hcp * 1_100.0) * vendor_reg * cpi_mult * surg_mult, 0.048
    elif hkont == "640010":
        usd, sigma = fte * 1_900.0 * vendor_reg * cpi_mult * (1.20 if m in (5, 6, 9, 10) else 1.0), 0.052
    elif hkont == "640020":
        usd, sigma = (95_000.0 * cost_scale + fte * 1_100.0) * vendor_reg * cpi_mult * q4_bump, 0.072
    elif hkont == "640030":
        usd, sigma = (62_000.0 * cost_scale + fte * 750.0) * vendor_reg * cpi_mult * (1.18 if m in (6, 9, 11) else 1.0), 0.065
    elif hkont == "650010":
        usd, sigma = (135_000.0 * cost_scale + tiers_25 * 58_000.0) * cpi_mult * (1.06 if yr >= 2024 else 1.0), 0.020
    elif hkont == "650020":
        usd, sigma = (165_000.0 * cost_scale + fte * 1_200.0) * vendor_reg * cpi_mult * q4_bump, 0.070
    elif hkont == "650030":
        usd, sigma = 195_000.0 * cost_scale * vendor_reg * (1.0 + 0.025 * (yr - 2022)), 0.010
    elif hkont == "650040":
        usd, sigma = (130_000.0 * cost_scale + fte * 900.0) * vendor_reg * cpi_mult * (1.28 if m in (2, 3, 11, 12) else 1.0), 0.075
    else:
        usd, sigma = 0.0, 0.0

    return float(round(usd, 2)), sigma


# ---------------------------------------------------------------------------
# Stage 4: Point-in-Time PO Lifecycle + Chronological Double-Entry ACDOCA
# ---------------------------------------------------------------------------
def build_po_and_actuals(
    dims: Dict[str, pd.DataFrame],
    fact_drivers: pd.DataFrame,
    scenario_base_26: pd.DataFrame,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dim_cc = dims["dim_cost_center"]
    dim_gl = dims["dim_gl_account"]
    dim_comp = dims["dim_company_code"].set_index("BUKRS")
    dim_fx = dims["dim_fx_rates"]

    fx_lookup = {
        (int(r.GJAHR), int(r.MONAT), str(r.FROM_CURR), str(r.RATE_TYPE)): float(r.FX_RATE)
        for r in dim_fx.itertuples(index=False)
    }

    drv_lookup: Dict[Tuple[int, int, str], dict] = {}
    for row in fact_drivers.to_dict(orient="records"):
        drv_lookup[(int(row["GJAHR"]), int(row["MONAT"]), str(row["KOSTL"]))] = row
    for row in scenario_base_26[scenario_base_26["MONAT"].isin([1, 2])].to_dict(orient="records"):
        drv_lookup[(2026, int(row["MONAT"]), str(row["KOSTL"]))] = row

    opex_gls = dim_gl[dim_gl["ACCOUNT_TYPE"] == "OPEX"].copy()
    rfarea_to_gls: Dict[str, List[dict]] = {
        rfa: [g for g in opex_gls.to_dict(orient="records") if rfa in g["ALLOWED_RFAREAS"].split("|")]
        for rfa in ["R&D", "Commercial_SGA", "Medical_Affairs", "G&A"]
    }

    vendor_pools = {
        "R&D_CLINICAL": ["VND_CRO_IQVIA", "VND_CRO_PPD", "VND_CRO_SYNEOS", "VND_LAB_COVANCE"],
        "R&D_DISCOVERY": ["VND_SCI_THERMO", "VND_SCI_AGILENT", "VND_SCI_MERCK"],
        "R&D_REGULATORY": ["VND_REG_FDA", "VND_REG_EMA", "VND_REG_PAREXEL"],
        "COMMERCIAL_FIELD": ["VND_FLT_ELEMENT", "VND_TRV_AMEX"],
        "COMMERCIAL_MARKETING": ["VND_MKT_OMNICOM", "VND_MKT_PUBLICIS", "VND_EVT_CONGRESS"],
        "MEDICAL_AFFAIRS": ["VND_MED_WILEY", "VND_MED_ELSEVIER", "VND_MED_GRANTS"],
        "G&A_IT": ["VND_IT_SAP", "VND_IT_AWS", "VND_IT_ACCENTURE", "VND_IT_MSFT"],
        "G&A_CORP": ["VND_LEG_PWC", "VND_LEG_DELOITTE", "VND_RE_CBRE"],
        "WORKFORCE_CORE": ["INTERNAL_PAYROLL"],
    }
    human_users = [f"USER{i:03d}" for i in range(1, cfg.n_human_users + 1)]
    day_weight_cache = {(yr, m): _day_weights(yr, m, cfg) for yr in cfg.actual_years for m in range(1, 13)}

    po_snapshot_rows: List[dict] = []
    cell_records: List[dict] = []
    po_counter = 4500001000

    for _, cc in dim_cc.iterrows():
        bukrs, werks, kostl = cc["BUKRS"], cc["WERKS"], cc["KOSTL"]
        rfarea, region, scale, wbs = cc["RFAREA"], cc["REGION"], float(cc["COST_SCALE"]), cc["DEFAULT_WBS"]

        for g in rfarea_to_gls[rfarea]:
            hkont, uses_po, opex_grp, behavior = g["HKONT"], bool(g["USES_PO"]), g["OPEX_GROUP"], g["COST_BEHAVIOR"]
            vendors = vendor_pools[opex_grp]
            rho = 0.0 if behavior == "MILESTONE" else cfg.ar1_rho
            shock_prev = 0.0
            ym_horizon = [(y, m) for y in cfg.actual_years for m in range(1, 13)] + [(2026, 1), (2026, 2)]

            for yr, m in ym_horizon:
                drv = drv_lookup[(yr, m, kostl)]
                prev_yr, prev_m = add_months(yr, m, -1)
                prev_drv = drv_lookup.get((prev_yr, prev_m, kostl), drv)

                det_usd, sigma = compute_cell_runrate_usd(hkont, drv, prev_drv, scale, region)
                if det_usd <= 0.0:
                    continue

                if yr in cfg.actual_years:
                    # Mean-preserving multiplicative lognormal AR(1) shock: E[exp(shock - stat_var/2)] == 1
                    stat_var = (sigma ** 2) / (1.0 - rho ** 2) if rho > 0.0 else (sigma ** 2)
                    shock_curr = float(np.clip(rho * shock_prev + float(rng.normal(0.0, sigma)), -0.35, 0.35))
                    shock_prev = shock_curr
                    base_usd = max(500.0, round(det_usd * math.exp(shock_curr - 0.5 * stat_var), 2))
                else:
                    base_usd = det_usd

                final_usd = base_usd
                ebeln_1, ebeln_2 = "", ""
                lifnr_used = str(rng.choice(vendors))

                if uses_po:
                    po_total = round(base_usd * (float(rng.uniform(0.90, 1.06)) if yr in cfg.actual_years else 0.96), 2)
                    gr_month_str = f"{yr:04d}-{m:02d}"
                    amt_1a = round(po_total * cfg.po_lead1_share * 0.70, 2)
                    amt_1b = round(po_total * cfg.po_lead1_share * 0.30, 2)
                    amt_2a = round(po_total - amt_1a - amt_1b, 2)
                    po_defs = [(1, [("00010", amt_1a), ("00020", amt_1b)]), (2, [("00010", amt_2a)])]

                    for lead_m, lines in po_defs:
                        po_yr, po_m = add_months(yr, m, -lead_m)
                        if po_yr < 2021:
                            continue
                        po_day = int(rng.integers(5, 25))
                        po_create = date(po_yr, po_m, min(po_day, calendar.monthrange(po_yr, po_m)[1]))
                        if po_create > cfg.max_cutoff_date:
                            continue

                        po_counter += 1
                        ebeln = f"PO{po_counter}"
                        if lead_m == 1:
                            ebeln_1 = ebeln
                        else:
                            ebeln_2 = ebeln

                        has_partial = (lead_m == 2) and (rng.random() < cfg.po_partial_gr_prob)
                        for ebelp, orig_amt in lines:
                            if orig_amt <= 0.0:
                                continue
                            for step in range(lead_m + 1):
                                s_yr, s_m = add_months(po_yr, po_m, step)
                                snap_dt = month_end_date(s_yr, s_m)
                                if snap_dt > cfg.max_cutoff_date:
                                    break

                                if (s_yr, s_m) == (yr, m):
                                    gr_amt, inv_amt, open_amt, status = orig_amt, orig_amt, 0.0, "CLOSED"
                                elif has_partial and step == 1:
                                    gr_amt = round(orig_amt * cfg.po_partial_gr_frac, 2)
                                    inv_amt = round(gr_amt * 0.80, 2)
                                    open_amt = round(orig_amt - gr_amt, 2)
                                    status = "PARTIALLY_RECEIVED"
                                else:
                                    gr_amt, inv_amt, open_amt, status = 0.0, 0.0, orig_amt, "OPEN"

                                po_snapshot_rows.append(
                                    {
                                        "EBELN": ebeln, "EBELP": ebelp, "BUKRS": bukrs, "WERKS": werks,
                                        "KOSTL": kostl, "HKONT": hkont, "WBS_ELEMENT": wbs, "LIFNR": lifnr_used,
                                        "PO_CREATE_DATE": po_create.isoformat(),
                                        "EXPECTED_GR_MONTH": gr_month_str,
                                        "SNAPSHOT_DATE": snap_dt.isoformat(),
                                        "ORIGINAL_COMMITMENT_USD": orig_amt,
                                        "GR_AMOUNT_USD": gr_amt,
                                        "INVOICED_AMOUNT_USD": inv_amt,
                                        "OPEN_COMMITMENT_USD": open_amt,
                                        "PO_STATUS": status,
                                    }
                                )

                    if yr in cfg.actual_years:
                        final_usd = round(0.80 * po_total * float(rng.uniform(0.95, 1.05)) + 0.20 * base_usd, 2)

                if yr in cfg.actual_years:
                    cell_records.append(
                        {
                            "GJAHR": yr, "MONAT": m, "BUKRS": bukrs, "WERKS": werks,
                            "KOSTL": kostl, "PRCTR": cc["PRCTR"], "RFAREA": rfarea,
                            "WBS_ELEMENT": wbs, "HKONT": hkont, "COST_BEHAVIOR": behavior,
                            "OPEX_GROUP": opex_grp, "USES_PO": uses_po, "LIFNR": lifnr_used,
                            "EBELN_1": ebeln_1, "EBELN_2": ebeln_2, "RUNRATE_USD": final_usd,
                        }
                    )

    fact_po = pd.DataFrame(po_snapshot_rows)
    cell_df = pd.DataFrame(cell_records)

    # Disaggregate Monthly Cells into Balanced Double-Entry Postings (Temp BELNRs, remapped chronologically below)
    actual_rows: List[dict] = []
    anomaly_labels: List[dict] = []
    tmp_doc_counters: Dict[Tuple[str, int], int] = {(b, y): 0 for b in dim_comp.index for y in cfg.actual_years}

    def next_tmp_belnr(bukrs: str, yr: int) -> str:
        tmp_doc_counters[(bukrs, yr)] += 1
        return f"T{tmp_doc_counters[(bukrs, yr)]:09d}"

    def add_balanced_doc(
        bukrs: str, werks: str, yr: int, monat: int, bldat: date, budat: date, cpudt: date,
        blart: str, usnam: str, hkont: str, offset_hkont: str, kostl: str, prctr: str,
        rfarea: str, wbs: str, lifnr: str, ebeln: str,
        usd_amount: float = 0.0, dmbtr_override: float | None = None,
        is_reversal: bool = False, stblg: str = "", stjah: int = 0,
        special_flag: str = "NONE", svc_start: date | None = None, svc_end: date | None = None,
        belnr_override: str | None = None,
    ) -> Tuple[str, float, float]:
        waers = str(dim_comp.loc[bukrs, "WAERS"])
        fx_m = min(monat, 12)
        act_fx = fx_lookup[(yr, fx_m, waers, "ACTUAL_AVG")]
        bud_fx = fx_lookup[(yr, fx_m, waers, "BUDGET_PLAN")]

        dmbtr = round(abs(dmbtr_override), 2) if dmbtr_override is not None else round(abs(usd_amount) / act_fx, 2)
        ksl = round(dmbtr * act_fx, 2)
        ksl_bud = round(dmbtr * bud_fx, 2)

        belnr = belnr_override or next_tmp_belnr(bukrs, yr)
        s_start = (svc_start or bldat).isoformat()
        s_end = (svc_end or budat).isoformat()

        opex_shkzg, off_shkzg = ("H", "S") if is_reversal else ("S", "H")
        opex_bschl, off_bschl = ("50", "40") if is_reversal else ("40", "50")

        base_common = {
            "BUKRS": bukrs, "WERKS": werks, "GJAHR": yr, "MONAT": monat, "BELNR": belnr,
            "BLDAT": bldat.isoformat(), "BUDAT": budat.isoformat(), "CPUDT": cpudt.isoformat(),
            "BLART": blart, "USNAM": usnam, "PRCTR": prctr, "WBS_ELEMENT": wbs,
            "LIFNR": lifnr, "EBELN": ebeln, "WAERS": waers,
            "DMBTR": dmbtr, "KSL": ksl, "KSL_BUDGET_FX": ksl_bud,
            "STBLG": stblg, "STJAH": stjah if stblg else 0,
            "SPECIAL_ITEM_FLAG": special_flag,
            "SERVICE_START_DATE": s_start, "SERVICE_END_DATE": s_end, "VERSION": 0,
        }
        actual_rows.append({**base_common, "BUZEI": 1, "HKONT": hkont, "KOSTL": kostl, "RFAREA": rfarea, "SHKZG": opex_shkzg, "BSCHL": opex_bschl})
        actual_rows.append({**base_common, "BUZEI": 2, "HKONT": offset_hkont, "KOSTL": "", "RFAREA": "", "SHKZG": off_shkzg, "BSCHL": off_bschl})
        return belnr, dmbtr, ksl

    trial_pause_targets = {
        ("CC1010", "620010", 2022, 8), ("CC1010", "620010", 2022, 9),
        ("CC1030", "620020", 2023, 3), ("CC1030", "620020", 2023, 4),
        ("CC2010", "620010", 2023, 8), ("CC2010", "620010", 2023, 9),
        ("CC1010", "620010", 2024, 5), ("CC1010", "620010", 2024, 6),
        ("CC2020", "620020", 2024, 7), ("CC2020", "620020", 2024, 8),
        ("CC3010", "620010", 2024, 10), ("CC3010", "620010", 2024, 11),
        ("CC1030", "620010", 2025, 4), ("CC1030", "620010", 2025, 5),
        ("CC2010", "620020", 2025, 6), ("CC2010", "620020", 2025, 7),
    }
    q4_flush_targets = {
        ("CC1050", "630030", 2022, 12), ("CC1060", "630020", 2022, 11), ("CC1100", "650020", 2022, 12),
        ("CC2040", "630030", 2022, 11), ("CC1070", "640020", 2022, 12), ("CC1110", "650040", 2022, 12),
        ("CC1050", "630020", 2023, 11), ("CC1060", "630030", 2023, 12), ("CC1100", "650020", 2023, 12),
        ("CC2060", "630030", 2023, 11), ("CC2070", "640020", 2023, 12), ("CC3050", "650040", 2023, 12),
        ("CC1080", "630020", 2024, 11), ("CC1050", "630030", 2024, 12), ("CC2040", "630030", 2024, 12),
        ("CC2080", "650040", 2024, 11), ("CC1070", "640030", 2024, 12), ("CC3020", "630020", 2024, 12),
        ("CC1050", "630020", 2025, 11), ("CC1060", "630030", 2025, 12), ("CC1110", "650040", 2025, 12),
        ("CC2060", "630030", 2025, 12), ("CC2050", "630020", 2025, 11), ("CC3040", "640020", 2025, 12),
    }

    dup_candidates_same, dup_candidates_cross = [], []
    accrual_candidates, dec2025_accrual_candidates = [], []

    # Sort cells chronologically first for cleaner generation flow
    cell_df = cell_df.sort_values(["GJAHR", "MONAT", "BUKRS", "KOSTL", "HKONT"]).reset_index(drop=True)

    for row in cell_df.itertuples(index=False):
        yr, m = int(row.GJAHR), int(row.MONAT)
        bukrs, werks, kostl, hkont = row.BUKRS, row.WERKS, row.KOSTL, row.HKONT
        prctr, rfarea, wbs = row.PRCTR, row.RFAREA, row.WBS_ELEMENT
        lifnr, ebeln_1, ebeln_2 = row.LIFNR, row.EBELN_1, row.EBELN_2
        orig_runrate = float(row.RUNRATE_USD)
        runrate_usd = orig_runrate
        last_day = calendar.monthrange(yr, m)[1]
        d_probs = day_weight_cache[(yr, m)]
        days_arr = np.arange(1, last_day + 1)

        is_trial_pause = (kostl, hkont, yr, m) in trial_pause_targets
        if is_trial_pause:
            runrate_usd = round(orig_runrate * 0.38, 2)

        # Build behavior-aware invoice splits & PO assignments (guaranteeing BOTH ebeln_1 & ebeln_2 get invoiced)
        doc_specs: List[Tuple[float, str, bool]] = []  # (amount_usd, ebeln, is_prepaid_release)
        if row.USES_PO:
            n_total = max(2, int(rng.poisson(cfg.po_vendor_poisson_lam)))
            if ebeln_1 and ebeln_2:
                n_1 = max(1, min(n_total - 1, int(round(n_total * cfg.po_lead1_share))))
                n_2 = max(1, n_total - n_1)
                amt_1_tot = round(runrate_usd * cfg.po_lead1_share, 2)
                amt_2_tot = round(runrate_usd - amt_1_tot, 2)
                for a in _split_amount_dirichlet(amt_1_tot, n_1, cfg.po_dirichlet_alpha, rng):
                    doc_specs.append((a, ebeln_1, False))
                for a in _split_amount_dirichlet(amt_2_tot, n_2, cfg.po_dirichlet_alpha, rng):
                    doc_specs.append((a, ebeln_2, False))
            else:
                po_ref = ebeln_1 or ebeln_2
                for a in _split_amount_dirichlet(runrate_usd, n_total, cfg.po_dirichlet_alpha, rng):
                    doc_specs.append((a, po_ref, False))
            # Prepaids on Enterprise Cloud/SaaS (650010): first slice is quarterly prepaid release if non-PO; keep PO invoices as KR
            if hkont == "650010":
                # Keep one clean quarterly prepaid release line (SA/140000) without dropping PO coverage
                if len(doc_specs) >= 3:
                    a0, p0, _ = doc_specs[0]
                    # Only convert to prepaid release if another invoice still covers p0
                    if sum(1 for _, px, _ in doc_specs if px == p0) >= 2:
                        doc_specs[0] = (a0, "", True)
        elif row.OPEX_GROUP == "WORKFORCE_CORE":
            # Biweekly payroll runs (2 balanced postings)
            for a in _split_amount_dirichlet(runrate_usd, 2, 25.0, rng):
                doc_specs.append((a, "", False))
        elif row.COST_BEHAVIOR == "FIXED":
            # Fixed rent/leases: 1 quarterly prepaid release + 1 utility/CAM charge
            for idx_f, a in enumerate(_split_amount_dirichlet(runrate_usd, 2, 3.0, rng)):
                doc_specs.append((a, "", idx_f == 0 and hkont in ("620070", "650030")))
        elif row.COST_BEHAVIOR == "MILESTONE":
            doc_specs.append((runrate_usd, "", False))
        else:
            n_v = max(2, int(rng.poisson(cfg.non_po_var_poisson_lam)))
            for a in _split_amount_dirichlet(runrate_usd, n_v, cfg.non_po_var_dirichlet_alpha, rng):
                doc_specs.append((a, "", False))

        # Track largest KR invoice in this cell-month for realistic duplicate candidates
        kr_posted_this_cell = []
        next_yr, next_m = add_months(yr, m, 1)
        next_m_start = date(next_yr, next_m, 1)

        for idx, (amt, po_ref, is_prepaid) in enumerate(doc_specs):
            if is_prepaid:
                post_day = 1
                budat = date(yr, m, post_day)
                bldat = budat
                cpudt = budat + timedelta(days=int(rng.integers(0, 2)))
                blart, offset_gl, usnam = "SA", "140000", "BATCH_PREPAID"
                q_start_m = ((m - 1) // 3) * 3 + 1
                svc_start, svc_end = date(yr, q_start_m, 1), month_end_date(yr, q_start_m + 2)
            elif row.OPEX_GROUP == "WORKFORCE_CORE":
                post_day = 15 if idx == 0 else min(last_day, 28)
                budat = date(yr, m, post_day)
                bldat = budat
                cpudt = budat
                blart, offset_gl, usnam = "PR", "210000", "BATCH_PAYROLL"
                svc_start, svc_end = date(yr, m, 1), date(yr, m, last_day)
            else:
                post_day = int(rng.choice(days_arr, p=d_probs))
                budat = date(yr, m, post_day)
                bldat = budat - timedelta(days=int(rng.integers(0, 4)))
                if post_day >= last_day - 2 and rng.random() < 0.45:
                    cpudt = next_m_start + timedelta(days=int(rng.integers(1, 6)))
                else:
                    cpudt = min(budat + timedelta(days=int(rng.integers(0, 3))), next_m_start + timedelta(days=5))
                blart = "KR" if row.USES_PO else "SA"
                offset_gl = "215000" if blart == "KR" else "113100"
                usnam = str(rng.choice(human_users))
                svc_start, svc_end = date(yr, m, 1), date(yr, m, last_day)

            belnr, dmbtr_posted, ksl_posted = add_balanced_doc(
                bukrs, werks, yr, m, bldat, budat, cpudt, blart, usnam, hkont, offset_gl,
                kostl, prctr, rfarea, wbs, lifnr, po_ref, usd_amount=amt,
                svc_start=svc_start, svc_end=svc_end,
            )

            if is_trial_pause and idx == 0:
                anomaly_labels.append(
                    {
                        "BUKRS": bukrs, "GJAHR": yr, "MONAT": m, "BELNR": belnr, "BUZEI": 1,
                        "KOSTL": kostl, "HKONT": hkont, "ANOMALY_TAG": "TRIAL_PAUSE_DROP",
                        "GRANULARITY": "CELL_MONTH",
                        "ESTIMATED_IMPACT_USD": round(runrate_usd - orig_runrate, 2),
                        "DESCRIPTION": "Sudden 62% CRO spend drop due to temporary Phase III clinical hold",
                    }
                )

            if blart == "KR" and amt > 12_000.0:
                kr_posted_this_cell.append(
                    (post_day, bukrs, werks, yr, m, bldat, budat, hkont, kostl, prctr, rfarea, wbs, lifnr, po_ref, dmbtr_posted, ksl_posted, svc_start, svc_end)
                )

        if kr_posted_this_cell:
            # Pick the highest-value KR invoice in this cell-month as candidate
            best = max(kr_posted_this_cell, key=lambda x: x[15])
            p_day = best[0]
            cand = best[1:]
            if p_day >= 26 and not (yr == 2025 and m == 12):
                dup_candidates_cross.append(cand)
            elif p_day <= 24:
                dup_candidates_same.append(cand)

        # Q4_BUDGET_FLUSH surge
        if (kostl, hkont, yr, m) in q4_flush_targets:
            flush_amt = round(runrate_usd * float(rng.uniform(0.55, 0.72)), 2)
            budat = date(yr, m, min(last_day, 26))
            bldat = budat - timedelta(days=2)
            cpudt = budat + timedelta(days=2)
            belnr, _, ksl_flush = add_balanced_doc(
                bukrs, werks, yr, m, bldat, budat, cpudt, "KR", str(rng.choice(human_users)),
                hkont, "215000", kostl, prctr, rfarea, wbs, lifnr, ebeln_1, usd_amount=flush_amt,
                svc_start=date(yr, m, 1), svc_end=date(yr, m, last_day),
            )
            anomaly_labels.append(
                {
                    "BUKRS": bukrs, "GJAHR": yr, "MONAT": m, "BELNR": belnr, "BUZEI": 1,
                    "KOSTL": kostl, "HKONT": hkont, "ANOMALY_TAG": "Q4_BUDGET_FLUSH",
                    "GRANULARITY": "CELL_MONTH",
                    "ESTIMATED_IMPACT_USD": round(ksl_flush, 2),
                    "DESCRIPTION": "Abnormal Q4 use-it-or-lose-it discretionary vendor spend surge",
                }
            )

        # Month-end Accrual candidates (including Dec 2025 open year-end accruals)
        if row.USES_PO and rng.random() < cfg.accrual_prob:
            acc_amt = round(runrate_usd * float(rng.uniform(0.10, 0.18)), 2)
            acc_budat = date(yr, m, last_day)
            acc_bldat = acc_budat - timedelta(days=1)
            acc_cpudt = date(next_yr, next_m, int(rng.integers(2, 6)))
            item = (bukrs, werks, yr, m, next_yr, next_m, acc_bldat, acc_budat, acc_cpudt, hkont, kostl, prctr, rfarea, wbs, lifnr, ebeln_1, acc_amt)
            if yr == 2025 and m == 12:
                dec2025_accrual_candidates.append(item)
            else:
                accrual_candidates.append(item)

    # Dedicated Accrual Doc Type BLART="ZA" (human booked) + Reversal BLART="AB" (BATCH_REV)
    rng.shuffle(accrual_candidates)
    n_unrev = min(cfg.n_unreversed_accruals, len(accrual_candidates))
    for i, item in enumerate(accrual_candidates):
        bukrs, werks, yr, m, next_yr, next_m, acc_bldat, acc_budat, acc_cpudt, hkont, kostl, prctr, rfarea, wbs, lifnr, po_ref, acc_amt = item
        if i < n_unrev:
            belnr_acc, _, ksl_acc = add_balanced_doc(
                bukrs, werks, yr, m, acc_bldat, acc_budat, acc_cpudt, "ZA", str(rng.choice(human_users)),
                hkont, "215000", kostl, prctr, rfarea, wbs, lifnr, po_ref, usd_amount=acc_amt,
                is_reversal=False, stblg="", stjah=0,
            )
            anomaly_labels.append(
                {
                    "BUKRS": bukrs, "GJAHR": yr, "MONAT": m, "BELNR": belnr_acc, "BUZEI": 1,
                    "KOSTL": kostl, "HKONT": hkont, "ANOMALY_TAG": "UNREVERSED_ACCRUAL",
                    "GRANULARITY": "DOCUMENT",
                    "ESTIMATED_IMPACT_USD": round(ksl_acc, 2),
                    "DESCRIPTION": "Month-end accrual (ZA) missing auto-reversal (AB) in following period",
                }
            )
        else:
            belnr_acc = next_tmp_belnr(bukrs, yr)
            belnr_rev = next_tmp_belnr(bukrs, next_yr)
            _, za_dmbtr, _ = add_balanced_doc(
                bukrs, werks, yr, m, acc_bldat, acc_budat, acc_cpudt, "ZA", str(rng.choice(human_users)),
                hkont, "215000", kostl, prctr, rfarea, wbs, lifnr, po_ref, usd_amount=acc_amt,
                is_reversal=False, stblg=belnr_rev, stjah=next_yr, belnr_override=belnr_acc,
            )
            rev_budat = date(next_yr, next_m, 1)
            add_balanced_doc(
                bukrs, werks, next_yr, next_m, rev_budat, rev_budat, rev_budat + timedelta(days=1),
                "AB", "BATCH_REV", hkont, "215000", kostl, prctr, rfarea, wbs, lifnr, po_ref,
                dmbtr_override=za_dmbtr, is_reversal=True, stblg=belnr_acc, stjah=yr, belnr_override=belnr_rev,
            )

    # Legitimate Dec 2025 open year-end accruals (reversals occur in Jan 2026)
    for item in dec2025_accrual_candidates:
        bukrs, werks, yr, m, next_yr, next_m, acc_bldat, acc_budat, acc_cpudt, hkont, kostl, prctr, rfarea, wbs, lifnr, po_ref, acc_amt = item
        add_balanced_doc(
            bukrs, werks, yr, m, acc_bldat, acc_budat, acc_cpudt, "ZA", str(rng.choice(human_users)),
            hkont, "215000", kostl, prctr, rfarea, wbs, lifnr, po_ref, usd_amount=acc_amt,
            is_reversal=False, stblg="", stjah=0,
        )

    # 36 DUPLICATE_INVOICE cases (same-month + cross-month boundary)
    rng.shuffle(dup_candidates_same)
    rng.shuffle(dup_candidates_cross)
    n_cross = min(cfg.n_cross_month_duplicates, len(dup_candidates_cross))
    n_same = min(cfg.n_duplicate_invoices - n_cross, len(dup_candidates_same))

    for item in dup_candidates_same[:n_same]:
        bukrs, werks, yr, m, bldat, budat, hkont, kostl, prctr, rfarea, wbs, lifnr, po_ref, orig_dmbtr, _, svc_s, svc_e = item
        last_day = calendar.monthrange(yr, m)[1]
        dup_budat = date(yr, m, min(last_day, budat.day + 2))
        dup_cpudt = dup_budat + timedelta(days=1)
        belnr_dup, _, ksl_dup = add_balanced_doc(
            bukrs, werks, yr, m, bldat, dup_budat, dup_cpudt, "KR", str(rng.choice(human_users)),
            hkont, "215000", kostl, prctr, rfarea, wbs, lifnr, po_ref, dmbtr_override=orig_dmbtr,
            svc_start=svc_s, svc_end=svc_e,
        )
        anomaly_labels.append(
            {
                "BUKRS": bukrs, "GJAHR": yr, "MONAT": m, "BELNR": belnr_dup, "BUZEI": 1,
                "KOSTL": kostl, "HKONT": hkont, "ANOMALY_TAG": "DUPLICATE_INVOICE",
                "GRANULARITY": "DOCUMENT",
                "ESTIMATED_IMPACT_USD": round(ksl_dup, 2),
                "DESCRIPTION": "Same-month duplicate vendor invoice matching prior LIFNR, BLDAT, and DMBTR",
            }
        )

    for item in dup_candidates_cross[:n_cross]:
        bukrs, werks, yr, m, bldat, budat, hkont, kostl, prctr, rfarea, wbs, lifnr, po_ref, orig_dmbtr, _, svc_s, svc_e = item
        next_yr, next_m = add_months(yr, m, 1)
        dup_budat = date(next_yr, next_m, int(rng.integers(1, 4)))
        dup_cpudt = dup_budat + timedelta(days=1)
        belnr_dup, _, ksl_dup = add_balanced_doc(
            bukrs, werks, next_yr, next_m, bldat, dup_budat, dup_cpudt, "KR", str(rng.choice(human_users)),
            hkont, "215000", kostl, prctr, rfarea, wbs, lifnr, po_ref, dmbtr_override=orig_dmbtr,
            svc_start=svc_s, svc_end=svc_e,
        )
        anomaly_labels.append(
            {
                "BUKRS": bukrs, "GJAHR": next_yr, "MONAT": next_m, "BELNR": belnr_dup, "BUZEI": 1,
                "KOSTL": kostl, "HKONT": hkont, "ANOMALY_TAG": "DUPLICATE_INVOICE",
                "GRANULARITY": "DOCUMENT",
                "ESTIMATED_IMPACT_USD": round(ksl_dup, 2),
                "DESCRIPTION": "Cross-month duplicate vendor invoice matching prior-month LIFNR, BLDAT, and DMBTR",
            }
        )

    # GAAP Special Items
    special_events = [
        ("US10", 2022, 9, "CC1010", "620030", "ACQUIRED_IPRD", 4_800_000.0, "VND_BIO_ACQ01"),
        ("US10", 2023, 6, "CC1110", "650040", "LITIGATION", 3_200_000.0, "VND_LEG_SETTLE"),
        ("EU20", 2023, 11, "CC2080", "650040", "RESTRUCTURING", 2_100_000.0, "VND_ADV_RESTR"),
        ("US10", 2024, 3, "CC1090", "620030", "ACQUIRED_IPRD", 6_500_000.0, "VND_MED_ACQ02"),
        ("US10", 2024, 10, "CC1100", "650020", "MA_INTEGRATION", 1_950_000.0, "VND_IT_ACCENTURE"),
        ("US10", 2025, 4, "CC1030", "620030", "ACQUIRED_IPRD", 5_400_000.0, "VND_BIO_ACQ03"),
        ("EU20", 2025, 8, "CC2080", "650040", "RESTRUCTURING", 1_750_000.0, "VND_ADV_RESTR"),
        ("US10", 2025, 11, "CC1110", "650040", "LITIGATION", 2_850_000.0, "VND_LEG_SETTLE"),
    ]
    cc_idx = dim_cc.set_index("KOSTL")
    for bukrs, yr, m, kostl, hkont, sp_flag, amt_usd, lifnr in special_events:
        cc = cc_idx.loc[kostl]
        budat = date(yr, m, 20)
        add_balanced_doc(
            bukrs, cc["WERKS"], yr, m, budat - timedelta(days=2), budat, budat + timedelta(days=2),
            "KR", str(rng.choice(human_users)), hkont, "215000", kostl, cc["PRCTR"], cc["RFAREA"],
            cc["DEFAULT_WBS"], lifnr, "", usd_amount=amt_usd, special_flag=sp_flag,
        )

    # Special Periods 13-16 (strictly honoring ALLOWED_RFAREAS)
    sp_gl_by_rfarea = {
        "R&D": ("610030", "620010"),
        "Commercial_SGA": ("610030", "630030"),
        "Medical_Affairs": ("610030", "640020"),
        "G&A": ("610030", "650040"),
    }
    for yr in cfg.actual_years:
        for sp_period in (13, 14, 15, 16):
            sample_ccs = dim_cc.sample(n=4, random_state=int(rng.integers(0, 10000)))
            for cc in sample_ccs.itertuples(index=False):
                bonus_gl, audit_gl = sp_gl_by_rfarea[cc.RFAREA]
                hkont = bonus_gl if sp_period == 13 else audit_gl
                budat = date(yr, 12, 31)
                cpudt = date(yr + 1, 1, min(28, 4 + (sp_period - 13) * 4))
                adj_usd = round(float(rng.uniform(18_000.0, 95_000.0)) * cc.COST_SCALE, 2)
                add_balanced_doc(
                    cc.BUKRS, cc.WERKS, yr, sp_period, budat, budat, cpudt, "SA", "BATCH_AUDIT",
                    hkont, "215000", cc.KOSTL, cc.PRCTR, cc.RFAREA, cc.DEFAULT_WBS,
                    "YEAR_END_AUDIT", "", usd_amount=adj_usd,
                )

    # -----------------------------------------------------------------------
    # Chronological Sort & Sequential BELNR Reassignment (Eliminates Leakage!)
    # -----------------------------------------------------------------------
    acdoca_actuals = pd.DataFrame(actual_rows)
    eval_anomalies = pd.DataFrame(anomaly_labels)

    docs = (
        acdoca_actuals[acdoca_actuals["BUZEI"] == 1][["BUKRS", "GJAHR", "MONAT", "BUDAT", "CPUDT", "BLART", "BELNR"]]
        .sort_values(["BUKRS", "GJAHR", "MONAT", "BUDAT", "CPUDT", "BLART", "BELNR"])
        .reset_index(drop=True)
    )
    docs["SEQ"] = docs.groupby(["BUKRS", "GJAHR"]).cumcount() + 100000001
    docs["NEW_BELNR"] = docs["SEQ"].map(lambda x: f"{x:010d}")

    belnr_map: Dict[Tuple[str, int, str], str] = {
        (r.BUKRS, int(r.GJAHR), r.BELNR): r.NEW_BELNR for r in docs.itertuples(index=False)
    }

    acdoca_actuals["BELNR"] = [
        belnr_map[(b, int(y), n)]
        for b, y, n in zip(acdoca_actuals["BUKRS"], acdoca_actuals["GJAHR"], acdoca_actuals["BELNR"])
    ]
    acdoca_actuals["STBLG"] = [
        belnr_map[(b, int(sy), st)] if st else ""
        for b, sy, st in zip(acdoca_actuals["BUKRS"], acdoca_actuals["STJAH"], acdoca_actuals["STBLG"])
    ]
    eval_anomalies["BELNR"] = [
        belnr_map[(b, int(y), n)]
        for b, y, n in zip(eval_anomalies["BUKRS"], eval_anomalies["GJAHR"], eval_anomalies["BELNR"])
    ]

    acdoca_actuals = (
        acdoca_actuals.sort_values(["BUKRS", "GJAHR", "MONAT", "BUDAT", "BELNR", "BUZEI"]).reset_index(drop=True)
    )
    eval_anomalies = (
        eval_anomalies.sort_values(["BUKRS", "GJAHR", "MONAT", "BELNR", "BUZEI"]).reset_index(drop=True)
    )

    # Derive S/4HANA Universal Journal strictly AFTER chronological sort & BELNR remap
    sign = np.where(acdoca_actuals["SHKZG"] == "S", 1.0, -1.0)
    acdoca_uj = pd.DataFrame(
        {
            "RBUKRS": acdoca_actuals["BUKRS"],
            "WERKS": acdoca_actuals["WERKS"],
            "GJAHR": acdoca_actuals["GJAHR"],
            "RYEAR": acdoca_actuals["GJAHR"],
            "POPER": acdoca_actuals["MONAT"],
            "BELNR": acdoca_actuals["BELNR"],
            "DOCLN": acdoca_actuals["BUZEI"].map(lambda x: f"{int(x):06d}"),
            "BLDAT": acdoca_actuals["BLDAT"],
            "BUDAT": acdoca_actuals["BUDAT"],
            "CPUDT": acdoca_actuals["CPUDT"],
            "BLART": acdoca_actuals["BLART"],
            "USNAM": acdoca_actuals["USNAM"],
            "RACCT": acdoca_actuals["HKONT"],
            "RCNTR": acdoca_actuals["KOSTL"],
            "PRCTR": acdoca_actuals["PRCTR"],
            "RFAREA": acdoca_actuals["RFAREA"],
            "PS_PSP_PNR": acdoca_actuals["WBS_ELEMENT"],
            "LIFNR": acdoca_actuals["LIFNR"],
            "EBELN": acdoca_actuals["EBELN"],
            "RWCUR": acdoca_actuals["WAERS"],
            "DRCRK": acdoca_actuals["SHKZG"],
            "BSCHL": acdoca_actuals["BSCHL"],
            "HSL": np.round(acdoca_actuals["DMBTR"] * sign, 2),
            "KSL": np.round(acdoca_actuals["KSL"] * sign, 2),
            "KSL_BUDGET_FX": np.round(acdoca_actuals["KSL_BUDGET_FX"] * sign, 2),
            "STBLG": acdoca_actuals["STBLG"],
            "STJAH": acdoca_actuals["STJAH"],
            "SPECIAL_ITEM_FLAG": acdoca_actuals["SPECIAL_ITEM_FLAG"],
            "SERVICE_START_DATE": acdoca_actuals["SERVICE_START_DATE"],
            "SERVICE_END_DATE": acdoca_actuals["SERVICE_END_DATE"],
            "RLDNR": "0L",
            "VERSION": 0,
        }
    )

    return fact_po, acdoca_actuals, acdoca_uj, eval_anomalies


# ---------------------------------------------------------------------------
# Stage 5: Leak-Free J&J Plan Versions (BP, Q1_RF 3+9, JU 5+7, Q3_RF 9+3, NU 10+2)
# ---------------------------------------------------------------------------
def build_plan_versions(
    dims: Dict[str, pd.DataFrame],
    plan_drivers: pd.DataFrame,
    acdoca_uj: pd.DataFrame,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    dim_cc = dims["dim_cost_center"].set_index("KOSTL")
    dim_comp = dims["dim_company_code"].set_index("BUKRS")
    dim_gl = dims["dim_gl_account"]
    dim_fx = dims["dim_fx_rates"]

    fx_lookup = {
        (int(r.GJAHR), int(r.MONAT), str(r.FROM_CURR), str(r.RATE_TYPE)): float(r.FX_RATE)
        for r in dim_fx.itertuples(index=False)
    }
    opex_set = set(dim_gl.loc[dim_gl["ACCOUNT_TYPE"] == "OPEX", "HKONT"])
    clean_uj = acdoca_uj[
        (acdoca_uj["RACCT"].isin(opex_set))
        & (acdoca_uj["POPER"] <= 12)
        & (acdoca_uj["SPECIAL_ITEM_FLAG"] == "NONE")
    ].copy()

    pdrv_lookup = {(int(r.GJAHR), int(r.MONAT), str(r.KOSTL)): r._asdict() for r in plan_drivers.itertuples(index=False)}
    active_pairs = sorted(set(zip(clean_uj["RCNTR"], clean_uj["RACCT"])))
    plan_rows: List[dict] = []
    bp_map: Dict[Tuple[int, int, str, str], float] = {}

    for yr in cfg.all_years:
        snap_date = f"{yr - 1}-10-15"
        for kostl, hkont in active_pairs:
            cc = dim_cc.loc[kostl]
            bukrs, werks, region, scale = str(cc["BUKRS"]), str(cc["WERKS"]), str(cc["REGION"]), float(cc["COST_SCALE"])
            waers = str(dim_comp.loc[bukrs, "WAERS"])
            cc_plan_bias = float(rng.uniform(0.94, 1.06)) if hkont not in ("620030", "620060") else 1.0

            for m in range(1, 13):
                pdrv = pdrv_lookup[(yr, m, kostl)]
                causal_drv = {
                    "GJAHR": yr, "MONAT": m,
                    "ACTIVE_FTE": pdrv["PLAN_FTE"],
                    "FIELD_SALES_REPS": round(pdrv["PLAN_FTE"] * 0.68, 1) if cc["RFAREA"] == "Commercial_SGA" else 0.0,
                    "ACTIVE_SITES": pdrv["PLAN_ACTIVE_SITES"],
                    "ENROLLED_PATIENTS": pdrv["PLAN_ENROLLED_PATIENTS"],
                    "HCP_EVENTS_COUNT": pdrv["PLAN_HCP_EVENTS"],
                    "PRODUCT_LAUNCH_FLAG": 0, "LOE_FLAG": 0,
                    "SURGICAL_PROCEDURE_INDEX": 100.0 * (1.0 + 0.025 * (yr - 2022)),
                    "MILESTONE_EVENT": pdrv["PLAN_MILESTONE_EVENT"],
                    "CPI_INDEX": pdrv["PLAN_CPI_INDEX"], "WAGE_INDEX": pdrv["PLAN_WAGE_INDEX"],
                }
                bp_usd_det, _ = compute_cell_runrate_usd(hkont, causal_drv, causal_drv, scale, region)
                bp_usd_raw = bp_usd_det * cc_plan_bias * (float(rng.uniform(0.985, 1.015)) if bp_usd_det > 0 else 1.0)

                plan_fx = fx_lookup[(yr, m, waers, "BUDGET_PLAN")]
                plan_lc = round(bp_usd_raw / plan_fx, 2)
                bp_usd = round(plan_lc * plan_fx, 2)
                bp_map[(yr, m, kostl, hkont)] = bp_usd

                plan_rows.append(
                    {
                        "GJAHR": yr, "MONAT": m, "BUKRS": bukrs, "WERKS": werks,
                        "KOSTL": kostl, "HKONT": hkont, "VERSION": 1, "CYCLE_NAME": "BP",
                        "SNAPSHOT_DATE": snap_date, "WAERS": waers, "PLAN_FX_RATE": plan_fx,
                        "DMBTR_PLAN": plan_lc, "KSL_BUDGET": bp_usd,
                        "PLAN_FTE": pdrv["PLAN_FTE"],
                        "PLAN_ENROLLED_PATIENTS": pdrv["PLAN_ENROLLED_PATIENTS"],
                        "PLAN_HCP_EVENTS": pdrv["PLAN_HCP_EVENTS"],
                    }
                )

    rf_cycles = [("Q1_RF", 3, "04-10"), ("JU", 5, "06-15"), ("Q3_RF", 9, "10-10"), ("NU", 10, "11-15")]
    for yr in cfg.actual_years:
        yr_uj = clean_uj[clean_uj["RYEAR"] == yr]
        for cycle_name, cutoff_m, md_str in rf_cycles:
            snap_date = f"{yr}-{md_str}"
            closed_uj = yr_uj[(yr_uj["POPER"] <= cutoff_m) & (yr_uj["CPUDT"] <= snap_date)]
            closed_agg = closed_uj.groupby(["POPER", "RCNTR", "RACCT"], as_index=False)["HSL"].sum()
            hsl_map = {(int(r.POPER), r.RCNTR, r.RACCT): float(r.HSL) for r in closed_agg.itertuples(index=False)}

            for kostl, hkont in active_pairs:
                cc = dim_cc.loc[kostl]
                bukrs, werks = str(cc["BUKRS"]), str(cc["WERKS"])
                waers = str(dim_comp.loc[bukrs, "WAERS"])
                plan_fx = fx_lookup[(yr, 1, waers, "BUDGET_PLAN")]

                trail_start = max(1, cutoff_m - 2)
                trail_act = sum(round(hsl_map.get((mm, kostl, hkont), 0.0) * plan_fx, 2) for mm in range(trail_start, cutoff_m + 1))
                trail_bp = sum(bp_map.get((yr, mm, kostl, hkont), 0.0) for mm in range(trail_start, cutoff_m + 1))
                ytd_act = sum(round(hsl_map.get((mm, kostl, hkont), 0.0) * plan_fx, 2) for mm in range(1, cutoff_m + 1))
                ytd_bp = sum(bp_map.get((yr, mm, kostl, hkont), 0.0) for mm in range(1, cutoff_m + 1))

                if hkont in ("620030", "620060") or ytd_bp <= 500.0:
                    eff_ratio = 1.0
                else:
                    r_trail = float(np.clip(trail_act / trail_bp, 0.72, 1.35)) if trail_bp > 500.0 else 1.0
                    r_ytd = float(np.clip(ytd_act / ytd_bp, 0.72, 1.35))
                    eff_ratio = 0.65 * r_trail + 0.35 * r_ytd

                for m in range(1, 13):
                    pdrv = pdrv_lookup[(yr, m, kostl)]
                    bp_val = bp_map[(yr, m, kostl, hkont)]
                    if m <= cutoff_m:
                        plan_lc = round(hsl_map.get((m, kostl, hkont), 0.0), 2)
                        rf_usd = round(plan_lc * plan_fx, 2)
                    else:
                        h = m - cutoff_m
                        w = 0.88 * (0.94 ** (h - 1))
                        rf_usd_raw = ((1.0 - w) * bp_val + w * bp_val * eff_ratio) * (
                            float(rng.uniform(0.985, 1.015)) if bp_val > 0 else 1.0
                        )
                        plan_lc = round(rf_usd_raw / plan_fx, 2)
                        rf_usd = round(plan_lc * plan_fx, 2)

                    plan_rows.append(
                        {
                            "GJAHR": yr, "MONAT": m, "BUKRS": bukrs, "WERKS": werks,
                            "KOSTL": kostl, "HKONT": hkont, "VERSION": 2, "CYCLE_NAME": cycle_name,
                            "SNAPSHOT_DATE": snap_date, "WAERS": waers, "PLAN_FX_RATE": plan_fx,
                            "DMBTR_PLAN": plan_lc, "KSL_BUDGET": rf_usd,
                            "PLAN_FTE": pdrv["PLAN_FTE"],
                            "PLAN_ENROLLED_PATIENTS": pdrv["PLAN_ENROLLED_PATIENTS"],
                            "PLAN_HCP_EVENTS": pdrv["PLAN_HCP_EVENTS"],
                        }
                    )

    return pd.DataFrame(plan_rows)


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAP-Inspired Analytical Pharma OPEX Dataset (v2.2 Freeze)")
    parser.add_argument("out_dir", nargs="?", default="./data/raw/pharma_v22", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    cfg = GeneratorConfig(seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    print(f"[1/6] Building J&J master dimensions (incl. dim_plant & 210000 Payroll Clearing) + FX (seed={cfg.seed})...")
    dims = build_dimensions(cfg, rng)

    print("[2/6] Generating realized drivers (FY22-25) & Nov-Dec bridged causal BP drivers (FY22-26)...")
    fact_drivers, plan_drivers, dec2025_state, cpi_map, wage_idx_map, wage_rate_map = build_opex_drivers(
        dims["dim_cost_center"], cfg, rng
    )

    print("[3/6] Building deterministic FY2026 What-If scenarios from Dec 2025 state...")
    scenarios = build_scenario_assumptions(dec2025_state, dims["dim_fx_rates"], cpi_map, wage_idx_map, wage_rate_map)
    scenario_base_26 = scenarios[scenarios["SCENARIO_ID"] == "BASE"].copy()

    print("[4/6] Generating PO lifecycle & chronologically ordered ACDOCA actuals (reconciled EBELN, USNAM/BSCHL)...")
    fact_po, acdoca_actuals, acdoca_uj, eval_anomalies = build_po_and_actuals(
        dims, fact_drivers, scenario_base_26, cfg, rng
    )

    print("[5/6] Generating leak-free J&J plan versions (BP, Q1_RF 3+9, JU 5+7, Q3_RF 9+3, NU 10+2)...")
    plan_versions = build_plan_versions(dims, plan_drivers, acdoca_uj, cfg, rng)

    print(f"[6/6] Writing 14 CSV tables to {args.out_dir}...")
    outputs = {
        **dims,
        "fact_opex_drivers": fact_drivers,
        "plan_drivers": plan_drivers,
        "fact_po_commitments": fact_po,
        "acdoca_actuals": acdoca_actuals,
        "acdoca_universal_journal": acdoca_uj,
        "plan_versions": plan_versions,
        "scenario_assumptions": scenarios,
        "eval_anomaly_labels": eval_anomalies,
    }
    for name, df in outputs.items():
        path = os.path.join(args.out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  - {name}.csv: {len(df):,} rows x {df.shape[1]} cols")

    print("\nv2.2 generation complete.")


if __name__ == "__main__":
    main()