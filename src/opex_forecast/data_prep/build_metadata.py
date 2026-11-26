#!/usr/bin/env python3
"""
Deterministic Metadata & Data Dictionary Builder (v2.2.2)
=========================================================
Reads the generated v2.2 CSV tables directly from disk (e.g., ./data/raw/pharma_v22),
asserts exact schema parity against all 14 tables (206 columns), samples REAL
non-blank example values and observed domains from every column, computes exact
row counts / anomaly rates / EBELN & PO reconciliation metrics, and emits:
  1. pharma_opex_v22_metadata.json
  2. pharma_opex_v22_data_dictionary.csv

Usage:
  python src/opex_forecast/data_prep/build_metadata.py ./data/raw/pharma_v22 ./docs [--seed 42] [--validation-checks 36] [--markdown ./docs/sap_pharma_opex_data_dictionary.md]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from opex_forecast.column_map import get_table_column_map
from opex_forecast.config import BLANK_STR_COLS, STR_COLS
from opex_forecast.io import load_pharma_table


def sample_real_example(series: pd.Series) -> Any:
    """Return the first non-null, non-empty real value from the actual CSV column."""
    for val in series:
        if pd.isna(val):
            continue
        if isinstance(val, str) and val == "":
            continue
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return round(float(val), 6)
        if isinstance(val, (np.bool_,)):
            return bool(val)
        return val
    return ""


def observed_unique_values(series: pd.Series, max_vals: int = 16) -> Optional[List[Any]]:
    uniq = series.dropna().unique().tolist()
    if len(uniq) > max_vals:
        return None
    cleaned = []
    for v in uniq:
        if isinstance(v, np.integer):
            cleaned.append(int(v))
        elif isinstance(v, np.floating):
            cleaned.append(round(float(v), 6))
        elif isinstance(v, np.bool_):
            cleaned.append(bool(v))
        else:
            cleaned.append(v)
    return sorted(cleaned, key=lambda x: (0, x) if isinstance(x, (int, float)) and not isinstance(x, bool) else (1, str(x)))


def col_spec(
    name: str,
    dtype: str,
    pk: bool = False,
    uk: bool = False,
    fk: Optional[str] = None,
    nullable: bool = False,
    provenance: str = "SAP_STANDARD",
    sap_ref: Optional[str] = None,
    allowed: Optional[List[Any]] = None,
    ml_role: str = "METADATA",
    desc: str = "",
    ecc_equiv: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "column_name": name,
        "dtype": dtype,
        "is_primary_key": pk,
        "is_unique_key": uk,
        "foreign_key": fk,
        "nullable": nullable,
        "provenance": provenance,
        "sap_reference": sap_ref,
        "allowed_values": allowed,
        "ml_role": ml_role,
        "ecc_equivalent": ecc_equiv,
        "description": desc,
    }


def driver_columns_schema(is_scenario: bool = False) -> List[Dict[str, Any]]:
    yr_note = "Fiscal year (2026 in scenario_assumptions)." if is_scenario else "Fiscal year (2022-2025)."
    hires_note = (
        "Deterministic planned monthly hires (set to 0 from M4 onward under HIRING_FREEZE_Q2)."
        if is_scenario
        else "Realized gross monthly Poisson hires feeding ACTIVE_FTE evolution."
    )
    return [
        col_spec("GJAHR", "int64", pk=True, sap_ref="GJAHR", ml_role="TIME_INDEX", desc=yr_note),
        col_spec("MONAT", "int64", pk=True, sap_ref="MONAT", allowed=list(range(1, 13)), ml_role="TIME_INDEX", desc="Fiscal posting period (1-12)."),
        col_spec("BUKRS", "string", fk="dim_company_code.BUKRS", sap_ref="BUKRS", allowed=["US10", "EU20", "AP30"], ml_role="HIERARCHY_KEY", desc="Company code owning the cost center."),
        col_spec("WERKS", "string", fk="dim_plant.WERKS", provenance="ANALYTICAL_EXTENSION", sap_ref="Plant Link", allowed=["US01", "US02", "US03", "BE01", "NL01", "IN01"], ml_role="HIERARCHY_KEY", desc="Plant / campus code where cost center operates."),
        col_spec("KOSTL", "string", pk=True, fk="dim_cost_center.KOSTL", sap_ref="CSKS-KOSTL", ml_role="ENTITY_KEY", desc="Cost center identifier."),
        col_spec("WBS_ELEMENT", "string", sap_ref="PRPS-POSID", ml_role="HIERARCHY_KEY", desc="Clinical trial WBS element ('NONE' for non-project cost centers)."),
        col_spec("ACTIVE_FTE", "float64", provenance="ANALYTICAL_EXTENSION", sap_ref="HR/Workday", ml_role="EXOGENOUS_DRIVER", desc="Active full-time equivalent headcount; primary driver of workforce and step-fixed IT OPEX."),
        col_spec("PLANNED_HIRES", "int64", provenance="ANALYTICAL_EXTENSION", sap_ref="HR/Workday", ml_role="EXOGENOUS_DRIVER", desc=hires_note),
        col_spec("FIELD_SALES_REPS", "float64", provenance="ANALYTICAL_EXTENSION", sap_ref="CRM/HR", ml_role="EXOGENOUS_DRIVER", desc="Field commercial sales force headcount (~68% of Commercial_SGA FTE; 0.0 otherwise)."),
        col_spec("MERIT_CYCLE_FLAG", "int64", provenance="ANALYTICAL_EXTENSION", allowed=[0, 1], ml_role="CALENDAR_DRIVER", desc="Binary indicator for annual April merit step-up (1 in April, 0 otherwise)."),
        col_spec("TRIAL_PHASE", "string", provenance="ANALYTICAL_EXTENSION", sap_ref="CTMS", allowed=["NONE", "P1", "P2", "P3", "FILING"], ml_role="EXOGENOUS_DRIVER", desc="Clinical development phase (P1 graduates to P2 in FY25; P2 graduates to P3 in FY24)."),
        col_spec("ACTIVE_SITES", "float64", provenance="ANALYTICAL_EXTENSION", sap_ref="CTMS", ml_role="EXOGENOUS_DRIVER", desc="Active clinical investigator sites (enters CRO spend with a 1-month billing lag)."),
        col_spec("ENROLLED_PATIENTS", "float64", provenance="ANALYTICAL_EXTENSION", sap_ref="CTMS", ml_role="EXOGENOUS_DRIVER", desc="Active enrolled trial patients (enters CRO/site spend with a 1-month billing lag)."),
        col_spec("MILESTONE_EVENT", "string", provenance="ANALYTICAL_EXTENSION", sap_ref="CTMS", allowed=["NONE", "FPI", "DB_LOCK", "FDA_FILING"], ml_role="EXOGENOUS_DRIVER", desc="Realized or scenario clinical/regulatory milestone triggering lump-sum payments."),
        col_spec("PRODUCT_LAUNCH_FLAG", "int64", provenance="ANALYTICAL_EXTENSION", allowed=[0, 1], ml_role="EXOGENOUS_DRIVER", desc="Binary commercial launch flag (Oncology FY23 M4-M10; Immunology FY25 M2-M7; FY26 M3-M9 under LAUNCH_ACCEL)."),
        col_spec("LOE_FLAG", "int64", provenance="ANALYTICAL_EXTENSION", allowed=[0, 1], ml_role="EXOGENOUS_DRIVER", desc="Binary Loss of Exclusivity / biosimilar defense flag (Immunology Commercial_SGA FY24 M7-M12; 0 otherwise)."),
        col_spec("HCP_EVENTS_COUNT", "int64", provenance="ANALYTICAL_EXTENSION", sap_ref="Commercial_CRM", ml_role="EXOGENOUS_DRIVER", desc="Monthly Healthcare Professional speaker programs & congress events (Commercial_SGA only)."),
        col_spec("SURGICAL_PROCEDURE_INDEX", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="EXOGENOUS_DRIVER", desc="MedTech surgical procedure volume index (base ~100; 100.0 for non-MedTech)."),
        col_spec("CPI_INDEX", "float64", provenance="MACRO_EXTERNAL", ml_role="MACRO_DRIVER", desc="Compounded Consumer Price Index (base 100.0 in Jan 2022)."),
        col_spec("WAGE_INDEX", "float64", provenance="MACRO_EXTERNAL", ml_role="MACRO_DRIVER", desc="Compounded wage index stepping monotonically only in April (base 100.0 in Jan 2022)."),
        col_spec("WAGE_INFLATION_RATE", "float64", provenance="MACRO_EXTERNAL", ml_role="MACRO_DRIVER", desc="Annualized merit wage inflation rate (0.048 FY22 -> 0.033 FY26)."),
    ]


def build_catalog(data_dir: str, seed: int = 42, validation_checks_count: int = 36) -> Dict[str, Any]:
    # Load all 14 CSVs directly from disk to verify schema parity and compute real metrics
    table_defs: List[Dict[str, Any]] = [
        {
            "table_name": "dim_company_code",
            "file_name": "dim_company_code.csv",
            "category": "DIMENSION",
            "time_span": "Static",
            "ml_pipeline_role": "Entity / Currency Lookup",
            "sap_inspiration": "T001",
            "primary_key": ["BUKRS"],
            "description": "Legal entity master defining company codes, local reporting currencies, and regions.",
            "columns": [
                col_spec("BUKRS", "string", pk=True, sap_ref="T001-BUKRS", allowed=["US10", "EU20", "AP30"], ml_role="HIERARCHY_KEY", desc="SAP Company Code."),
                col_spec("BUTXT", "string", sap_ref="T001-BUTXT", ml_role="METADATA", desc="Legal entity name."),
                col_spec("WAERS", "string", sap_ref="T001-WAERS", allowed=["USD", "EUR", "INR"], ml_role="CURRENCY_KEY", desc="Local functional currency (1:1 per BUKRS here; N:1 in multi-entity SAP)."),
                col_spec("REGION", "string", provenance="ANALYTICAL_EXTENSION", allowed=["US", "EMEA", "APAC"], ml_role="HIERARCHY_KEY", desc="Geographic reporting region."),
            ],
        },
        {
            "table_name": "dim_plant",
            "file_name": "dim_plant.csv",
            "category": "DIMENSION",
            "time_span": "Static",
            "ml_pipeline_role": "Campus / Geography Slicing",
            "sap_inspiration": "T001W / T001K",
            "primary_key": ["WERKS"],
            "description": "Six J&J-aligned campuses (US01 Titusville, US02 Raritan, US03 Spring House, BE01 Beerse, NL01 Leiden, IN01 Mumbai/Hyderabad) mapped N:1 to company codes.",
            "columns": [
                col_spec("WERKS", "string", pk=True, sap_ref="T001W-WERKS", allowed=["US01", "US02", "US03", "BE01", "NL01", "IN01"], ml_role="HIERARCHY_KEY", desc="Plant / campus identifier."),
                col_spec("PLANT_NAME", "string", sap_ref="T001W-NAME1", ml_role="METADATA", desc="Campus or regional hub name."),
                col_spec("BUKRS", "string", fk="dim_company_code.BUKRS", sap_ref="T001W/T001K-BUKRS", allowed=["US10", "EU20", "AP30"], ml_role="HIERARCHY_KEY", desc="Parent company code (in native SAP, T001W maps to T001K-BUKRS via BWKEY)."),
                col_spec("REGION", "string", provenance="ANALYTICAL_EXTENSION", allowed=["US", "EMEA", "APAC"], ml_role="HIERARCHY_KEY", desc="Geographic reporting region."),
                col_spec("CITY", "string", sap_ref="T001W-ORT01", ml_role="METADATA", desc="Campus city and state/country."),
                col_spec("PLANT_TYPE", "string", provenance="ANALYTICAL_EXTENSION", allowed=["Commercial_HQ", "RD_Campus", "Regional_Hub"], ml_role="HIERARCHY_ATTRIBUTE", desc="Campus operational classification."),
            ],
        },
        {
            "table_name": "dim_cost_center",
            "file_name": "dim_cost_center.csv",
            "category": "DIMENSION",
            "time_span": "Static",
            "ml_pipeline_role": "Hierarchical Rollups (ban COST_SCALE)",
            "sap_inspiration": "CSKS / CSKT + Analytical Hierarchy Extensions",
            "primary_key": ["KOSTL"],
            "description": "24 J&J-aligned cost centers providing N:1 links to Plants, Profit Centers, Departments, Franchises/TAs, Functional Areas, and Segments.",
            "columns": [
                col_spec("BUKRS", "string", fk="dim_company_code.BUKRS", sap_ref="CSKS-BUKRS", allowed=["US10", "EU20", "AP30"], ml_role="HIERARCHY_KEY", desc="Owning company code."),
                col_spec("WERKS", "string", fk="dim_plant.WERKS", provenance="ANALYTICAL_EXTENSION", sap_ref="Plant Link", allowed=["US01", "US02", "US03", "BE01", "NL01", "IN01"], ml_role="HIERARCHY_KEY", desc="Assigned plant (analytical extension; standard SAP CSKS has no WERKS column)."),
                col_spec("KOSTL", "string", pk=True, sap_ref="CSKS-KOSTL", ml_role="ENTITY_KEY", desc="Cost center identifier (CC1010-CC3050)."),
                col_spec("KOSTL_NAME", "string", sap_ref="CSKT-LTEXT", ml_role="METADATA", desc="Cost center description."),
                col_spec("DEPARTMENT", "string", sap_ref="CSKS-ABTEI", ml_role="HIERARCHY_ATTRIBUTE", desc="Operational department (10 values across R&D, Commercial, Medical, IT, Finance/Legal)."),
                col_spec("KOSTL_GROUP", "string", provenance="ANALYTICAL_EXTENSION", allowed=["RND", "SALES", "ADMIN"], ml_role="HIERARCHY_ATTRIBUTE", desc="Coarse legacy cost center group."),
                col_spec("SEGMENT", "string", provenance="ANALYTICAL_EXTENSION", sap_ref="FAGL_SEGM", allowed=["Innovative_Medicine", "MedTech", "Enterprise_Corporate"], ml_role="HIERARCHY_KEY", desc="J&J top-level business segment."),
                col_spec("FRANCHISE_TA", "string", provenance="ANALYTICAL_EXTENSION", allowed=["Oncology", "Immunology", "Neuroscience", "Cardiovascular", "Surgery", "Vision", "Corporate"], ml_role="HIERARCHY_KEY", desc="Therapeutic area or MedTech franchise."),
                col_spec("REGION", "string", provenance="ANALYTICAL_EXTENSION", allowed=["US", "EMEA", "APAC"], ml_role="HIERARCHY_KEY", desc="Geographic region."),
                col_spec("RFAREA", "string", sap_ref="CSKS-FUNC_AREA", allowed=["R&D", "Commercial_SGA", "Medical_Affairs", "G&A"], ml_role="HIERARCHY_KEY", desc="Functional area for P&L classification and posting-matrix enforcement."),
                col_spec("PRCTR", "string", fk="dim_profit_center.PRCTR", sap_ref="CSKS-PRCTR", ml_role="HIERARCHY_KEY", desc="Assigned profit center (N:1 KOSTL -> PRCTR)."),
                col_spec("DEFAULT_WBS", "string", provenance="ANALYTICAL_EXTENSION", sap_ref="PRPS-POSID", ml_role="HIERARCHY_ATTRIBUTE", desc="Default clinical study WBS element ('NONE' if non-project)."),
                col_spec("COST_SCALE", "float64", provenance="GENERATOR_METADATA", ml_role="DO_NOT_USE_IN_MODEL", desc="GENERATOR METADATA ONLY - NEVER USE AS AN ML FEATURE. Synthetic relative sizing knob."),
            ],
        },
        {
            "table_name": "dim_profit_center",
            "file_name": "dim_profit_center.csv",
            "category": "DIMENSION",
            "time_span": "Static",
            "ml_pipeline_role": "Segment / TA / Region Rollups",
            "sap_inspiration": "CEPC / CEPCT",
            "primary_key": ["PRCTR"],
            "description": "Profit center hierarchy combining business segment, franchise/TA, and region.",
            "columns": [
                col_spec("PRCTR", "string", pk=True, sap_ref="CEPC-PRCTR", ml_role="HIERARCHY_KEY", desc="Profit center identifier."),
                col_spec("SEGMENT", "string", provenance="ANALYTICAL_EXTENSION", allowed=["Innovative_Medicine", "MedTech", "Enterprise_Corporate"], ml_role="HIERARCHY_KEY", desc="J&J business segment."),
                col_spec("FRANCHISE_TA", "string", provenance="ANALYTICAL_EXTENSION", allowed=["Oncology", "Immunology", "Neuroscience", "Cardiovascular", "Surgery", "Vision", "Corporate"], ml_role="HIERARCHY_KEY", desc="Therapeutic area or franchise."),
                col_spec("REGION", "string", provenance="ANALYTICAL_EXTENSION", allowed=["US", "EMEA", "APAC"], ml_role="HIERARCHY_KEY", desc="Geographic region."),
                col_spec("PRCTR_NAME", "string", sap_ref="CEPCT-LTEXT", ml_role="METADATA", desc="Profit center full description."),
            ],
        },
        {
            "table_name": "dim_gl_account",
            "file_name": "dim_gl_account.csv",
            "category": "DIMENSION",
            "time_span": "Static",
            "ml_pipeline_role": "Cost-Behavior Segmentation & Filtering",
            "sap_inspiration": "SKA1 / SKAT",
            "primary_key": ["HKONT"],
            "unique_keys": ["RACCT"],
            "description": "Chart of Accounts covering 22 OPEX GLs (610010-650040) and 4 BS offset GLs (113100, 140000, 210000, 215000).",
            "columns": [
                col_spec("HKONT", "string", pk=True, sap_ref="SKA1-SAKNR", ml_role="ENTITY_KEY", desc="6-char GL account number (ECC naming)."),
                col_spec("GL_NAME", "string", sap_ref="SKAT-TXT50", ml_role="METADATA", desc="GL account text description."),
                col_spec("ACCOUNT_TYPE", "string", provenance="ANALYTICAL_EXTENSION", allowed=["OPEX", "BS"], ml_role="FILTER_FLAG", desc="P&L OPEX vs. Balance Sheet offset classification."),
                col_spec("COST_BEHAVIOR", "string", provenance="ANALYTICAL_EXTENSION", allowed=["FIXED", "STEP_FIXED", "VARIABLE", "DISCRETIONARY", "MILESTONE", "NONE"], ml_role="MODEL_SEGMENTATION", desc="Economic cost behavior driving segmented model selection."),
                col_spec("OPEX_GROUP", "string", provenance="ANALYTICAL_EXTENSION", allowed=["WORKFORCE_CORE", "R&D_CLINICAL", "R&D_DISCOVERY", "R&D_REGULATORY", "COMMERCIAL_FIELD", "COMMERCIAL_MARKETING", "MEDICAL_AFFAIRS", "G&A_IT", "G&A_CORP", "BALANCE_SHEET"], ml_role="HIERARCHY_KEY", desc="Sub-functional OPEX rollup group."),
                col_spec("ALLOWED_RFAREAS", "string", provenance="ANALYTICAL_EXTENSION", ml_role="VALIDATION_RULE", desc="Pipe-delimited functional areas permitted to post to this GL ('ALL' for BS)."),
                col_spec("USES_PO", "bool", provenance="ANALYTICAL_EXTENSION", allowed=[True, False], ml_role="FEATURE_FLAG", desc="True if external vendor spend flows through Purchase Orders."),
                col_spec("RACCT", "string", uk=True, sap_ref="ACDOCA-RACCT", ml_role="ENTITY_KEY", desc="S/4HANA Universal Journal GL alias (unique 1:1 mirror of HKONT)."),
            ],
        },
        {
            "table_name": "dim_fx_rates",
            "file_name": "dim_fx_rates.csv",
            "category": "DIMENSION",
            "time_span": "FY22-FY26",
            "ml_pipeline_role": "Reported vs. Constant FX Translation",
            "sap_inspiration": "Simplified analytical abstraction of TCURR/TCURF/TCURX",
            "primary_key": ["GJAHR", "MONAT", "FROM_CURR", "RATE_TYPE"],
            "description": "Direct-quote monthly exchange rates to USD across ACTUAL_AVG (FY22-25), BUDGET_PLAN (constant per year, locked Oct yr-1), and FORWARD_CURVE (FY22-26).",
            "columns": [
                col_spec("GJAHR", "int64", pk=True, sap_ref="GJAHR", ml_role="TIME_INDEX", desc="Fiscal year (2022-2026)."),
                col_spec("MONAT", "int64", pk=True, sap_ref="MONAT", allowed=list(range(1, 13)), ml_role="TIME_INDEX", desc="Fiscal period (1-12)."),
                col_spec("FROM_CURR", "string", pk=True, sap_ref="TCURR-FCURR", allowed=["USD", "EUR", "INR"], ml_role="CURRENCY_KEY", desc="Source local currency."),
                col_spec("TO_CURR", "string", sap_ref="TCURR-TCURR", allowed=["USD"], ml_role="CURRENCY_KEY", desc="Group reporting currency (always 'USD')."),
                col_spec("RATE_TYPE", "string", pk=True, provenance="ANALYTICAL_EXTENSION", sap_ref="TCURR-KURST", allowed=["ACTUAL_AVG", "BUDGET_PLAN", "FORWARD_CURVE"], ml_role="FILTER_FLAG", desc="Exchange rate curve type."),
                col_spec("FX_RATE", "float64", sap_ref="TCURR-UKURS", ml_role="FX_DRIVER", desc="Direct quote multiplier: USD per 1 unit of FROM_CURR."),
            ],
        },
        {
            "table_name": "fact_opex_drivers",
            "file_name": "fact_opex_drivers.csv",
            "category": "FACT_DRIVERS",
            "time_span": "FY22-FY25",
            "ml_pipeline_role": "Exogenous Features (Strictly Lagged <= cutoff)",
            "sap_inspiration": "Workday HR + CTMS + Commercial CRM + Macro Drivers",
            "primary_key": ["GJAHR", "MONAT", "KOSTL"],
            "description": "Realized monthly operational, clinical, commercial, and macro drivers for FY2022-FY2025. Only use lags <= forecast cutoff when backtesting.",
            "columns": driver_columns_schema(is_scenario=False),
        },
        {
            "table_name": "plan_drivers",
            "file_name": "plan_drivers.csv",
            "category": "FACT_PLAN_DRIVERS",
            "time_span": "FY22-FY26",
            "ml_pipeline_role": "Safe Future Plan Inputs & Variance Bridge",
            "sap_inspiration": "Annual Business Plan (BP) Driver Assumptions",
            "primary_key": ["GJAHR", "MONAT", "KOSTL"],
            "description": "Point-in-time, leak-free Annual Business Plan (BP) driver targets for FY2022-FY2026 locked in Oct yr-1 (bridged across Nov-Dec).",
            "columns": [
                col_spec("GJAHR", "int64", pk=True, sap_ref="GJAHR", ml_role="TIME_INDEX", desc="Fiscal year (2022-2026)."),
                col_spec("MONAT", "int64", pk=True, sap_ref="MONAT", allowed=list(range(1, 13)), ml_role="TIME_INDEX", desc="Fiscal period (1-12)."),
                col_spec("BUKRS", "string", fk="dim_company_code.BUKRS", sap_ref="BUKRS", allowed=["US10", "EU20", "AP30"], ml_role="HIERARCHY_KEY", desc="Company code."),
                col_spec("WERKS", "string", fk="dim_plant.WERKS", provenance="ANALYTICAL_EXTENSION", sap_ref="Plant Link", allowed=["US01", "US02", "US03", "BE01", "NL01", "IN01"], ml_role="HIERARCHY_KEY", desc="Plant / campus code."),
                col_spec("KOSTL", "string", pk=True, fk="dim_cost_center.KOSTL", sap_ref="CSKS-KOSTL", ml_role="ENTITY_KEY", desc="Cost center identifier."),
                col_spec("WBS_ELEMENT", "string", sap_ref="PRPS-POSID", ml_role="HIERARCHY_KEY", desc="Clinical trial WBS element."),
                col_spec("PLAN_FTE", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="SAFE_FUTURE_PLAN_FEATURE", desc="BP target FTE headcount projected leak-free from Oct yr-1 anchor (with Nov-Dec bridge)."),
                col_spec("PLAN_ACTIVE_SITES", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="SAFE_FUTURE_PLAN_FEATURE", desc="BP target active clinical sites projected from Oct yr-1 anchor."),
                col_spec("PLAN_ENROLLED_PATIENTS", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="SAFE_FUTURE_PLAN_FEATURE", desc="BP target enrolled clinical patients projected from Oct yr-1 anchor."),
                col_spec("PLAN_HCP_EVENTS", "int64", provenance="ANALYTICAL_EXTENSION", ml_role="SAFE_FUTURE_PLAN_FEATURE", desc="BP target commercial HCP events count."),
                col_spec("PLAN_MILESTONE_EVENT", "string", provenance="ANALYTICAL_EXTENSION", allowed=["NONE", "FPI", "DB_LOCK", "FDA_FILING"], ml_role="SAFE_FUTURE_PLAN_FEATURE", desc="BP locked milestone schedule (decoupled from realized milestone slips/accelerations)."),
                col_spec("PLAN_CPI_INDEX", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="SAFE_FUTURE_PLAN_FEATURE", desc="BP locked CPI assumption (3.0%/yr from Oct yr-1)."),
                col_spec("PLAN_WAGE_INDEX", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="SAFE_FUTURE_PLAN_FEATURE", desc="BP locked wage index assumption (3.6% April merit step from Oct yr-1)."),
            ],
        },
        {
            "table_name": "fact_po_commitments",
            "file_name": "fact_po_commitments.csv",
            "category": "FACT_LEADING_INDICATOR",
            "time_span": "FY21-FY25 (<=2025-12-31)",
            "ml_pipeline_role": "Leading Feature (Filter Single SNAPSHOT_DATE)",
            "sap_inspiration": "COOI / EKPO Point-in-Time Commitment Snapshots",
            "primary_key": ["EBELN", "EBELP", "SNAPSHOT_DATE"],
            "description": "Monthly point-in-time Purchase Order commitment snapshots (1-2 month lead before GR/invoice), hard-capped <= 2025-12-31. Filter to a SINGLE SNAPSHOT_DATE per forecast origin before summing OPEN_COMMITMENT_USD.",
            "columns": [
                col_spec("EBELN", "string", pk=True, sap_ref="EKPO-EBELN", ml_role="DOCUMENT_KEY", desc="Purchase order number."),
                col_spec("EBELP", "string", pk=True, sap_ref="EKPO-EBELP", allowed=["00010", "00020"], ml_role="DOCUMENT_KEY", desc="5-char PO line item."),
                col_spec("BUKRS", "string", fk="dim_company_code.BUKRS", sap_ref="EKKO-BUKRS", allowed=["US10", "EU20", "AP30"], ml_role="HIERARCHY_KEY", desc="Company code."),
                col_spec("WERKS", "string", fk="dim_plant.WERKS", sap_ref="EKPO-WERKS", allowed=["US01", "US02", "US03", "BE01", "NL01", "IN01"], ml_role="HIERARCHY_KEY", desc="Plant / campus code."),
                col_spec("KOSTL", "string", fk="dim_cost_center.KOSTL", sap_ref="EKKN-KOSTL", ml_role="ENTITY_KEY", desc="Assigned cost center."),
                col_spec("HKONT", "string", fk="dim_gl_account.HKONT", sap_ref="EKKN-SAKTO", ml_role="ENTITY_KEY", desc="Assigned OPEX GL account (USES_PO == True)."),
                col_spec("WBS_ELEMENT", "string", sap_ref="EKKN-PS_PSP_PNR", ml_role="HIERARCHY_KEY", desc="Assigned WBS element."),
                col_spec("LIFNR", "string", sap_ref="EKKO-LIFNR", ml_role="AUDIT_ATTRIBUTE", desc="Vendor account ID."),
                col_spec("PO_CREATE_DATE", "date", sap_ref="EKKO-AEDAT", ml_role="AUDIT_CUTOFF", desc="Date PO was created (hard-capped <= 2025-12-31)."),
                col_spec("EXPECTED_GR_MONTH", "string", provenance="ANALYTICAL_EXTENSION", sap_ref="EKET-EINDT", ml_role="TARGET_HORIZON_KEY", desc="Expected goods receipt month ('YYYY-MM'; includes 2026-01/2026-02 open pipeline)."),
                col_spec("SNAPSHOT_DATE", "date", pk=True, provenance="ANALYTICAL_EXTENSION", ml_role="PIT_CUTOFF_KEY", desc="Month-end point-in-time snapshot date (hard-capped <= 2025-12-31)."),
                col_spec("ORIGINAL_COMMITMENT_USD", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="LEADING_INDICATOR", desc="Initial PO line commitment value in USD."),
                col_spec("GR_AMOUNT_USD", "float64", provenance="ANALYTICAL_EXTENSION", sap_ref="EKBE", ml_role="LEADING_INDICATOR", desc="Cumulative goods receipt amount in USD as of SNAPSHOT_DATE."),
                col_spec("INVOICED_AMOUNT_USD", "float64", provenance="ANALYTICAL_EXTENSION", sap_ref="EKBE", ml_role="LEADING_INDICATOR", desc="Cumulative invoiced amount in USD as of SNAPSHOT_DATE (<= GR_AMOUNT_USD)."),
                col_spec("OPEN_COMMITMENT_USD", "float64", sap_ref="COOI-WKGBTR", ml_role="LEADING_INDICATOR", desc="Remaining open commitment USD (= ORIGINAL_COMMITMENT_USD - GR_AMOUNT_USD)."),
                col_spec("PO_STATUS", "string", provenance="ANALYTICAL_EXTENSION", allowed=["OPEN", "PARTIALLY_RECEIVED", "CLOSED"], ml_role="FILTER_FLAG", desc="PO lifecycle state as of SNAPSHOT_DATE."),
            ],
        },
        {
            "table_name": "acdoca_actuals",
            "file_name": "acdoca_actuals.csv",
            "category": "FACT_TRANSACTIONAL_ECC",
            "time_span": "FY22-FY25",
            "ml_pipeline_role": "Audit / Duplicate & Accrual QA (BKPF/BSEG)",
            "sap_inspiration": "BKPF / BSEG (unsigned amounts qualified by SHKZG)",
            "primary_key": ["BUKRS", "GJAHR", "BELNR", "BUZEI"],
            "description": "Chronologically sorted double-entry GL posting lines (FY2022-FY2025, periods 1-16) in classic ECC field names with unsigned amounts qualified by SHKZG.",
            "columns": [
                col_spec("BUKRS", "string", pk=True, fk="dim_company_code.BUKRS", sap_ref="BKPF-BUKRS", allowed=["US10", "EU20", "AP30"], ml_role="HIERARCHY_KEY", desc="Company code."),
                col_spec("WERKS", "string", fk="dim_plant.WERKS", sap_ref="BSEG-WERKS", allowed=["US01", "US02", "US03", "BE01", "NL01", "IN01"], ml_role="HIERARCHY_KEY", desc="Plant / campus code."),
                col_spec("GJAHR", "int64", pk=True, sap_ref="BKPF-GJAHR", allowed=[2022, 2023, 2024, 2025], ml_role="TIME_INDEX", desc="Fiscal year (2022-2025)."),
                col_spec("MONAT", "int64", sap_ref="BKPF-MONAT", allowed=list(range(1, 17)), ml_role="TIME_INDEX", desc="Posting period (1-12 regular; 13-16 year-end special audit periods)."),
                col_spec("BELNR", "string", pk=True, sap_ref="BKPF-BELNR", ml_role="DOCUMENT_KEY", desc="10-char chronological document number per (BUKRS, GJAHR)."),
                col_spec("BLDAT", "date", sap_ref="BKPF-BLDAT", ml_role="AUDIT_ATTRIBUTE", desc="Document / invoice date (BLDAT <= BUDAT)."),
                col_spec("BUDAT", "date", sap_ref="BKPF-BUDAT", ml_role="AUDIT_ATTRIBUTE", desc="Posting date (YYYY-12-31 for periods 13-16)."),
                col_spec("CPUDT", "date", sap_ref="BKPF-CPUDT", ml_role="PIT_CUTOFF_KEY", desc="System entry date (BUDAT <= CPUDT); filter CPUDT <= SNAPSHOT_DATE for PIT backtesting."),
                col_spec("BLART", "string", sap_ref="BKPF-BLART", allowed=["KR", "PR", "SA", "ZA", "AB"], ml_role="AUDIT_ATTRIBUTE", desc="Document type: KR=Vendor Invoice, PR=Payroll (210000), SA=Regular/Prepaid/Audit, ZA=Month-End Accrual (215000), AB=Reversal."),
                col_spec("USNAM", "string", sap_ref="BKPF-USNAM", ml_role="AUDIT_ATTRIBUTE", desc="User or batch ID ('USER001'..'USER039', 'BATCH_PAYROLL', 'BATCH_REV', 'BATCH_PREPAID', 'BATCH_AUDIT')."),
                col_spec("PRCTR", "string", fk="dim_profit_center.PRCTR", sap_ref="BSEG-PRCTR", ml_role="HIERARCHY_KEY", desc="Profit center."),
                col_spec("WBS_ELEMENT", "string", sap_ref="BSEG-PROJK", ml_role="HIERARCHY_KEY", desc="Clinical trial WBS element ('NONE' if non-project)."),
                col_spec("LIFNR", "string", sap_ref="BSEG-LIFNR", ml_role="AUDIT_ATTRIBUTE", desc="Vendor ID, 'INTERNAL_PAYROLL', or 'YEAR_END_AUDIT'."),
                col_spec("EBELN", "string", fk="fact_po_commitments.EBELN (optional)", nullable=True, sap_ref="BSEG-EBELN", ml_role="AUDIT_ATTRIBUTE", desc="PO reference (PO*) on PO-backed KR invoices and 100% of ZA/AB accrual pairs; blank string '' on non-PO, 650010 prepaid splits & GAAP Special Items."),
                col_spec("WAERS", "string", sap_ref="BKPF-WAERS", allowed=["USD", "EUR", "INR"], ml_role="CURRENCY_KEY", desc="Local currency key."),
                col_spec("DMBTR", "float64", sap_ref="BSEG-DMBTR", ml_role="AMOUNT_LOCAL_UNSIGNED", desc="Unsigned amount in local currency (qualify by SHKZG: S=+, H=-)."),
                col_spec("KSL", "float64", provenance="ANALYTICAL_EXTENSION", sap_ref="ACDOCA-KSL, S/4HANA carry-over", ml_role="TARGET_REPORTED_UNSIGNED", desc="Unsigned Reported Group USD amount translated at ACTUAL_AVG FX (S/4HANA carry-over)."),
                col_spec("KSL_BUDGET_FX", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="TARGET_OPERATIONAL_UNSIGNED", desc="Unsigned Constant-Currency USD amount translated at annual BUDGET_PLAN FX."),
                col_spec("STBLG", "string", nullable=True, sap_ref="BKPF-STBLG", ml_role="AUDIT_ATTRIBUTE", desc="Reversal document BELNR (bidirectional ZA<->AB; '' otherwise, incl. legitimate Dec 2025 open accruals)."),
                col_spec("STJAH", "int64", sap_ref="BKPF-STJAH", ml_role="AUDIT_ATTRIBUTE", desc="Fiscal year of reversal document STBLG (0 when STBLG == '')."),
                col_spec("SPECIAL_ITEM_FLAG", "string", provenance="ANALYTICAL_EXTENSION", allowed=["NONE", "ACQUIRED_IPRD", "RESTRUCTURING", "MA_INTEGRATION", "LITIGATION"], ml_role="FILTER_FLAG", desc="GAAP vs. Adjusted Operational non-recurring item classification."),
                col_spec("SERVICE_START_DATE", "date", provenance="ANALYTICAL_EXTENSION", ml_role="AUDIT_ATTRIBUTE", desc="Service start: month start regular, calendar quarter start (~90d) on 140000 prepaids, or BLDAT default on ZA/AB/Special/P13-16."),
                col_spec("SERVICE_END_DATE", "date", provenance="ANALYTICAL_EXTENSION", ml_role="AUDIT_ATTRIBUTE", desc="Service end: month end regular, calendar quarter end on 140000 prepaids, or BUDAT default on ZA/AB/Special/P13-16."),
                col_spec("VERSION", "int64", sap_ref="ACDOCA-VERSN", allowed=[0], ml_role="FILTER_FLAG", desc="SAP version (0 = Actuals)."),
                col_spec("BUZEI", "int64", pk=True, sap_ref="BSEG-BUZEI", allowed=[1, 2], ml_role="DOCUMENT_KEY", desc="Line item number (1 = OPEX line, 2 = Balance Sheet offset line)."),
                col_spec("HKONT", "string", fk="dim_gl_account.HKONT", sap_ref="BSEG-HKONT", ml_role="ENTITY_KEY", desc="GL account number (610010-650040 on BUZEI=1; BS offsets on BUZEI=2)."),
                col_spec("KOSTL", "string", fk="dim_cost_center.KOSTL", nullable=True, sap_ref="BSEG-KOSTL", ml_role="ENTITY_KEY", desc="Cost center ID (populated on OPEX lines; '' on BS offset lines)."),
                col_spec("RFAREA", "string", nullable=True, sap_ref="BSEG-FKBER", allowed=["R&D", "Commercial_SGA", "Medical_Affairs", "G&A", ""], ml_role="HIERARCHY_KEY", desc="Functional area on OPEX lines; '' on BS offset lines."),
                col_spec("SHKZG", "string", sap_ref="BSEG-SHKZG", allowed=["S", "H"], ml_role="ACCOUNTING_SIGN", desc="Debit/Credit indicator ('S'=Debit, 'H'=Credit)."),
                col_spec("BSCHL", "string", sap_ref="BSEG-BSCHL", allowed=["40", "50"], ml_role="ACCOUNTING_SIGN", desc="Posting key ('40'=Debit, '50'=Credit)."),
            ],
        },
        {
            "table_name": "acdoca_universal_journal",
            "file_name": "acdoca_universal_journal.csv",
            "category": "FACT_TRANSACTIONAL_S4",
            "time_span": "FY22-FY25",
            "ml_pipeline_role": "Primary Target Source (Signed HSL/KSL)",
            "sap_inspiration": "S/4HANA ACDOCA (native PK: RCLNT, RLDNR, RBUKRS, GJAHR, BELNR, DOCLN)",
            "primary_key": ["RBUKRS", "RYEAR", "BELNR", "DOCLN"],
            "native_s4hana_primary_key": ["RCLNT", "RLDNR", "RBUKRS", "GJAHR", "BELNR", "DOCLN"],
            "description": "Exact row-for-row S/4HANA Universal Journal projection of acdoca_actuals with signed HSL, KSL, and KSL_BUDGET_FX so SUM() yields net balances directly.",
            "columns": [
                col_spec("RBUKRS", "string", pk=True, fk="dim_company_code.BUKRS", sap_ref="ACDOCA-RBUKRS", allowed=["US10", "EU20", "AP30"], ml_role="HIERARCHY_KEY", ecc_equiv="BUKRS", desc="Company code."),
                col_spec("WERKS", "string", fk="dim_plant.WERKS", sap_ref="ACDOCA-WERKS", allowed=["US01", "US02", "US03", "BE01", "NL01", "IN01"], ml_role="HIERARCHY_KEY", ecc_equiv="WERKS", desc="Plant / campus code."),
                col_spec("GJAHR", "int64", sap_ref="ACDOCA-GJAHR", allowed=[2022, 2023, 2024, 2025], ml_role="TIME_INDEX", ecc_equiv="GJAHR", desc="Fiscal year key (identical to RYEAR)."),
                col_spec("RYEAR", "int64", pk=True, sap_ref="ACDOCA-RYEAR", allowed=[2022, 2023, 2024, 2025], ml_role="TIME_INDEX", ecc_equiv="GJAHR", desc="General Ledger fiscal year."),
                col_spec("POPER", "int64", sap_ref="ACDOCA-POPER", allowed=list(range(1, 17)), ml_role="TIME_INDEX", ecc_equiv="MONAT", desc="Posting period (1-12 regular; 13-16 special audit; integer for numeric filtering)."),
                col_spec("BELNR", "string", pk=True, sap_ref="ACDOCA-BELNR", ml_role="DOCUMENT_KEY", ecc_equiv="BELNR", desc="10-char chronological document number per (RBUKRS, RYEAR)."),
                col_spec("DOCLN", "string", pk=True, sap_ref="ACDOCA-DOCLN", allowed=["000001", "000002"], ml_role="DOCUMENT_KEY", ecc_equiv="BUZEI", desc="Zero-padded 6-char line item."),
                col_spec("BLDAT", "date", sap_ref="ACDOCA-BLDAT", ml_role="AUDIT_ATTRIBUTE", ecc_equiv="BLDAT", desc="Document / invoice date."),
                col_spec("BUDAT", "date", sap_ref="ACDOCA-BUDAT", ml_role="AUDIT_ATTRIBUTE", ecc_equiv="BUDAT", desc="Posting date."),
                col_spec("CPUDT", "date", sap_ref="BKPF-CPUDT", ml_role="PIT_CUTOFF_KEY", ecc_equiv="CPUDT", desc="System entry date (filter CPUDT <= SNAPSHOT_DATE for PIT backtesting)."),
                col_spec("BLART", "string", sap_ref="ACDOCA-BLART", allowed=["KR", "PR", "SA", "ZA", "AB"], ml_role="AUDIT_ATTRIBUTE", ecc_equiv="BLART", desc="Document type."),
                col_spec("USNAM", "string", sap_ref="ACDOCA-USNAM", ml_role="AUDIT_ATTRIBUTE", ecc_equiv="USNAM", desc="Posting user or batch job ID."),
                col_spec("RACCT", "string", fk="dim_gl_account.RACCT", sap_ref="ACDOCA-RACCT", ml_role="ENTITY_KEY", ecc_equiv="HKONT", desc="GL account number."),
                col_spec("RCNTR", "string", fk="dim_cost_center.KOSTL", nullable=True, sap_ref="ACDOCA-RCNTR", ml_role="ENTITY_KEY", ecc_equiv="KOSTL", desc="Cost center ID ('' on BS lines)."),
                col_spec("PRCTR", "string", fk="dim_profit_center.PRCTR", sap_ref="ACDOCA-PRCTR", ml_role="HIERARCHY_KEY", ecc_equiv="PRCTR", desc="Profit center."),
                col_spec("RFAREA", "string", nullable=True, sap_ref="ACDOCA-RFAREA", allowed=["R&D", "Commercial_SGA", "Medical_Affairs", "G&A", ""], ml_role="HIERARCHY_KEY", ecc_equiv="RFAREA", desc="Functional area ('' on BS lines so GROUP BY RFAREA never nets to 0)."),
                col_spec("PS_PSP_PNR", "string", sap_ref="ACDOCA-PS_PSP_PNR", ml_role="HIERARCHY_KEY", ecc_equiv="WBS_ELEMENT", desc="Clinical study WBS element."),
                col_spec("LIFNR", "string", sap_ref="ACDOCA-LIFNR", ml_role="AUDIT_ATTRIBUTE", ecc_equiv="LIFNR", desc="Vendor ID."),
                col_spec("EBELN", "string", fk="fact_po_commitments.EBELN (optional)", nullable=True, sap_ref="ACDOCA-EBELN", ml_role="AUDIT_ATTRIBUTE", ecc_equiv="EBELN", desc="PO reference (PO* on PO-backed KR & 100% of ZA/AB docs; '' on non-PO, 650010 prepaids & Special Items)."),
                col_spec("RWCUR", "string", sap_ref="ACDOCA-RWCUR", allowed=["USD", "EUR", "INR"], ml_role="CURRENCY_KEY", ecc_equiv="WAERS", desc="Local currency key."),
                col_spec("DRCRK", "string", sap_ref="ACDOCA-DRCRK", allowed=["S", "H"], ml_role="ACCOUNTING_SIGN", ecc_equiv="SHKZG", desc="Debit/Credit indicator."),
                col_spec("BSCHL", "string", sap_ref="ACDOCA-BSCHL", allowed=["40", "50"], ml_role="ACCOUNTING_SIGN", ecc_equiv="BSCHL", desc="Posting key."),
                col_spec("HSL", "float64", sap_ref="ACDOCA-HSL", ml_role="AMOUNT_LOCAL_SIGNED", ecc_equiv="Signed DMBTR", desc="Signed local currency amount (+ Debit, - Credit; sums to 0.00 per doc)."),
                col_spec("KSL", "float64", sap_ref="ACDOCA-KSL", ml_role="TARGET_REPORTED_SIGNED", ecc_equiv="Signed KSL", desc="Signed Reported Group USD amount translated at ACTUAL_AVG FX."),
                col_spec("KSL_BUDGET_FX", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="TARGET_OPERATIONAL_SIGNED", ecc_equiv="Signed KSL_BUDGET_FX", desc="Signed Operational Constant-Currency USD translated at BUDGET_PLAN FX (primary ML target)."),
                col_spec("STBLG", "string", nullable=True, sap_ref="BKPF-STBLG", ml_role="AUDIT_ATTRIBUTE", ecc_equiv="STBLG", desc="Reversal document BELNR ('' if unreversed, incl. Dec 2025 open accruals)."),
                col_spec("STJAH", "int64", sap_ref="BKPF-STJAH", ml_role="AUDIT_ATTRIBUTE", ecc_equiv="STJAH", desc="Reversal fiscal year."),
                col_spec("SPECIAL_ITEM_FLAG", "string", provenance="ANALYTICAL_EXTENSION", allowed=["NONE", "ACQUIRED_IPRD", "RESTRUCTURING", "MA_INTEGRATION", "LITIGATION"], ml_role="FILTER_FLAG", ecc_equiv="SPECIAL_ITEM_FLAG", desc="GAAP vs. Adjusted Operational tag (filter == 'NONE' for recurring OPEX)."),
                col_spec("SERVICE_START_DATE", "date", provenance="ANALYTICAL_EXTENSION", ml_role="AUDIT_ATTRIBUTE", ecc_equiv="SERVICE_START_DATE", desc="Service start date."),
                col_spec("SERVICE_END_DATE", "date", provenance="ANALYTICAL_EXTENSION", ml_role="AUDIT_ATTRIBUTE", ecc_equiv="SERVICE_END_DATE", desc="Service end date."),
                col_spec("RLDNR", "string", sap_ref="ACDOCA-RLDNR", allowed=["0L"], ml_role="METADATA", ecc_equiv="(S/4HANA only)", desc="SAP General Ledger ID ('0L' = Leading Ledger)."),
                col_spec("VERSION", "int64", sap_ref="ACDOCA-VERSN", allowed=[0], ml_role="FILTER_FLAG", ecc_equiv="VERSION", desc="SAP version (0 = Actuals)."),
            ],
        },
        {
            "table_name": "plan_versions",
            "file_name": "plan_versions.csv",
            "category": "FACT_PLAN_AND_BENCHMARK",
            "time_span": "FY22-FY26",
            "ml_pipeline_role": "VERSION=1 (BP) safe baseline; VERSION=2 Benchmark Only",
            "sap_inspiration": "ACDOCP (SAP S/4HANA Planning Table)",
            "primary_key": ["GJAHR", "MONAT", "KOSTL", "HKONT", "CYCLE_NAME"],
            "description": "Point-in-time, leak-free J&J planning snapshots (VERSION=1 BP FY22-26; VERSION=2 Q1_RF/JU/Q3_RF/NU FY22-25). Keep VERSION=2 strictly for evaluation; PLAN_FTE/PATIENTS/HCP on VERSION=2 carry static annual BP baselines.",
            "columns": [
                col_spec("GJAHR", "int64", pk=True, sap_ref="ACDOCP-RYEAR", ml_role="TIME_INDEX", desc="Fiscal year (2022-2026 for BP; 2022-2025 for RF cycles)."),
                col_spec("MONAT", "int64", pk=True, sap_ref="ACDOCP-POPER", allowed=list(range(1, 13)), ml_role="TIME_INDEX", desc="Fiscal period (1-12)."),
                col_spec("BUKRS", "string", fk="dim_company_code.BUKRS", sap_ref="ACDOCP-RBUKRS", allowed=["US10", "EU20", "AP30"], ml_role="HIERARCHY_KEY", desc="Company code."),
                col_spec("WERKS", "string", fk="dim_plant.WERKS", sap_ref="ACDOCP-WERKS", allowed=["US01", "US02", "US03", "BE01", "NL01", "IN01"], ml_role="HIERARCHY_KEY", desc="Plant / campus code."),
                col_spec("KOSTL", "string", pk=True, fk="dim_cost_center.KOSTL", sap_ref="ACDOCP-RCNTR", ml_role="ENTITY_KEY", desc="Cost center ID."),
                col_spec("HKONT", "string", pk=True, fk="dim_gl_account.HKONT", sap_ref="ACDOCP-RACCT", ml_role="ENTITY_KEY", desc="OPEX GL account."),
                col_spec("VERSION", "int64", sap_ref="ACDOCP-CATEGORY", allowed=[1, 2], ml_role="FILTER_FLAG", desc="1 = Annual Business Plan (BP), 2 = Human Rolling Forecast (benchmark only)."),
                col_spec("CYCLE_NAME", "string", pk=True, provenance="ANALYTICAL_EXTENSION", allowed=["BP", "Q1_RF", "JU", "Q3_RF", "NU"], ml_role="BENCHMARK_KEY", desc="J&J planning cycle (BP=0+12, Q1_RF=3+9, JU=5+7, Q3_RF=9+3, NU=10+2)."),
                col_spec("SNAPSHOT_DATE", "date", provenance="ANALYTICAL_EXTENSION", ml_role="PIT_CUTOFF_KEY", desc="Hard point-in-time cutoff date for the plan snapshot."),
                col_spec("WAERS", "string", sap_ref="ACDOCP-RWCUR", allowed=["USD", "EUR", "INR"], ml_role="CURRENCY_KEY", desc="Local currency key."),
                col_spec("PLAN_FX_RATE", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="FX_DRIVER", desc="Constant annual BUDGET_PLAN FX rate locked Oct yr-1."),
                col_spec("DMBTR_PLAN", "float64", sap_ref="ACDOCP-HSL", ml_role="PLAN_OR_BENCHMARK", desc="Planned/forecasted local currency spend (closed RF months copy realized clean HSL; 0.0 on non-milestone months for 620030)."),
                col_spec("KSL_BUDGET", "float64", sap_ref="ACDOCP-KSL", ml_role="PLAN_OR_BENCHMARK", desc="Planned/forecasted Constant-Currency USD spend (= DMBTR_PLAN * PLAN_FX_RATE)."),
                col_spec("PLAN_FTE", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="SAFE_FUTURE_PLAN_FEATURE", desc="Static annual BP FTE baseline (carried unchanged on VERSION=2 rows)."),
                col_spec("PLAN_ENROLLED_PATIENTS", "float64", provenance="ANALYTICAL_EXTENSION", ml_role="SAFE_FUTURE_PLAN_FEATURE", desc="Static annual BP enrolled patients baseline (carried unchanged on VERSION=2 rows)."),
                col_spec("PLAN_HCP_EVENTS", "int64", provenance="ANALYTICAL_EXTENSION", ml_role="SAFE_FUTURE_PLAN_FEATURE", desc="Static annual BP HCP events baseline (carried unchanged on VERSION=2 rows)."),
            ],
        },
        {
            "table_name": "scenario_assumptions",
            "file_name": "scenario_assumptions.csv",
            "category": "FACT_SCENARIOS",
            "time_span": "FY26 (2025-12-31)",
            "ml_pipeline_role": "Live FY26 What-If Model Scoring Input",
            "sap_inspiration": "FP&A Driver-Based What-If Simulation Input Table",
            "primary_key": ["SCENARIO_ID", "GJAHR", "MONAT", "KOSTL"],
            "description": "Deterministic FY2026 driver and FX perturbations anchored strictly at 2025-12-31 across 5 scenarios with 100% feature parity to fact_opex_drivers.",
            "columns": (
                [
                    col_spec("SCENARIO_ID", "string", pk=True, provenance="ANALYTICAL_EXTENSION", allowed=["BASE", "HIRING_FREEZE_Q2", "PHASE3_DELAY_1Q", "USD_STRENGTHEN_10PCT", "LAUNCH_ACCEL"], ml_role="SCENARIO_KEY", desc="Executive What-If scenario identifier."),
                    col_spec("SNAPSHOT_DATE", "date", provenance="ANALYTICAL_EXTENSION", ml_role="AUDIT_CUTOFF", desc="Planning-date anchor ('2025-12-31')."),
                ]
                + driver_columns_schema(is_scenario=True)
                + [
                    col_spec("FX_EUR_USD", "float64", provenance="ANALYTICAL_EXTENSION", sap_ref="TCURR_ABSTRACT", ml_role="FX_DRIVER", desc="Scenario-specific EUR->USD forward FX rate (0.90x under USD_STRENGTHEN_10PCT)."),
                    col_spec("FX_INR_USD", "float64", provenance="ANALYTICAL_EXTENSION", sap_ref="TCURR_ABSTRACT", ml_role="FX_DRIVER", desc="Scenario-specific INR->USD forward FX rate (0.90x under USD_STRENGTHEN_10PCT)."),
                ]
            ),
        },
        {
            "table_name": "eval_anomaly_labels",
            "file_name": "eval_anomaly_labels.csv",
            "category": "EVALUATION_ONLY",
            "time_span": "FY22-FY25",
            "ml_pipeline_role": "STRICTLY EVALUATION ONLY (Never Join to Train)",
            "sap_inspiration": "Out-of-Band Audit Ground Truth",
            "primary_key": ["BUKRS", "GJAHR", "BELNR", "BUZEI"],
            "description": "Physically isolated ground-truth anomaly labels across 4 finance anomaly classes for Precision/Recall evaluation only.",
            "columns": [
                col_spec("BUKRS", "string", pk=True, fk="acdoca_actuals.BUKRS", sap_ref="BKPF-BUKRS", allowed=["US10", "EU20", "AP30"], ml_role="EVAL_KEY", desc="Company code of anomalous document."),
                col_spec("GJAHR", "int64", pk=True, fk="acdoca_actuals.GJAHR", sap_ref="BKPF-GJAHR", ml_role="EVAL_KEY", desc="Fiscal year of anomalous document."),
                col_spec("MONAT", "int64", sap_ref="BKPF-MONAT", ml_role="EVAL_METADATA", desc="Posting period (1-12)."),
                col_spec("BELNR", "string", pk=True, fk="acdoca_actuals.BELNR", sap_ref="BKPF-BELNR", ml_role="EVAL_KEY", desc="Chronologically remapped BELNR matching ledger tables."),
                col_spec("BUZEI", "int64", pk=True, fk="acdoca_actuals.BUZEI", sap_ref="BSEG-BUZEI", allowed=[1], ml_role="EVAL_KEY", desc="Line item number (always 1 = OPEX line)."),
                col_spec("KOSTL", "string", fk="dim_cost_center.KOSTL", sap_ref="BSEG-KOSTL", ml_role="EVAL_METADATA", desc="Impacted cost center."),
                col_spec("HKONT", "string", fk="dim_gl_account.HKONT", sap_ref="BSEG-HKONT", ml_role="EVAL_METADATA", desc="Impacted OPEX GL account."),
                col_spec("ANOMALY_TAG", "string", provenance="EVALUATION_LABEL", allowed=["DUPLICATE_INVOICE", "UNREVERSED_ACCRUAL", "Q4_BUDGET_FLUSH", "TRIAL_PAUSE_DROP"], ml_role="EVAL_TARGET_ONLY", desc="Ground-truth finance anomaly class."),
                col_spec("GRANULARITY", "string", provenance="EVALUATION_LABEL", allowed=["DOCUMENT", "CELL_MONTH"], ml_role="EVAL_METADATA", desc="Detection granularity: DOCUMENT vs. CELL_MONTH."),
                col_spec("ESTIMATED_IMPACT_USD", "float64", provenance="EVALUATION_LABEL", ml_role="EVAL_METADATA", desc="Signed USD P&L distortion (- for TRIAL_PAUSE_DROP; + for overstatements)."),
                col_spec("DESCRIPTION", "string", provenance="EVALUATION_LABEL", ml_role="EVAL_METADATA", desc="Human-readable audit description (identifies same-month vs. cross-month duplicates)."),
            ],
        },
    ]

    # Populate REAL row counts, examples, observed values, and dataset metrics from actual CSVs
    loaded: Dict[str, pd.DataFrame] = {}
    for tbl in table_defs:
        path = os.path.join(data_dir, tbl["file_name"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing CSV table: {path}. Run generate_dataset.py first.")
        df = load_pharma_table(tbl["file_name"], data_dir=data_dir)
        loaded[tbl["table_name"]] = df

        expected_cols = [c["column_name"] for c in tbl["columns"]]
        actual_cols = list(df.columns)
        if expected_cols != actual_cols:
            raise ValueError(f"Schema mismatch in {tbl['file_name']}: expected={expected_cols}, actual={actual_cols}")

        tbl["row_count"] = int(len(df))
        tbl["column_count"] = int(len(actual_cols))
        col_map = get_table_column_map(tbl["table_name"], columns=expected_cols)
        for col in tbl["columns"]:
            col["readable_column"] = col_map[col["column_name"]]
            s = df[col["column_name"]]
            has_null = bool(s.isna().any() or (s == "").any())
            if has_null != col["nullable"]:
                raise ValueError(
                    f"Nullability mismatch in {tbl['file_name']}.{col['column_name']}: "
                    f"observed_nullable={has_null}, expected={col['nullable']}"
                )
            col["example"] = sample_real_example(s)
            obs = observed_unique_values(s)
            if obs is not None:
                col["observed_values"] = obs
            if col["allowed_values"] is not None:
                if obs is None:
                    raise ValueError(
                        f"Domain cardinality exceeded in {tbl['file_name']}.{col['column_name']}: "
                        f"unique values exceed max_vals while allowed_values={col['allowed_values']}"
                    )
                unexpected = set(obs) - set(col["allowed_values"])
                if unexpected:
                    raise ValueError(
                        f"Disallowed values in {tbl['file_name']}.{col['column_name']}: "
                        f"{sorted(unexpected, key=str)} not in {col['allowed_values']}"
                    )

    # Compute exact live data metrics
    act = loaded["acdoca_actuals"]
    po = loaded["fact_po_commitments"]
    anom = loaded["eval_anomaly_labels"]
    l1 = act[act["BUZEI"] == 1]
    n_docs = int(len(l1))
    n_lines = int(len(act))

    kr_l1 = l1[l1["BLART"] == "KR"]
    za_l1 = l1[l1["BLART"] == "ZA"]
    ab_l1 = l1[l1["BLART"] == "AB"]
    dec25_open_za = int(((za_l1["GJAHR"] == 2025) & (za_l1["MONAT"] == 12) & (za_l1["STBLG"] == "")).sum())

    closed_po_inv = po[po["PO_STATUS"] == "CLOSED"].groupby("EBELN")["INVOICED_AMOUNT_USD"].sum()
    kr_ebeln_ksl = kr_l1[kr_l1["EBELN"] != ""].groupby("EBELN")["KSL"].sum()
    po_mrg = pd.DataFrame({"po_inv": closed_po_inv}).join(pd.DataFrame({"acdoca_ksl": kr_ebeln_ksl}), how="left")
    po_key_cov = float(po_mrg["acdoca_ksl"].notna().mean())
    po_med_delta = float(((po_mrg["acdoca_ksl"] - po_mrg["po_inv"]).abs() / po_mrg["po_inv"]).median())

    return {
        "schema_version": "2.2.2-freeze",
        "dataset_name": "SAP-Inspired Analytical Pharmaceutical OPEX Dataset (J&J FP&A Aligned)",
        "generator_script": "src/opex_forecast/data_prep/generate_dataset.py",
        "validator_script": "src/opex_forecast/data_prep/validate_dataset.py",
        "column_map_module": "src/opex_forecast/column_map.py",
        "column_crosswalk_csv": "docs/sap_column_crosswalk.csv",
        "validation_checks_count": validation_checks_count,
        "deterministic_seed": seed,
        "terminology_disclaimer": (
            "Driver-based synthetic OPEX generator producing an SAP-inspired analytical Universal Journal. "
            "KSL_BUDGET_FX is an analytical constant-FX extension, dim_fx_rates abstracts SAP TCURR/TCURF/TCURX, "
            "and dim_cost_center.WERKS is an analytical plant link (standard SAP CSKS has no WERKS column). "
            "Synthetic structural equations generate realistic spend patterns and do not establish empirical causality."
        ),
        "live_dataset_metrics": {
            "total_documents": n_docs,
            "total_journal_lines": n_lines,
            "max_belnr_observed": str(l1["BELNR"].max()),
            "ebeln_population": {
                "KR_populated": int((kr_l1["EBELN"] != "").sum()),
                "KR_total": int(len(kr_l1)),
                "ZA_populated": int((za_l1["EBELN"] != "").sum()),
                "ZA_total": int(len(za_l1)),
                "AB_populated": int((ab_l1["EBELN"] != "").sum()),
                "AB_total": int(len(ab_l1)),
            },
            "closed_po_reconciliation": {
                "closed_po_count": int(len(closed_po_inv)),
                "ebeln_key_coverage": round(po_key_cov, 6),
                "median_abs_dollar_delta": round(po_med_delta, 4),
            },
            "dec2025_legitimate_open_za_accruals": dec25_open_za,
            "anomalies": {
                "total": int(len(anom)),
                "rate_of_documents": round(len(anom) / n_docs, 6),
                "rate_of_journal_lines": round(len(anom) / n_lines, 6),
                "by_class": {k: int(v) for k, v in anom["ANOMALY_TAG"].value_counts().items()},
            },
        },
        "entity_cardinality": {
            "KOSTL_to_WERKS": "N:1 (24 cost centers -> 6 plants)",
            "KOSTL_to_PRCTR": "N:1 (24 cost centers -> 17 profit centers)",
            "WERKS_to_BUKRS": "N:1 (6 plants -> 3 company codes via T001W/T001K)",
            "BUKRS_to_WAERS": "1:1 in this 3-entity dataset (N:1 in multi-entity SAP systems)",
        },
        "csv_ingestion_spec": {
            "string_dtype_columns": sorted(STR_COLS.keys()),
            "blank_string_columns_requiring_fillna": list(BLANK_STR_COLS),
        },
        "scenario_definitions": {
            "BASE": "Deterministic projection from 2025-12-31 realized state + FY26 FORWARD_CURVE FX + scheduled milestones.",
            "HIRING_FREEZE_Q2": "PLANNED_HIRES = 0 from M4-M12 + smooth 0.9%/month natural attrition on ACTIVE_FTE & FIELD_SALES_REPS.",
            "PHASE3_DELAY_1Q": "Q2 (M4-M6) enrollment/site pause held flat near M3 + 3-month shifted ramp M7-M12 + milestones pushed +3 months.",
            "USD_STRENGTHEN_10PCT": "Operational drivers identical to BASE; FX_EUR_USD and FX_INR_USD multiplied by 0.90 (-10%).",
            "LAUNCH_ACCEL": "PRODUCT_LAUNCH_FLAG = 1 and +30% HCP_EVENTS_COUNT across Immunology Commercial_SGA in M3-M9.",
        },
        "table_count": len(table_defs),
        "total_column_count": sum(t["column_count"] for t in table_defs),
        "tables": table_defs,
    }


def write_outputs(catalog: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "pharma_opex_v22_metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    flat_rows = []
    for tbl in catalog["tables"]:
        pk_str = "|".join(tbl["primary_key"])
        for idx, col in enumerate(tbl["columns"], start=1):
            allowed = col.get("allowed_values")
            obs = col.get("observed_values")
            flat_rows.append(
                {
                    "table_name": tbl["table_name"],
                    "file_name": tbl["file_name"],
                    "table_category": tbl["category"],
                    "time_span": tbl["time_span"],
                    "row_count": tbl["row_count"],
                    "table_primary_key": pk_str,
                    "col_ordinal": idx,
                    "column_name": col["column_name"],
                    "readable_column": col["readable_column"],
                    "dtype": col["dtype"],
                    "is_primary_key": col["is_primary_key"],
                    "is_unique_key": col["is_unique_key"],
                    "foreign_key": col["foreign_key"] or "",
                    "nullable": col["nullable"],
                    "provenance": col["provenance"],
                    "sap_reference": col["sap_reference"] or "",
                    "ecc_equivalent": col.get("ecc_equivalent") or "",
                    "ml_role": col["ml_role"],
                    "allowed_values": "|".join('""' if v == "" else str(v) for v in allowed) if allowed is not None else "",
                    "observed_values": "|".join('""' if v == "" else str(v) for v in obs) if obs is not None else "",
                    "example": col["example"],
                    "description": col["description"],
                }
            )

    csv_path = os.path.join(out_dir, "pharma_opex_v22_data_dictionary.csv")
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    print(f"Wrote {json_path} ({catalog['table_count']} tables, {catalog['total_column_count']} columns)")
    print(f"Wrote {csv_path} ({len(flat_rows)} rows, 100% sampled from live CSVs)")


METRICS_START_MARKER = "<!-- LIVE_DATASET_METRICS_START -->"
METRICS_END_MARKER = "<!-- LIVE_DATASET_METRICS_END -->"


def format_live_metrics_markdown(catalog: Dict[str, Any]) -> str:
    m = catalog["live_dataset_metrics"]
    eb = m["ebeln_population"]
    po = m["closed_po_reconciliation"]
    anom = m["anomalies"]
    anom_classes = ", ".join(f"`{k}`: `{v:,}`" for k, v in anom["by_class"].items())
    row_counts = ", ".join(f"`{t['table_name']}` (`{t['row_count']:,}`)" for t in catalog["tables"])

    lines = [
        METRICS_START_MARKER,
        f"### Live Dataset Metrics Snapshot (`seed = {catalog['deterministic_seed']}`)",
        "",
        f"- **Documents & Journal Lines:** `{m['total_documents']:,}` documents · `{m['total_journal_lines']:,}` journal lines · max `BELNR`: `{m['max_belnr_observed']}`",
        f"- **Table Row Counts:** {row_counts}",
        f"- **`EBELN` Population:** `KR` `{eb['KR_populated']:,} / {eb['KR_total']:,}` · `ZA` `{eb['ZA_populated']:,} / {eb['ZA_total']:,}` · `AB` `{eb['AB_populated']:,} / {eb['AB_total']:,}`",
        f"- **Closed PO Reconciliation:** `{po['closed_po_count']:,}` closed POs · `{po['ebeln_key_coverage']:.2%}` `EBELN` key coverage · `{po['median_abs_dollar_delta']:.2%}` median abs dollar delta",
        f"- **Dec 2025 Legitimate Open `ZA` Accruals:** `{m['dec2025_legitimate_open_za_accruals']:,}`",
        f"- **Labeled Anomalies:** `{anom['total']:,}` total (`{anom['rate_of_documents']:.4%}` of docs / `{anom['rate_of_journal_lines']:.4%}` of lines) — {anom_classes}",
        METRICS_END_MARKER,
    ]
    return "\n".join(lines)


def inject_markdown_metrics(md_path: str, catalog: Dict[str, Any]) -> None:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    start_idx = text.find(METRICS_START_MARKER)
    end_idx = text.find(METRICS_END_MARKER)
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError(
            f"Missing {METRICS_START_MARKER} ... {METRICS_END_MARKER} markers in {md_path}"
        )
    end_idx += len(METRICS_END_MARKER)
    updated = text[:start_idx] + format_live_metrics_markdown(catalog) + text[end_idx:]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"Updated live_dataset_metrics block in {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build live-verified v2.2.2 JSON & CSV metadata catalogs")
    parser.add_argument("data_dir", nargs="?", default="./data/raw/pharma_v22", help="Directory with generated v2.2 CSVs")
    parser.add_argument("out_dir", nargs="?", default="./docs", help="Output directory for metadata JSON & CSV")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic generator seed recorded in metadata (default: 42)")
    parser.add_argument("--validation-checks", type=int, default=36, help="Validation check count recorded in metadata (default: 36)")
    parser.add_argument("--markdown", default=None, help="Optional Markdown data dictionary path to refresh live_dataset_metrics block")
    args = parser.parse_args()

    catalog = build_catalog(args.data_dir, seed=args.seed, validation_checks_count=args.validation_checks)
    write_outputs(catalog, args.out_dir)
    if args.markdown:
        inject_markdown_metrics(args.markdown, catalog)


if __name__ == "__main__":
    main()