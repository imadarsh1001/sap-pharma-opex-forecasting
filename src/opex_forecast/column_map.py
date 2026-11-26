"""Bidirectional SAP <-> readable (UPPER_SNAKE) column mapping utilities.

Codebase naming contract:
- Core pipelines under ``src/opex_forecast/`` operate on canonical SAP column names
  (``load_pharma_table(..., readable=False)`` default).
- Interactive notebooks, EDA, and presentation charts may opt into
  ``load_pharma_table(..., readable=True)`` and convert back via
  ``rename_to_sap(df, table_name)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import pandas as pd

from opex_forecast.config import DOCS_DIR

DEFAULT_DATA_DICTIONARY_CSV = DOCS_DIR / "pharma_opex_v22_data_dictionary.csv"

SAP_TO_READABLE_COMMON: dict[str, str] = {
    "BUKRS": "COMPANY_CODE",
    "RBUKRS": "COMPANY_CODE",
    "BUTXT": "COMPANY_NAME",
    "WERKS": "PLANT_CODE",
    "KOSTL": "COST_CENTER_ID",
    "RCNTR": "COST_CENTER_ID",
    "KOSTL_NAME": "COST_CENTER_NAME",
    "KOSTL_GROUP": "COST_CENTER_GROUP",
    "PRCTR": "PROFIT_CENTER_ID",
    "PRCTR_NAME": "PROFIT_CENTER_NAME",
    "RFAREA": "FUNCTIONAL_AREA",
    "HKONT": "GL_ACCOUNT_ID",
    "RACCT": "GL_ACCOUNT_ID",
    "GJAHR": "FISCAL_YEAR",
    "RYEAR": "LEDGER_FISCAL_YEAR",
    "MONAT": "POSTING_PERIOD",
    "POPER": "POSTING_PERIOD",
    "BELNR": "DOCUMENT_NUMBER",
    "BUZEI": "LINE_ITEM_NUMBER",
    "DOCLN": "LINE_ITEM_NUMBER",
    "BLART": "DOCUMENT_TYPE",
    "BLDAT": "DOCUMENT_DATE",
    "BUDAT": "POSTING_DATE",
    "CPUDT": "ENTRY_DATE",
    "USNAM": "POSTING_USER",
    "LIFNR": "VENDOR_ID",
    "EBELN": "PO_NUMBER",
    "EBELP": "PO_LINE_ITEM",
    "WAERS": "LOCAL_CURRENCY",
    "RWCUR": "LOCAL_CURRENCY",
    "SHKZG": "DEBIT_CREDIT_FLAG",
    "DRCRK": "DEBIT_CREDIT_FLAG",
    "BSCHL": "POSTING_KEY",
    "DMBTR": "AMOUNT_LOCAL_UNSIGNED",
    "HSL": "AMOUNT_LOCAL_SIGNED",
    "STBLG": "REVERSAL_DOCUMENT_NUMBER",
    "STJAH": "REVERSAL_FISCAL_YEAR",
    "PS_PSP_PNR": "WBS_ELEMENT",
    "RLDNR": "LEDGER_ID",
    "FROM_CURR": "SOURCE_CURRENCY",
    "TO_CURR": "TARGET_CURRENCY",
}

TABLE_SAP_TO_READABLE_OVERRIDES: dict[str, dict[str, str]] = {
    "dim_gl_account": {
        "HKONT": "GL_ACCOUNT_ID",
        "RACCT": "GL_ACCOUNT_ID_S4",
        "ALLOWED_RFAREAS": "ALLOWED_FUNCTIONAL_AREAS",
    },
    "acdoca_actuals": {
        "KSL": "AMOUNT_USD_REPORTED_UNSIGNED",
        "KSL_BUDGET_FX": "AMOUNT_USD_CONSTANT_FX_UNSIGNED",
    },
    "acdoca_universal_journal": {
        "KSL": "AMOUNT_USD_REPORTED_SIGNED",
        "KSL_BUDGET_FX": "AMOUNT_USD_CONSTANT_FX_SIGNED",
    },
    "plan_versions": {
        "DMBTR_PLAN": "PLAN_AMOUNT_LOCAL",
        "KSL_BUDGET": "PLAN_AMOUNT_USD",
    },
}

S4_ECC_EQUIVALENTS: dict[str, str] = {
    "RBUKRS": "BUKRS",
    "WERKS": "WERKS",
    "GJAHR": "GJAHR",
    "RYEAR": "GJAHR",
    "POPER": "MONAT",
    "BELNR": "BELNR",
    "DOCLN": "BUZEI",
    "BLDAT": "BLDAT",
    "BUDAT": "BUDAT",
    "CPUDT": "CPUDT",
    "BLART": "BLART",
    "USNAM": "USNAM",
    "RACCT": "HKONT",
    "RCNTR": "KOSTL",
    "PRCTR": "PRCTR",
    "RFAREA": "RFAREA",
    "PS_PSP_PNR": "WBS_ELEMENT",
    "LIFNR": "LIFNR",
    "EBELN": "EBELN",
    "RWCUR": "WAERS",
    "DRCRK": "SHKZG",
    "BSCHL": "BSCHL",
    "HSL": "Signed DMBTR",
    "KSL": "Signed KSL",
    "KSL_BUDGET_FX": "Signed KSL_BUDGET_FX",
    "STBLG": "STBLG",
    "STJAH": "STJAH",
    "SPECIAL_ITEM_FLAG": "SPECIAL_ITEM_FLAG",
    "SERVICE_START_DATE": "SERVICE_START_DATE",
    "SERVICE_END_DATE": "SERVICE_END_DATE",
    "RLDNR": "(S/4HANA only)",
    "VERSION": "VERSION",
}


def normalize_table_name(table_name: str) -> str:
    """Strip an optional directory or .csv suffix from a table name."""
    return Path(table_name).stem


@lru_cache(maxsize=4)
def _load_table_schemas(dd_path: str) -> dict[str, list[str]]:
    dd = pd.read_csv(dd_path, keep_default_na=False)
    schemas: dict[str, list[str]] = {}
    for tbl, grp in dd.groupby("table_name", sort=False):
        schemas[str(tbl)] = grp.sort_values("col_ordinal")["column_name"].tolist()
    return schemas


def get_table_column_map(
    table_name: str,
    dd_path: Path | str = DEFAULT_DATA_DICTIONARY_CSV,
    columns: Sequence[str] | None = None,
) -> dict[str, str]:
    """Return the exact {SAP_COL: READABLE_COL} dict for all columns of a table."""
    tbl = normalize_table_name(table_name)
    if columns is None:
        schemas = _load_table_schemas(str(Path(dd_path).resolve()))
        if tbl not in schemas:
            raise KeyError(f"Unknown table '{tbl}'. Known tables: {sorted(schemas)}")
        table_cols = schemas[tbl]
    else:
        table_cols = list(columns)

    overrides = TABLE_SAP_TO_READABLE_OVERRIDES.get(tbl, {})
    mapping: dict[str, str] = {}
    for col in table_cols:
        mapping[col] = overrides.get(col, SAP_TO_READABLE_COMMON.get(col, col))

    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"Duplicate readable column names detected for table '{tbl}': {mapping}")
    return mapping


def rename_to_readable(
    df: pd.DataFrame,
    table_name: str | None = None,
    dd_path: Path | str = DEFAULT_DATA_DICTIONARY_CSV,
) -> pd.DataFrame:
    """Rename DataFrame columns from canonical SAP names to readable UPPER_SNAKE names."""
    if table_name is not None:
        col_map = get_table_column_map(table_name, dd_path=dd_path)
    else:
        col_map = {col: SAP_TO_READABLE_COMMON.get(col, col) for col in df.columns}
    renamed = df.rename(columns=col_map)
    if renamed.columns.has_duplicates:
        dupes = renamed.columns[renamed.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns after readable rename ({table_name}): {dupes}")
    return renamed


def rename_to_sap(
    df: pd.DataFrame,
    table_name: str,
    strict: bool = False,
    dd_path: Path | str = DEFAULT_DATA_DICTIONARY_CSV,
) -> pd.DataFrame:
    """Convert readable column names back to table-specific canonical SAP names."""
    tbl = normalize_table_name(table_name)
    sap_to_readable = get_table_column_map(tbl, dd_path=dd_path)
    readable_to_sap = {readable: sap for sap, readable in sap_to_readable.items()}

    if strict:
        expected_readable = list(sap_to_readable.values())
        actual_cols = list(df.columns)
        if actual_cols != expected_readable:
            unknown = [c for c in actual_cols if c not in readable_to_sap]
            missing = [c for c in expected_readable if c not in actual_cols]
            raise ValueError(
                f"Strict SAP round-trip mismatch for '{tbl}': "
                f"unknown={unknown}, missing={missing}, order_equal={set(actual_cols) == set(expected_readable)}"
            )

    return df.rename(columns=readable_to_sap)


def build_column_crosswalk(
    dd_path: Path | str = DEFAULT_DATA_DICTIONARY_CSV,
) -> pd.DataFrame:
    """Build the full 206-row SAP <-> readable column crosswalk across all 14 tables."""
    resolved_dd = Path(dd_path).resolve()
    dd = pd.read_csv(resolved_dd, keep_default_na=False)
    rows: list[dict[str, object]] = []

    has_ecc_col = "ecc_equivalent" in dd.columns
    for tbl, grp in dd.groupby("table_name", sort=False):
        col_map = get_table_column_map(str(tbl), dd_path=resolved_dd)
        for _, r in grp.sort_values("col_ordinal").iterrows():
            sap_col = str(r["column_name"])
            ecc_val = ""
            if has_ecc_col and r["ecc_equivalent"] != "":
                ecc_val = str(r["ecc_equivalent"])
            elif str(tbl) == "acdoca_universal_journal":
                ecc_val = S4_ECC_EQUIVALENTS.get(sap_col, "")

            rows.append(
                {
                    "table_name": str(tbl),
                    "col_ordinal": int(r["col_ordinal"]),
                    "sap_column": sap_col,
                    "readable_column": col_map[sap_col],
                    "is_renamed": sap_col != col_map[sap_col],
                    "dtype": str(r["dtype"]),
                    "sap_reference": str(r["sap_reference"]),
                    "ecc_equivalent": ecc_val,
                    "ml_role": str(r["ml_role"]),
                    "description": str(r["description"]),
                }
            )
    return pd.DataFrame(rows)
