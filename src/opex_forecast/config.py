"""Shared workspace paths and canonical SAP CSV ingestion constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "pharma_v22"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

STR_COL_NAMES: tuple[str, ...] = (
    "BELNR",
    "STBLG",
    "DOCLN",
    "HKONT",
    "RACCT",
    "EBELP",
    "BSCHL",
    "BUKRS",
    "RBUKRS",
    "WERKS",
    "KOSTL",
    "RCNTR",
    "RFAREA",
    "EBELN",
    "USNAM",
)
STR_COLS: dict[str, type[str]] = {col: str for col in STR_COL_NAMES}

BLANK_STR_COLS: tuple[str, ...] = ("STBLG", "KOSTL", "RCNTR", "RFAREA", "EBELN")
