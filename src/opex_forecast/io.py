"""Canonical CSV loading helpers preserving SAP leading zeros and blank keys."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from opex_forecast.column_map import normalize_table_name, rename_to_readable
from opex_forecast.config import BLANK_STR_COLS, RAW_DATA_DIR, STR_COLS


def load_pharma_table(
    table_name: str,
    data_dir: Path | str | None = None,
    readable: bool = False,
) -> pd.DataFrame:
    """Load a raw v2.2 CSV table with canonical string dtypes and blank-string fillna."""
    base_dir = Path(data_dir) if data_dir is not None else RAW_DATA_DIR
    stem = normalize_table_name(table_name)
    file_name = table_name if table_name.endswith(".csv") else f"{stem}.csv"
    path = base_dir / file_name
    df = pd.read_csv(path, dtype=STR_COLS, low_memory=False)
    for col in BLANK_STR_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")
    if readable:
        return rename_to_readable(df, table_name=stem)
    return df
