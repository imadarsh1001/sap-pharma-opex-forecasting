# AGENTS.md — SAP Pharma OPEX Forecasting & Financial Intelligence

Compact engineering reference for AI coding sessions on `sap-pharma-opex-forecasting` (Python package: `opex_forecast`, `v2.2.2-freeze`, `seed=42`).

## 1. Architecture & Data Loading Rules
- **Always load raw tables via `opex_forecast.io.load_pharma_table(table_name, readable=False|True)`**, never raw `pd.read_csv()`. The loader enforces `dtype=str` on 15 leading-zero/ID columns (`STR_COLS`) and `.fillna("")` on 5 blank-string SAP columns (`BLANK_STR_COLS`: `STBLG`, `KOSTL`, `RCNTR`, `RFAREA`, `EBELN`).
- **Keep `data/raw/pharma_v22/` immutable.** Write derived panels to `data/processed/` and forecasts/benchmarks/reports to `outputs/`.
- **Use `opex_forecast.config` paths** (`PROJECT_ROOT`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `DOCS_DIR`, `OUTPUTS_DIR`) so scripts and `notebooks/` run identically from any working directory.

## 2. Naming Contract (No Split-Brain Code)
- **Core `src/opex_forecast/` pipelines:** Use **canonical SAP column names** (`readable=False`, the default).
- **Interactive `notebooks/`, EDA & executive charts:** Opt into **readable `UPPER_SNAKE` names** via `load_pharma_table(..., readable=True)` or `rename_to_readable(df, table_name)`.
- **Never mix SAP and readable column names in the same function.** Single source of truth: `src/opex_forecast/column_map.py` and `docs/sap_column_crosswalk.csv`.
- **Table-scoped reverse mapping:** Convert readable frames back via `rename_to_sap(df, table_name, strict=False|True)`. Never invert a flat global dict.

## 3. High-Frequency Column Crosswalk

| SAP (`S/4` / `ECC`) | Readable (`UPPER_SNAKE`) | Notes / Sign Convention |
| :--- | :--- | :--- |
| `RBUKRS` / `BUKRS` | `COMPANY_CODE` | Legal entity (`US10`, `EU20`, `AP30`) |
| `RCNTR` / `KOSTL` | `COST_CENTER_ID` | Cost center (`""` on BS lines) |
| `RACCT` / `HKONT` | `GL_ACCOUNT_ID` | `dim_gl_account.RACCT` -> `GL_ACCOUNT_ID_S4` |
| `POPER` / `MONAT` | `POSTING_PERIOD` | `1–12` regular; `13–16` special audit |
| `RYEAR` / `GJAHR` | `LEDGER_FISCAL_YEAR` / `FISCAL_YEAR` | `RYEAR == GJAHR` (`2022–2025` actuals) |
| `CPUDT` | `ENTRY_DATE` | PIT entry date (`CPUDT <= snapshot_date`) |
| `BUDAT` / `BLDAT` | `POSTING_DATE` / `DOCUMENT_DATE` | `BLDAT <= BUDAT <= CPUDT` |
| `BELNR` | `DOCUMENT_NUMBER` | 10-char zero-padded string |
| `DOCLN` / `BUZEI` | `LINE_ITEM_NUMBER` | **Dtype differs:** `DOCLN` `str`, `BUZEI` `int64` |
| `PS_PSP_PNR` / `WBS_ELEMENT` | `WBS_ELEMENT` | `PS_PSP_PNR` only in `acdoca_universal_journal` |
| `RWCUR` / `WAERS` | `LOCAL_CURRENCY` | `USD`, `EUR`, `INR` |
| `DRCRK` / `SHKZG` | `DEBIT_CREDIT_FLAG` | `S` = Debit (`40`), `H` = Credit (`50`) |
| `HSL` | `AMOUNT_LOCAL_SIGNED` | **Signed** local currency (`acdoca_universal_journal`) |
| `DMBTR` | `AMOUNT_LOCAL_UNSIGNED` | **Unsigned** local currency (`acdoca_actuals`) |
| `KSL` | `AMOUNT_USD_REPORTED_SIGNED` / `_UNSIGNED` | Reported USD (`ACTUAL_AVG` FX); **signed** in UJ, **unsigned** in ECC |
| `KSL_BUDGET_FX` | `AMOUNT_USD_CONSTANT_FX_SIGNED` / `_UNSIGNED` | Constant-FX USD (`BUDGET_PLAN` FX); primary ML target |
| `KSL_BUDGET` / `DMBTR_PLAN` | `PLAN_AMOUNT_USD` / `PLAN_AMOUNT_LOCAL` | `plan_versions` planned/forecasted spend |
| `STBLG` / `STJAH` | `REVERSAL_DOCUMENT_NUMBER` / `REVERSAL_FISCAL_YEAR` | Paired `ZA <-> AB`; `""` / `0` otherwise |
| `SPECIAL_ITEM_FLAG` | `SPECIAL_ITEM_FLAG` | Filter `== "NONE"` for recurring run-rate OPEX |

## 4. Known Schema Traps (Do Not Repeat)
1. **Use `acdoca_universal_journal` as the default analytics table.** Its amounts (`HSL`, `KSL`, `KSL_BUDGET_FX`) are **signed** so `SUM()` yields net balances directly. `acdoca_actuals` amounts (`DMBTR`, `KSL`, `KSL_BUDGET_FX`) are **unsigned** and require `SHKZG` decoding (`S` = `+`, `H` = `-`).
2. **`DOCLN` vs. `BUZEI` dtype mismatch:** Both map to `LINE_ITEM_NUMBER`, but `acdoca_universal_journal.DOCLN` is zero-padded `str` (`"000001"`, `"000002"`) while `acdoca_actuals.BUZEI` and `eval_anomaly_labels.BUZEI` are `int64` (`1`, `2`). Cast explicitly before joining UJ with `eval_anomaly_labels`.
3. **Table-scoped reverse mapping:** Always pass `table_name` to `rename_to_sap(df, table_name)` so `WBS_ELEMENT` stays `WBS_ELEMENT` on `fact_opex_drivers` (and 4 other tables) while reversing to `PS_PSP_PNR` on `acdoca_universal_journal`, and ECC tables reverse to `BUKRS`/`KOSTL`/`HKONT`/`MONAT`.

## 5. Non-Negotiable Accounting & Anti-Leakage Guardrails
1. **Canonical 4-condition recurring OPEX filter:**
   - `ACCOUNT_TYPE == "OPEX"` (`610010`–`650040`, excluding BS offsets `113100`, `140000`, `210000`, `215000`)
   - `POPER <= 12` (`POSTING_PERIOD <= 12`, excluding year-end special audit periods `13–16`)
   - `SPECIAL_ITEM_FLAG == "NONE"` (exclude GAAP non-recurring items)
   - `CPUDT <= cutoff_date` (`ENTRY_DATE <= cutoff_date`, strict point-in-time boundary)
2. **PO Commitments (`fact_po_commitments`):** Filter to a **single `SNAPSHOT_DATE`** per forecast origin before summing `OPEN_COMMITMENT_USD`.
3. **Ban `dim_cost_center.COST_SCALE`:** Synthetic generator metadata (`DO_NOT_USE_IN_MODEL`) — never use as an ML feature.
4. **Plan versions (`plan_versions`):** `VERSION == 1` (`BP`) is safe; keep `VERSION == 2` (`Q1_RF`, `JU`, `Q3_RF`, `NU`) strictly as **evaluation benchmarks**, never training features.
5. **Anomaly labels (`eval_anomaly_labels`):** Physically isolated ground truth — use **only** for Precision/Recall evaluation, never join into training or inference features.
6. **Dec 2025 open accruals boundary:** Exclude `GJAHR == 2025 & MONAT == 12` (`FISCAL_YEAR == 2025 & POSTING_PERIOD == 12`) from unreversed-accrual rules (`14` legitimate year-end open `ZA` accruals whose `AB` reversals belong to Jan 2026).

## 6. Verification Commands
- Validate raw dataset (`36/36`): `.\.venv\Scripts\python.exe src\opex_forecast\data_prep\validate_dataset.py`
- Sync JSON/CSV/Markdown metadata: `.\.venv\Scripts\python.exe src\opex_forecast\data_prep\build_metadata.py --markdown docs/sap_pharma_opex_data_dictionary.md`
