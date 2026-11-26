# SAP-Inspired Analytical Pharmaceutical OPEX Dataset — Data Dictionary (v2.2.2 Freeze)

## 1. Dataset Overview & Governance Metadata

| Metadata Attribute | Value |
| :--- | :--- |
| **Project / Repository** | SAP Pharma OPEX Forecasting & Financial Intelligence (`sap-pharma-opex-forecasting`) |
| **Schema Version** | `2.2.2-freeze` (Deterministic `seed = 42`) |
| **Generator Script** | `src/opex_forecast/data_prep/generate_dataset.py` (v2.2) |
| **Validator Script** | `src/opex_forecast/data_prep/validate_dataset.py` (`36/36` checks with Checks 35–36 applied; `34/34` on base v2.2) |
| **Metadata Builder** | `src/opex_forecast/data_prep/build_metadata.py` (auto-samples `seed=42` examples into `pharma_opex_v22_metadata.json` & `pharma_opex_v22_data_dictionary.csv`) |
| **Domain Alignment** | Pharmaceutical Enterprise FP&A (Johnson & Johnson-aligned hierarchy, campuses & planning cadence) |
| **Schema Provenance** | Driver-based synthetic OPEX generator producing an **SAP-inspired analytical Universal Journal** (inspired by S/4HANA `ACDOCA`, ECC `BKPF`/`BSEG`, and `COOI`/`EKPO`). `KSL_BUDGET_FX` is an analytical constant-FX extension, `dim_fx_rates` abstracts `TCURR`/`TCURF`/`TCURX`, and `dim_cost_center.WERKS` is an analytical plant link (standard SAP `CSKS` has no `WERKS` column). Structural driver equations generate realistic spend patterns for demo/benchmarking and do not prove empirical statistical causality. |
| **Actuals Coverage** | FY2022–FY2025 (`48` regular periods `1–12` + `16` special audit periods `13–16`) |
| **Plan & Scenario Horizon** | FY2022–FY2026 (`BP` for FY22–FY26; Rolling Forecasts for FY22–FY25; What-If Scenarios for FY26 anchored at `2025-12-31`) |
| **Organizational Scope** | `3` Company Codes (`US10` USD, `EU20` EUR, `AP30` INR) · `6` Plants (`WERKS`) · `24` Cost Centers (`KOSTL`) · `17` Profit Centers (`PRCTR`) · `26` GL Accounts (`22` OPEX `610010`–`650040` + `4` BS `113100`, `140000`, `210000`, `215000`) |
| **Reporting Currency** | Group USD: Reported `KSL` (`HSL * ACTUAL_AVG` FX) vs. Operational Constant Currency `KSL_BUDGET_FX` (`HSL * BUDGET_PLAN` FX) |

### Table Inventory, Keys, Exact Row Counts (`seed=42`) & ML Pipeline Roles

| # | Table (`file_name`) | Category | Cols | Time Span | Exact Rows (`seed=42`) | Primary Key | ML Pipeline Role |
| :-: | :--- | :--- | :-: | :--- | --: | :--- | :--- |
| 1 | `dim_company_code.csv` | `DIMENSION` | 4 | Static | 3 | `BUKRS` | Entity / Currency Lookup |
| 2 | `dim_plant.csv` | `DIMENSION` | 6 | Static | 6 | `WERKS` | Campus / Geography Slicing |
| 3 | `dim_cost_center.csv` | `DIMENSION` | 13 | Static | 24 | `KOSTL` | Hierarchical Rollups *(ban `COST_SCALE`)* |
| 4 | `dim_profit_center.csv` | `DIMENSION` | 5 | Static | 17 | `PRCTR` | Segment / TA / Region Rollups |
| 5 | `dim_gl_account.csv` | `DIMENSION` | 8 | Static | 26 | `HKONT` *(UK: `RACCT`)* | Cost-Behavior Segmentation & Filtering |
| 6 | `dim_fx_rates.csv` | `DIMENSION` | 6 | FY22–FY26 | 504 | `GJAHR, MONAT, FROM_CURR, RATE_TYPE` | Reported vs. Constant FX Translation |
| 7 | `fact_opex_drivers.csv` | `FACT_DRIVERS` | 21 | FY22–FY25 | 1,152 | `GJAHR, MONAT, KOSTL` | **Exogenous Features (Strictly Lagged `<= cutoff`)** |
| 8 | `plan_drivers.csv` | `FACT_PLAN_DRIVERS` | 13 | FY22–FY26 | 1,440 | `GJAHR, MONAT, KOSTL` | **Safe Future Plan Inputs & Variance Bridge** |
| 9 | `fact_po_commitments.csv` | `FACT_LEADING_INDICATOR` | 16 | FY21–FY25 (`<= 2025-12-31`) | 26,598 | `EBELN, EBELP, SNAPSHOT_DATE` | **Leading Feature (Filter Single `SNAPSHOT_DATE`)** |
| 10 | `acdoca_actuals.csv` | `FACT_TRANSACTIONAL_ECC` | 30 | FY22–FY25 | 75,268 (`37,634` docs) | `BUKRS, GJAHR, BELNR, BUZEI` | Audit / Duplicate & Accrual QA (`BKPF`/`BSEG`) |
| 11 | `acdoca_universal_journal.csv` | `FACT_TRANSACTIONAL_S4` | 32 | FY22–FY25 | 75,268 (`37,634` docs) | `RBUKRS, RYEAR, BELNR, DOCLN` | **Primary Target Source (Signed `HSL`/`KSL`)** |
| 12 | `plan_versions.csv` | `FACT_PLAN_AND_BENCHMARK` | 16 | FY22–FY26 | 49,392 (`196` series) | `GJAHR, MONAT, KOSTL, HKONT, CYCLE_NAME` | `VERSION=1` (`BP`) safe baseline; **`VERSION=2` Benchmark Only** |
| 13 | `scenario_assumptions.csv` | `FACT_SCENARIOS` | 25 | FY26 (`2025-12-31`) | 1,440 | `SCENARIO_ID, GJAHR, MONAT, KOSTL` | **Live FY26 What-If Model Scoring Input** |
| 14 | `eval_anomaly_labels.csv` | `EVALUATION_ONLY` | 11 | FY22–FY25 | 104 (`0.28%` docs) | `BUKRS, GJAHR, BELNR, BUZEI` | **STRICTLY EVALUATION ONLY (Never Join to Train)** |

**Totals:** `14` tables · `206` columns.

<!-- LIVE_DATASET_METRICS_START -->
### Live Dataset Metrics Snapshot (`seed = 42`)

- **Documents & Journal Lines:** `37,634` documents · `75,268` journal lines · max `BELNR`: `0100004399`
- **Table Row Counts:** `dim_company_code` (`3`), `dim_plant` (`6`), `dim_cost_center` (`24`), `dim_profit_center` (`17`), `dim_gl_account` (`26`), `dim_fx_rates` (`504`), `fact_opex_drivers` (`1,152`), `plan_drivers` (`1,440`), `fact_po_commitments` (`26,598`), `acdoca_actuals` (`75,268`), `acdoca_universal_journal` (`75,268`), `plan_versions` (`49,392`), `scenario_assumptions` (`1,440`), `eval_anomaly_labels` (`104`)
- **`EBELN` Population:** `KR` `24,337 / 24,345` · `ZA` `564 / 564` · `AB` `522 / 522`
- **Closed PO Reconciliation:** `7,488` closed POs · `100.00%` `EBELN` key coverage · `2.01%` median abs dollar delta
- **Dec 2025 Legitimate Open `ZA` Accruals:** `14`
- **Labeled Anomalies:** `104` total (`0.2763%` of docs / `0.1382%` of lines) — `DUPLICATE_INVOICE`: `36`, `UNREVERSED_ACCRUAL`: `28`, `Q4_BUDGET_FLUSH`: `24`, `TRIAL_PAUSE_DROP`: `16`
<!-- LIVE_DATASET_METRICS_END -->

---

## 2. Entity Relationship Cardinality & J&J Reporting Hierarchy

```text
Organizational & Currency Cardinality (Many-to-One Chain):
  KOSTL (24 Cost Centers)
    ├── N:1 ──> WERKS (6 Plants: US01, US02, US03, BE01, NL01, IN01)
    │             └── N:1 ──> BUKRS (3 Company Codes: US10, EU20, AP30 via T001W/T001K)
    │                           └── 1:1* ─> WAERS (USD, EUR, INR)
    └── N:1 ──> PRCTR (17 Profit Centers)

  * BUKRS -> WAERS is 1:1 in this 3-entity dataset (N:1 in multi-entity SAP systems).

Hierarchical Forecasting & Reconciliation Tree:
  RACCT (22 OPEX GLs: 610010-650040)
    └── RCNTR (24 Cost Centers)
          └── RFAREA (4 Functional Areas: R&D, Commercial_SGA, Medical_Affairs, G&A)
                └── FRANCHISE_TA (7 Franchises/TAs: Oncology, Immunology, Neuroscience,
                                  Cardiovascular, Surgery, Vision, Corporate)
                      └── SEGMENT (3 Segments: Innovative_Medicine, MedTech, Enterprise_Corporate)
                            └── Global Enterprise
```

---

## 3. Planning Cadence & Point-in-Time Anti-Leakage Cutoffs

| Cycle (`CYCLE_NAME`) | `VERSION` | Horizon | `SNAPSHOT_DATE` | Closed Actuals Cutoff | Driver & FX Source |
| :--- | :-: | :-: | :--- | :--- | :--- |
| `BP` (Annual Business Plan) | `1` | `0+12` | `{yr-1}-10-15` | None (`M0`) | Oct `{yr-1}` FTE/sites/patients bridged across Nov–Dec; Oct `BUDGET_PLAN` FX; decoupled `PLAN_MILESTONE_EVENT`. |
| `Q1_RF` (Q1 Latest Thinking) | `2` | `3+9` | `{yr}-04-10` | `MONAT <= 3` (`CPUDT <= 04-10`) | Closed M1–3 copy realized clean `HSL` at `BUDGET_PLAN` FX; open M4–12 run-rate heuristic. |
| `JU` (June Update) | `2` | `5+7` | `{yr}-06-15` | `MONAT <= 5` (`CPUDT <= 06-15`) | Closed M1–5 copy realized clean `HSL` at `BUDGET_PLAN` FX; open M6–12 run-rate heuristic. |
| `Q3_RF` (Q3 Latest Thinking) | `2` | `9+3` | `{yr}-10-10` | `MONAT <= 9` (`CPUDT <= 10-10`) | Closed M1–9 copy realized clean `HSL` at `BUDGET_PLAN` FX; open M10–12 run-rate heuristic. |
| `NU` (November Update) | `2` | `10+2` | `{yr}-11-15` | `MONAT <= 10` (`CPUDT <= 11-15`) | Closed M1–10 copy realized clean `HSL` at `BUDGET_PLAN` FX; open M11–12 run-rate heuristic. |
| FY26 Scenarios (`BASE` + 4 cases) | — | `0+12` | `2025-12-31` | FY25 closed | Deterministic projection strictly from Dec 2025 realized state + FY26 `FORWARD_CURVE` FX. |

---

## 4. Table-by-Table Field Dictionary (All 14 Tables · 206 Columns)

### 4.1 `dim_company_code.csv` (`3` rows · `4` cols)
- **Category:** `DIMENSION` · **Primary Key:** `BUKRS` · **SAP Inspiration:** `T001`
- **Description:** Legal entity master defining company codes, local reporting currencies, and regions.

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BUKRS` | `COMPANY_CODE` | `string` | PK | SAP (`T001-BUKRS`) | `HIERARCHY_KEY` | SAP Company Code. Values: `US10`, `EU20`, `AP30`. |
| `BUTXT` | `COMPANY_NAME` | `string` | — | SAP (`T001-BUTXT`) | `METADATA` | Legal entity name (`J&J Pharma US Inc.`, `J&J Pharma EMEA B.V.`, `J&J Pharma APAC Pvt Ltd`). |
| `WAERS` | `LOCAL_CURRENCY` | `string` | — | SAP (`T001-WAERS`) | `CURRENCY_KEY` | Local functional currency key (`1:1` per `BUKRS` here; `N:1` in multi-entity SAP). Values: `USD`, `EUR`, `INR`. |
| `REGION` | `REGION` | `string` | — | Analytical | `HIERARCHY_KEY` | Geographic reporting region. Values: `US`, `EMEA`, `APAC`. |

### 4.2 `dim_plant.csv` (`6` rows · `6` cols)
- **Category:** `DIMENSION` · **Primary Key:** `WERKS` · **SAP Inspiration:** `T001W` / `T001K`
- **Description:** Six J&J-aligned campuses mapped `N:1` to company codes: `US01` Titusville Commercial HQ, `US02` Raritan R&D Campus, `US03` Spring House R&D Campus, `BE01` Beerse EMEA R&D Hub, `NL01` Leiden EMEA Commercial Hub, and `IN01` Mumbai/Hyderabad APAC Hub.

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `WERKS` | `PLANT_CODE` | `string` | PK | SAP (`T001W-WERKS`) | `HIERARCHY_KEY` | Plant / campus code. Values: `US01`, `US02`, `US03`, `BE01`, `NL01`, `IN01`. |
| `PLANT_NAME` | `PLANT_NAME` | `string` | — | SAP (`T001W-NAME1`) | `METADATA` | Campus name (`Titusville Commercial HQ`, `Raritan R&D Campus`, `Spring House R&D Campus`, `Beerse EMEA R&D Hub`, `Leiden EMEA Commercial Hub`, `Mumbai/Hyderabad APAC Hub`). |
| `BUKRS` | `COMPANY_CODE` | `string` | FK -> `dim_company_code.BUKRS` | SAP (`T001W/T001K-BUKRS`) | `HIERARCHY_KEY` | Owning company code (in native SAP, `T001W` links to `T001K-BUKRS` via Valuation Area `BWKEY`). Values: `US10`, `EU20`, `AP30`. |
| `REGION` | `REGION` | `string` | — | Analytical | `HIERARCHY_KEY` | Geographic region. Values: `US`, `EMEA`, `APAC`. |
| `CITY` | `CITY` | `string` | — | SAP (`T001W-ORT01`) | `METADATA` | Site location (`Titusville, NJ`, `Raritan, NJ`, `Spring House, PA`, `Beerse, Belgium`, `Leiden, Netherlands`, `Mumbai / Hyderabad, India`). |
| `PLANT_TYPE` | `PLANT_TYPE` | `string` | — | Analytical | `HIERARCHY_ATTRIBUTE` | Campus operational type. Values: `Commercial_HQ`, `RD_Campus`, `Regional_Hub`. |

### 4.3 `dim_cost_center.csv` (`24` rows · `13` cols)
- **Category:** `DIMENSION` · **Primary Key:** `KOSTL` · **SAP Inspiration:** `CSKS` / `CSKT` + Analytical Hierarchy Extensions
- **Description:** Twenty-four organizational cost centers (`CC1010`–`CC3050`) providing the `N:1` hierarchy links across Plants, Profit Centers, Departments (`10` values), Franchises/TAs (`7` values), Functional Areas (`4` values), and Segments (`3` values).

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BUKRS` | `COMPANY_CODE` | `string` | FK -> `dim_company_code.BUKRS` | SAP (`CSKS-BUKRS`) | `HIERARCHY_KEY` | Owning company code. Values: `US10`, `EU20`, `AP30`. |
| `WERKS` | `PLANT_CODE` | `string` | FK -> `dim_plant.WERKS` | Analytical (`Plant Link`) | `HIERARCHY_KEY` | Assigned campus / plant (analytical extension; standard SAP `CSKS` has no `WERKS` column). Values: `US01`, `US02`, `US03`, `BE01`, `NL01`, `IN01`. |
| `KOSTL` | `COST_CENTER_ID` | `string` | PK | SAP (`CSKS-KOSTL`) | `ENTITY_KEY` | Cost center identifier (`CC1010`–`CC3050`). |
| `KOSTL_NAME` | `COST_CENTER_NAME` | `string` | — | SAP (`CSKT-LTEXT`) | `METADATA` | Cost center description (e.g., `US Oncology Clinical Ops`). |
| `DEPARTMENT` | `DEPARTMENT` | `string` | — | SAP (`CSKS-ABTEI`) | `HIERARCHY_ATTRIBUTE` | Operational department (`10` values: `Clinical_Development`, `Discovery_Research`, `Regulatory_Affairs`, `Brand_Commercial`, `MedTech_Commercial`, `MedTech_RD`, `Medical_Science`, `Enterprise_IT`, `Finance_Legal`, `Global_Shared_Services`). |
| `KOSTL_GROUP` | `COST_CENTER_GROUP` | `string` | — | Analytical | `HIERARCHY_ATTRIBUTE` | Coarse legacy cost center group. Values: `RND`, `SALES`, `ADMIN`. |
| `SEGMENT` | `SEGMENT` | `string` | — | Analytical (`FAGL_SEGM`) | `HIERARCHY_KEY` | J&J business segment. Values: `Innovative_Medicine`, `MedTech`, `Enterprise_Corporate`. |
| `FRANCHISE_TA` | `FRANCHISE_TA` | `string` | — | Analytical | `HIERARCHY_KEY` | Therapeutic Area or MedTech franchise (`7` values: `Oncology`, `Immunology`, `Neuroscience`, `Cardiovascular`, `Surgery`, `Vision`, `Corporate`). |
| `REGION` | `REGION` | `string` | — | Analytical | `HIERARCHY_KEY` | Geographic region. Values: `US`, `EMEA`, `APAC`. |
| `RFAREA` | `FUNCTIONAL_AREA` | `string` | — | SAP (`CSKS-FUNC_AREA`) | `HIERARCHY_KEY` | Functional area for P&L classification and posting-matrix validation. Values: `R&D`, `Commercial_SGA`, `Medical_Affairs`, `G&A`. |
| `PRCTR` | `PROFIT_CENTER_ID` | `string` | FK -> `dim_profit_center.PRCTR` | SAP (`CSKS-PRCTR`) | `HIERARCHY_KEY` | Assigned profit center (`N:1` `KOSTL -> PRCTR`). |
| `DEFAULT_WBS` | `DEFAULT_WBS` | `string` | — | Analytical (`PRPS-POSID`) | `HIERARCHY_ATTRIBUTE` | Default clinical study WBS element (`WBS-*` for clinical R&D cost centers; `NONE` otherwise). |
| `COST_SCALE` | `COST_SCALE` | `float64` | — | **Generator Only** | `DO_NOT_USE_IN_MODEL` | **GENERATOR METADATA ONLY — NEVER USE AS AN ML FEATURE.** Synthetic relative sizing multiplier (`0.45`–`1.40`) with no production SAP equivalent. |

### 4.4 `dim_profit_center.csv` (`17` rows · `5` cols)
- **Category:** `DIMENSION` · **Primary Key:** `PRCTR` · **SAP Inspiration:** `CEPC` / `CEPCT`
- **Description:** Profit center reporting rollups combining Business Segment, Franchise/TA, and Region.

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `PRCTR` | `PROFIT_CENTER_ID` | `string` | PK | SAP (`CEPC-PRCTR`) | `HIERARCHY_KEY` | Profit center identifier (e.g., `PC_ONC_US`, `PC_SUR_EU`, `PC_CORP_GLB`). |
| `SEGMENT` | `SEGMENT` | `string` | — | Analytical | `HIERARCHY_KEY` | J&J business segment. Values: `Innovative_Medicine`, `MedTech`, `Enterprise_Corporate`. |
| `FRANCHISE_TA` | `FRANCHISE_TA` | `string` | — | Analytical | `HIERARCHY_KEY` | Therapeutic Area or franchise (`Oncology`, `Immunology`, `Neuroscience`, `Cardiovascular`, `Surgery`, `Vision`, `Corporate`). |
| `REGION` | `REGION` | `string` | — | Analytical | `HIERARCHY_KEY` | Geographic region. Values: `US`, `EMEA`, `APAC`. |
| `PRCTR_NAME` | `PROFIT_CENTER_NAME` | `string` | — | SAP (`CEPCT-LTEXT`) | `METADATA` | Composite profit center label (`SEGMENT - FRANCHISE_TA (REGION)`). |

### 4.5 `dim_gl_account.csv` (`26` rows · `8` cols)
- **Category:** `DIMENSION` · **Primary Key:** `HKONT` · **Unique Key:** `RACCT` · **SAP Inspiration:** `SKA1` / `SKAT`
- **Description:** Life-sciences Chart of Accounts comprising `22` OPEX P&L accounts (`610010`–`650040`) and `4` Balance Sheet offset accounts (`113100` Cash & Bank Clearing, `140000` Prepaid Expenses, `210000` Payroll Clearing Liability, `215000` Accounts Payable & Accruals).

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `HKONT` | `GL_ACCOUNT_ID` | `string` | PK | SAP (`SKA1-SAKNR`) | `ENTITY_KEY` | 6-character GL account number (ECC naming: `610010`–`650040` OPEX; `113100`, `140000`, `210000`, `215000` BS). |
| `GL_NAME` | `GL_NAME` | `string` | — | SAP (`SKAT-TXT50`) | `METADATA` | GL account text description (e.g., `Clinical Trial CRO Services`, `Payroll Clearing Liability`). |
| `ACCOUNT_TYPE` | `ACCOUNT_TYPE` | `string` | — | Analytical | `FILTER_FLAG` | Financial statement class. Always filter `ACCOUNT_TYPE == "OPEX"` for OPEX modeling. Values: `OPEX`, `BS`. |
| `COST_BEHAVIOR` | `COST_BEHAVIOR` | `string` | — | Analytical | `MODEL_SEGMENTATION` | Economic cost behavior driving segmented model selection (ETS vs. global ML). Values: `FIXED`, `STEP_FIXED`, `VARIABLE`, `DISCRETIONARY`, `MILESTONE`, `NONE`. |
| `OPEX_GROUP` | `OPEX_GROUP` | `string` | — | Analytical | `HIERARCHY_KEY` | Sub-functional grouping (`10` values: `WORKFORCE_CORE`, `R&D_CLINICAL`, `R&D_DISCOVERY`, `R&D_REGULATORY`, `COMMERCIAL_FIELD`, `COMMERCIAL_MARKETING`, `MEDICAL_AFFAIRS`, `G&A_IT`, `G&A_CORP`, `BALANCE_SHEET`). |
| `ALLOWED_RFAREAS` | `ALLOWED_FUNCTIONAL_AREAS` | `string` | — | Analytical | `VALIDATION_RULE` | Pipe-delimited functional areas permitted to post to this GL (`ALL` for BS accounts). |
| `USES_PO` | `USES_PO` | `bool` | — | Analytical | `FEATURE_FLAG` | `True` if vendor spend flows through Purchase Orders (`fact_po_commitments`); `False` for payroll, rent, milestones, and T&E. Values: `True`, `False`. |
| `RACCT` | `GL_ACCOUNT_ID_S4` | `string` | **UK** | SAP (`ACDOCA-RACCT`) | `ENTITY_KEY` | S/4HANA Universal Journal GL account alias (unique 1:1 mirror of `HKONT`). |

### 4.6 `dim_fx_rates.csv` (`504` rows · `6` cols)
- **Category:** `DIMENSION` · **Primary Key:** `GJAHR, MONAT, FROM_CURR, RATE_TYPE` · **SAP Inspiration:** Simplified analytical abstraction of `TCURR`/`TCURF`/`TCURX`
- **Description:** Direct-quote monthly exchange rates to USD across `ACTUAL_AVG` (FY22–25), `BUDGET_PLAN` (constant M1–M12 per year, locked strictly to prior-year Oct `ACTUAL_AVG`), and `FORWARD_CURVE` (FY22–26).

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `GJAHR` | `FISCAL_YEAR` | `int64` | PK | SAP (`GJAHR`) | `TIME_INDEX` | Fiscal year (`2022`–`2026`). |
| `MONAT` | `POSTING_PERIOD` | `int64` | PK | SAP (`MONAT`) | `TIME_INDEX` | Fiscal month (`1`–`12`). |
| `FROM_CURR` | `SOURCE_CURRENCY` | `string` | PK | SAP (`TCURR-FCURR`) | `CURRENCY_KEY` | Source local currency. Values: `USD`, `EUR`, `INR`. |
| `TO_CURR` | `TARGET_CURRENCY` | `string` | — | SAP (`TCURR-TCURR`) | `CURRENCY_KEY` | Group reporting currency. Values: `USD`. |
| `RATE_TYPE` | `RATE_TYPE` | `string` | PK | Analytical (`TCURR-KURST`) | `FILTER_FLAG` | Exchange rate curve type. Values: `ACTUAL_AVG`, `BUDGET_PLAN`, `FORWARD_CURVE`. |
| `FX_RATE` | `FX_RATE` | `float64` | — | SAP (`TCURR-UKURS`) | `FX_DRIVER` | Direct quote multiplier: USD per 1 unit of `FROM_CURR` (`USD = Local * FX_RATE`). |

### 4.7 `fact_opex_drivers.csv` (`1,152` rows · `21` cols)
- **Category:** `FACT_DRIVERS` · **Primary Key:** `GJAHR, MONAT, KOSTL` · **Source Inspiration:** Workday HR + CTMS + Commercial CRM + Macro
- **Description:** Realized monthly operational, clinical, commercial, and macroeconomic drivers for FY2022–FY2025. **Anti-Leakage Rule:** At forecast origin $t$, use only observations with `(GJAHR, MONAT) <= t` (lags).

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `GJAHR` | `FISCAL_YEAR` | `int64` | PK | SAP (`GJAHR`) | `TIME_INDEX` | Fiscal year (`2022`–`2025`). |
| `MONAT` | `POSTING_PERIOD` | `int64` | PK | SAP (`MONAT`) | `TIME_INDEX` | Fiscal month (`1`–`12`). |
| `BUKRS` | `COMPANY_CODE` | `string` | FK -> `dim_company_code.BUKRS` | SAP (`BUKRS`) | `HIERARCHY_KEY` | Owning company code. Values: `US10`, `EU20`, `AP30`. |
| `WERKS` | `PLANT_CODE` | `string` | FK -> `dim_plant.WERKS` | Analytical (`Plant Link`) | `HIERARCHY_KEY` | Campus / plant code. Values: `US01`, `US02`, `US03`, `BE01`, `NL01`, `IN01`. |
| `KOSTL` | `COST_CENTER_ID` | `string` | PK, FK -> `dim_cost_center.KOSTL` | SAP (`CSKS-KOSTL`) | `ENTITY_KEY` | Cost center identifier. |
| `WBS_ELEMENT` | `WBS_ELEMENT` | `string` | — | SAP (`PRPS-POSID`) | `HIERARCHY_KEY` | Clinical study WBS element (`WBS-*` or `NONE`). |
| `ACTIVE_FTE` | `ACTIVE_FTE` | `float64` | — | Analytical (`HR/Workday`) | `EXOGENOUS_DRIVER` | Active Full-Time Equivalent headcount; primary driver of workforce and step-fixed IT OPEX. |
| `PLANNED_HIRES` | `PLANNED_HIRES` | `int64` | — | Analytical (`HR/Workday`) | `EXOGENOUS_DRIVER` | Realized gross monthly Poisson hires feeding `ACTIVE_FTE` evolution (compare with deterministic planned hires in `scenario_assumptions.csv`). |
| `FIELD_SALES_REPS` | `FIELD_SALES_REPS` | `float64` | — | Analytical (`CRM/HR`) | `EXOGENOUS_DRIVER` | Field commercial sales force headcount (`~68%` of `ACTIVE_FTE` in `Commercial_SGA`; `0.0` elsewhere). |
| `MERIT_CYCLE_FLAG` | `MERIT_CYCLE_FLAG` | `int64` | — | Analytical | `CALENDAR_DRIVER` | Binary indicator for annual April merit step-up (`1` when `MONAT == 4`; `0` otherwise). Values: `0`, `1`. |
| `TRIAL_PHASE` | `TRIAL_PHASE` | `string` | — | Analytical (`CTMS`) | `EXOGENOUS_DRIVER` | Clinical trial development phase (`P1` graduates to `P2` in FY25; `P2` graduates to `P3` in FY24). Values: `NONE`, `P1`, `P2`, `P3`, `FILING`. |
| `ACTIVE_SITES` | `ACTIVE_SITES` | `float64` | — | Analytical (`CTMS`) | `EXOGENOUS_DRIVER` | Active clinical investigator sites (enters CRO/site spend with a 1-month billing lag). |
| `ENROLLED_PATIENTS` | `ENROLLED_PATIENTS` | `float64` | — | Analytical (`CTMS`) | `EXOGENOUS_DRIVER` | Active enrolled trial patients (enters CRO/site spend with a 1-month billing lag). |
| `MILESTONE_EVENT` | `MILESTONE_EVENT` | `string` | — | Analytical (`CTMS`) | `EXOGENOUS_DRIVER` | Realized lumpy clinical/regulatory milestone triggering milestone payments. Values: `NONE`, `FPI`, `DB_LOCK`, `FDA_FILING`. |
| `PRODUCT_LAUNCH_FLAG` | `PRODUCT_LAUNCH_FLAG` | `int64` | — | Analytical | `EXOGENOUS_DRIVER` | Binary commercial launch window flag (`Oncology` `Commercial_SGA`: FY2023 M4–M10; `Immunology` `Commercial_SGA`: FY2025 M2–M7). Values: `0`, `1`. |
| `LOE_FLAG` | `LOE_FLAG` | `int64` | — | Analytical | `EXOGENOUS_DRIVER` | Binary Loss of Exclusivity / biosimilar defense flag (`Immunology` `Commercial_SGA`: FY2024 M7–M12; `0` otherwise). Values: `0`, `1`. |
| `HCP_EVENTS_COUNT` | `HCP_EVENTS_COUNT` | `int64` | — | Analytical (`Commercial_CRM`) | `EXOGENOUS_DRIVER` | Monthly Healthcare Professional speaker programs & congress events (`Commercial_SGA` only; `0` otherwise). |
| `SURGICAL_PROCEDURE_INDEX` | `SURGICAL_PROCEDURE_INDEX` | `float64` | — | Analytical | `EXOGENOUS_DRIVER` | MedTech surgical procedure volume index (base `~100`, Aug dip / Q4 surge; `100.0` for non-MedTech). |
| `CPI_INDEX` | `CPI_INDEX` | `float64` | — | Macro External | `MACRO_DRIVER` | Compounded Consumer Price Index (`Jan 2022 = 100.0`); escalates vendor/CRO/agency contracts. |
| `WAGE_INDEX` | `WAGE_INDEX` | `float64` | — | Macro External | `MACRO_DRIVER` | Compounded wage index (`Jan 2022 = 100.0`), strictly non-decreasing and stepping up only in April. |
| `WAGE_INFLATION_RATE` | `WAGE_INFLATION_RATE` | `float64` | — | Macro External | `MACRO_DRIVER` | Annualized merit wage inflation rate (`0.048` FY22 -> `0.034` FY25). |

### 4.8 `plan_drivers.csv` (`1,440` rows · `13` cols)
- **Category:** `FACT_PLAN_DRIVERS` · **Primary Key:** `GJAHR, MONAT, KOSTL` · **Source Inspiration:** Annual Business Plan (`BP`) Driver Assumptions
- **Description:** Point-in-time, leak-free Annual Business Plan (`BP`) driver targets for FY2022–FY2026 locked in October of `GJAHR - 1` (with a 2-month Nov–Dec bridge). Safe to use as future exogenous planning features and for Volume vs. Rate Variance Bridges.

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `GJAHR` | `FISCAL_YEAR` | `int64` | PK | SAP (`GJAHR`) | `TIME_INDEX` | Fiscal year (`2022`–`2026`). |
| `MONAT` | `POSTING_PERIOD` | `int64` | PK | SAP (`MONAT`) | `TIME_INDEX` | Fiscal month (`1`–`12`). |
| `BUKRS` | `COMPANY_CODE` | `string` | FK -> `dim_company_code.BUKRS` | SAP (`BUKRS`) | `HIERARCHY_KEY` | Company code. Values: `US10`, `EU20`, `AP30`. |
| `WERKS` | `PLANT_CODE` | `string` | FK -> `dim_plant.WERKS` | Analytical (`Plant Link`) | `HIERARCHY_KEY` | Campus / plant code. Values: `US01`, `US02`, `US03`, `BE01`, `NL01`, `IN01`. |
| `KOSTL` | `COST_CENTER_ID` | `string` | PK, FK -> `dim_cost_center.KOSTL` | SAP (`CSKS-KOSTL`) | `ENTITY_KEY` | Cost center identifier. |
| `WBS_ELEMENT` | `WBS_ELEMENT` | `string` | — | SAP (`PRPS-POSID`) | `HIERARCHY_KEY` | Clinical study WBS element. |
| `PLAN_FTE` | `PLAN_FTE` | `float64` | — | Analytical | `SAFE_FUTURE_PLAN_FEATURE` | BP target FTE headcount projected leak-free from Oct `yr-1` anchor (with Nov–Dec bridge). |
| `PLAN_ACTIVE_SITES` | `PLAN_ACTIVE_SITES` | `float64` | — | Analytical | `SAFE_FUTURE_PLAN_FEATURE` | BP target active clinical sites projected from Oct `yr-1` anchor. |
| `PLAN_ENROLLED_PATIENTS` | `PLAN_ENROLLED_PATIENTS` | `float64` | — | Analytical | `SAFE_FUTURE_PLAN_FEATURE` | BP target enrolled patients projected from Oct `yr-1` anchor. |
| `PLAN_HCP_EVENTS` | `PLAN_HCP_EVENTS` | `int64` | — | Analytical | `SAFE_FUTURE_PLAN_FEATURE` | BP target HCP speaker/congress events (blind to unscheduled launches). |
| `PLAN_MILESTONE_EVENT` | `PLAN_MILESTONE_EVENT` | `string` | — | Analytical | `SAFE_FUTURE_PLAN_FEATURE` | BP locked milestone schedule (decoupled from realized 1–2 month clinical slips/accelerations). Values: `NONE`, `FPI`, `DB_LOCK`, `FDA_FILING`. |
| `PLAN_CPI_INDEX` | `PLAN_CPI_INDEX` | `float64` | — | Analytical | `SAFE_FUTURE_PLAN_FEATURE` | BP locked CPI assumption (`3.0%/yr` from Oct `yr-1`). |
| `PLAN_WAGE_INDEX` | `PLAN_WAGE_INDEX` | `float64` | — | Analytical | `SAFE_FUTURE_PLAN_FEATURE` | BP locked wage index assumption (`3.6%` April merit step from Oct `yr-1`). |

### 4.9 `fact_po_commitments.csv` (`26,598` rows · `16` cols)
- **Category:** `FACT_LEADING_INDICATOR` · **Primary Key:** `EBELN, EBELP, SNAPSHOT_DATE` · **SAP Inspiration:** `COOI` / `EKPO`
- **Description:** Monthly month-end point-in-time Purchase Order commitment snapshots (`1–2` month lead before Goods Receipt/invoice), hard-capped at `PO_CREATE_DATE <= 2025-12-31` and `SNAPSHOT_DATE <= 2025-12-31`.
- **PO-to-Invoice Reconciliation Scope:** All **`7,488 / 7,488` (`100.0%`)** `CLOSED` POs reconcile by **`EBELN` key coverage** to `KR` invoices in `acdoca_actuals.EBELN`. Dollar totals intentionally differ (**`2.01%` median absolute delta**; `2 / 7,488` within `$1`) due to: (1) realistic `80/20` PO-to-spot spend variance (`0.80 * po_total * U(0.95, 1.05) + 0.20 * base_usd`), (2) `650010` Cloud/SaaS converting one invoice slice to an unlinked prepaid `SA` release (`EBELN == ""`), and (3) `TRIAL_PAUSE_DROP` months where actuals drop `62%` while pre-hold POs still close.
- **Anti-Leakage Join Rule:** Every PO spans `2–3` monthly snapshots; always filter to a **single `SNAPSHOT_DATE == forecast_origin_month_end`** before summing `OPEN_COMMITMENT_USD` by `(KOSTL, HKONT, EXPECTED_GR_MONTH)`.

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EBELN` | `PO_NUMBER` | `string` | PK | SAP (`EKPO-EBELN`) | `DOCUMENT_KEY` | Purchase order number (`PO4500001001`+). |
| `EBELP` | `PO_LINE_ITEM` | `string` | PK | SAP (`EKPO-EBELP`) | `DOCUMENT_KEY` | 5-char PO line item. Values: `00010`, `00020`. |
| `BUKRS` | `COMPANY_CODE` | `string` | FK -> `dim_company_code.BUKRS` | SAP (`EKKO-BUKRS`) | `HIERARCHY_KEY` | Company code. Values: `US10`, `EU20`, `AP30`. |
| `WERKS` | `PLANT_CODE` | `string` | FK -> `dim_plant.WERKS` | SAP (`EKPO-WERKS`) | `HIERARCHY_KEY` | Plant / campus code. Values: `US01`, `US02`, `US03`, `BE01`, `NL01`, `IN01`. |
| `KOSTL` | `COST_CENTER_ID` | `string` | FK -> `dim_cost_center.KOSTL` | SAP (`EKKN-KOSTL`) | `ENTITY_KEY` | Assigned cost center. |
| `HKONT` | `GL_ACCOUNT_ID` | `string` | FK -> `dim_gl_account.HKONT` | SAP (`EKKN-SAKTO`) | `ENTITY_KEY` | Assigned OPEX GL account (`USES_PO == True`). |
| `WBS_ELEMENT` | `WBS_ELEMENT` | `string` | — | SAP (`EKKN-PS_PSP_PNR`) | `HIERARCHY_KEY` | Assigned WBS element. |
| `LIFNR` | `VENDOR_ID` | `string` | — | SAP (`EKKO-LIFNR`) | `AUDIT_ATTRIBUTE` | Vendor account ID (e.g., `VND_CRO_IQVIA`). |
| `PO_CREATE_DATE` | `PO_CREATE_DATE` | `date` | — | SAP (`EKKO-AEDAT`) | `AUDIT_CUTOFF` | PO creation date (`PO_CREATE_DATE < EXPECTED_GR_MONTH`, hard-capped `<= 2025-12-31`). |
| `EXPECTED_GR_MONTH` | `EXPECTED_GR_MONTH` | `string` | — | Analytical (`EKET-EINDT`) | `TARGET_HORIZON_KEY` | Target Goods Receipt month (`YYYY-MM`; includes `2026-01`/`2026-02` open pipeline as of `2025-12-31`). |
| `SNAPSHOT_DATE` | `SNAPSHOT_DATE` | `date` | PK | Analytical | `PIT_CUTOFF_KEY` | Month-end observation cutoff date (`>= PO_CREATE_DATE`, hard-capped `<= 2025-12-31`). |
| `ORIGINAL_COMMITMENT_USD` | `ORIGINAL_COMMITMENT_USD` | `float64` | — | Analytical | `LEADING_INDICATOR` | Initial PO line item commitment value in USD. |
| `GR_AMOUNT_USD` | `GR_AMOUNT_USD` | `float64` | — | Analytical (`EKBE`) | `LEADING_INDICATOR` | Cumulative Goods Receipt amount in USD (`0 <= GR <= ORIGINAL`). |
| `INVOICED_AMOUNT_USD` | `INVOICED_AMOUNT_USD` | `float64` | — | Analytical (`EKBE`) | `LEADING_INDICATOR` | Cumulative invoiced amount in USD (`0 <= INVOICED <= GR`). |
| `OPEN_COMMITMENT_USD` | `OPEN_COMMITMENT_USD` | `float64` | — | SAP (`COOI-WKGBTR`) | `LEADING_INDICATOR` | Remaining open commitment USD (`ORIGINAL_COMMITMENT_USD - GR_AMOUNT_USD >= 0`). Primary leading ML feature. |
| `PO_STATUS` | `PO_STATUS` | `string` | — | Analytical | `FILTER_FLAG` | Lifecycle state at `SNAPSHOT_DATE`: `OPEN` (`GR == 0`), `PARTIALLY_RECEIVED` (`0 < GR < ORIGINAL`), `CLOSED` (`OPEN == 0`). |

### 4.10 `acdoca_actuals.csv` (`75,268` rows / `37,634` docs · `30` cols)
- **Category:** `FACT_TRANSACTIONAL_ECC` · **Primary Key:** `BUKRS, GJAHR, BELNR, BUZEI` · **SAP Inspiration:** `BKPF` / `BSEG`
- **Description:** Chronologically sorted double-entry GL posting lines (`FY2022–FY2025`, periods `1–16`) in classic SAP ECC field names with unsigned amounts qualified by `SHKZG` (`S` = Debit, `H` = Credit) and `BSCHL` (`40`/`50`).
- **Dec 2025 Open-Accrual Boundary:** Exactly **`14` legitimate Dec 2025 `ZA` accruals** sit with `STBLG == ""` and `BUDAT == "2025-12-31"` because their `AB` reversals belong to Jan 2026 (outside FY22–25). Always exclude `(GJAHR == 2025 & MONAT == 12)` from unreversed-accrual detection rules.

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BUKRS` | `COMPANY_CODE` | `string` | PK, FK -> `dim_company_code.BUKRS` | SAP (`BKPF-BUKRS`) | `HIERARCHY_KEY` | Company code. Values: `US10`, `EU20`, `AP30`. |
| `WERKS` | `PLANT_CODE` | `string` | FK -> `dim_plant.WERKS` | SAP (`BSEG-WERKS`) | `HIERARCHY_KEY` | Plant / campus code. Values: `US01`, `US02`, `US03`, `BE01`, `NL01`, `IN01`. |
| `GJAHR` | `FISCAL_YEAR` | `int64` | PK | SAP (`BKPF-GJAHR`) | `TIME_INDEX` | Fiscal year (`2022`–`2025`). |
| `MONAT` | `POSTING_PERIOD` | `int64` | — | SAP (`BKPF-MONAT`) | `TIME_INDEX` | Posting period (`1–12` regular; `13–16` year-end audit adjustments). |
| `BELNR` | `DOCUMENT_NUMBER` | `string` | PK | SAP (`BKPF-BELNR`) | `DOCUMENT_KEY` | 10-character chronological accounting document number per `(BUKRS, GJAHR)` starting at `0100000001`. |
| `BLDAT` | `DOCUMENT_DATE` | `date` | — | SAP (`BKPF-BLDAT`) | `AUDIT_ATTRIBUTE` | Document / invoice date (`BLDAT <= BUDAT`). |
| `BUDAT` | `POSTING_DATE` | `date` | — | SAP (`BKPF-BUDAT`) | `AUDIT_ATTRIBUTE` | Posting date (matches `GJAHR`/`MONAT` for `1–12`; `YYYY-12-31` for `13–16`). |
| `CPUDT` | `ENTRY_DATE` | `date` | — | SAP (`BKPF-CPUDT`) | `PIT_CUTOFF_KEY` | System entry date (`BUDAT <= CPUDT`); filter `CPUDT <= SNAPSHOT_DATE` for point-in-time backtesting. |
| `BLART` | `DOCUMENT_TYPE` | `string` | — | SAP (`BKPF-BLART`) | `AUDIT_ATTRIBUTE` | Document type: `KR` (Vendor Invoice), `PR` (Payroll -> `210000`), `SA` (Cash `113100` / Prepaid `140000` / Audit), `ZA` (Month-End Accrual -> `215000`), `AB` (Accrual Reversal). |
| `USNAM` | `POSTING_USER` | `string` | — | SAP (`BKPF-USNAM`) | `AUDIT_ATTRIBUTE` | Posting user / batch job ID (`USER001`–`USER039`, `BATCH_PAYROLL`, `BATCH_REV`, `BATCH_PREPAID`, `BATCH_AUDIT`). |
| `PRCTR` | `PROFIT_CENTER_ID` | `string` | FK -> `dim_profit_center.PRCTR` | SAP (`BSEG-PRCTR`) | `HIERARCHY_KEY` | Profit center. |
| `WBS_ELEMENT` | `WBS_ELEMENT` | `string` | — | SAP (`BSEG-PROJK`) | `HIERARCHY_KEY` | Clinical study WBS element (`WBS-*` or `NONE`). |
| `LIFNR` | `VENDOR_ID` | `string` | — | SAP (`BSEG-LIFNR`) | `AUDIT_ATTRIBUTE` | Vendor ID (`VND_*`), `INTERNAL_PAYROLL`, or `YEAR_END_AUDIT`. |
| `EBELN` | `PO_NUMBER` | `string` | FK (optional) -> `fact_po_commitments.EBELN` | SAP (`BSEG-EBELN`) | `AUDIT_ATTRIBUTE` | Purchase Order reference (`PO*`) populated on PO-backed `KR` invoices (`24,337 / 24,345` in `seed=42`; `8` blank `KR` rows are GAAP Special Items) and `100%` of `ZA` accruals (`564/564`) and `AB` reversals (`522/522`). Blank string `""` on non-PO postings, `650010` prepaid splits, and GAAP Special Items. |
| `WAERS` | `LOCAL_CURRENCY` | `string` | — | SAP (`BKPF-WAERS`) | `CURRENCY_KEY` | Local currency key. Values: `USD`, `EUR`, `INR`. |
| `DMBTR` | `AMOUNT_LOCAL_UNSIGNED` | `float64` | — | SAP (`BSEG-DMBTR`) | `AMOUNT_LOCAL_UNSIGNED` | Unsigned amount in local currency (qualify by `SHKZG`: `S` = `+`, `H` = `-`). |
| `KSL` | `AMOUNT_USD_REPORTED_UNSIGNED` | `float64` | — | Analytical (`ACDOCA-KSL`, S/4HANA carry-over) | `TARGET_REPORTED_UNSIGNED` | Unsigned amount in Reported Group USD translated at `ACTUAL_AVG` FX (S/4HANA carry-over; classic ECC `BSEG` has no `KSL`). |
| `KSL_BUDGET_FX` | `AMOUNT_USD_CONSTANT_FX_UNSIGNED` | `float64` | — | Analytical | `TARGET_OPERATIONAL_UNSIGNED` | Unsigned amount in Constant-Currency USD translated at annual `BUDGET_PLAN` FX. |
| `STBLG` | `REVERSAL_DOCUMENT_NUMBER` | `string` | — | SAP (`BKPF-STBLG`) | `AUDIT_ATTRIBUTE` | Reversal document `BELNR` (bidirectional `ZA <-> AB`; blank string `""` otherwise, including `14` legitimate Dec 2025 open accruals). |
| `STJAH` | `REVERSAL_FISCAL_YEAR` | `int64` | — | SAP (`BKPF-STJAH`) | `AUDIT_ATTRIBUTE` | Fiscal year of reversal document `STBLG` (`0` when `STBLG == ""`). |
| `SPECIAL_ITEM_FLAG` | `SPECIAL_ITEM_FLAG` | `string` | — | Analytical | `FILTER_FLAG` | J&J GAAP vs. Adjusted Operational tag. Values: `NONE`, `ACQUIRED_IPRD`, `RESTRUCTURING`, `MA_INTEGRATION`, `LITIGATION`. |
| `SERVICE_START_DATE` | `SERVICE_START_DATE` | `date` | — | Analytical | `AUDIT_ATTRIBUTE` | Service start: calendar month start on regular postings, calendar quarter start (`~90d` span) on `140000` prepaid releases, or `BLDAT` default (`0–2d`) on `ZA`/`AB`/Special Items/Periods `13–16`. |
| `SERVICE_END_DATE` | `SERVICE_END_DATE` | `date` | — | Analytical | `AUDIT_ATTRIBUTE` | Service end: calendar month end on regular postings, calendar quarter end on `140000` prepaid releases, or `BUDAT` default on `ZA`/`AB`/Special Items/Periods `13–16`. |
| `VERSION` | `VERSION` | `int64` | — | SAP (`ACDOCA-VERSN`) | `FILTER_FLAG` | SAP ledger version (`0` = Actuals). |
| `BUZEI` | `LINE_ITEM_NUMBER` | `int64` | PK | SAP (`BSEG-BUZEI`) | `DOCUMENT_KEY` | Line item number (`1` = OPEX P&L line, `2` = Balance Sheet offset line). Values: `1`, `2`. |
| `HKONT` | `GL_ACCOUNT_ID` | `string` | FK -> `dim_gl_account.HKONT` | SAP (`BSEG-HKONT`) | `ENTITY_KEY` | GL account (`610010`–`650040` on `BUZEI=1`; `113100`/`140000`/`210000`/`215000` on `BUZEI=2`). |
| `KOSTL` | `COST_CENTER_ID` | `string` | FK -> `dim_cost_center.KOSTL` | SAP (`BSEG-KOSTL`) | `ENTITY_KEY` | Cost center ID (populated on OPEX lines; blank string `""` on BS offset lines). |
| `RFAREA` | `FUNCTIONAL_AREA` | `string` | — | SAP (`BSEG-FKBER`) | `HIERARCHY_KEY` | Functional area on OPEX lines; empty string on BS offset lines. Values: `R&D`, `Commercial_SGA`, `Medical_Affairs`, `G&A`, `""` (empty string). |
| `SHKZG` | `DEBIT_CREDIT_FLAG` | `string` | — | SAP (`BSEG-SHKZG`) | `ACCOUNTING_SIGN` | Debit/Credit indicator (`S` = Debit, `H` = Credit). Values: `S`, `H`. |
| `BSCHL` | `POSTING_KEY` | `string` | — | SAP (`BSEG-BSCHL`) | `ACCOUNTING_SIGN` | SAP FI posting key (`40` = Debit matching `S`, `50` = Credit matching `H`). Values: `40`, `50`. |

### 4.11 `acdoca_universal_journal.csv` (`75,268` rows / `37,634` docs · `32` cols)
- **Category:** `FACT_TRANSACTIONAL_S4` · **Primary Key:** `RBUKRS, RYEAR, BELNR, DOCLN` · **SAP Inspiration:** S/4HANA `ACDOCA`
- **Description:** Exact row-for-row S/4HANA Universal Journal projection of `acdoca_actuals` with **signed** `HSL`, `KSL`, and `KSL_BUDGET_FX` so `SUM()` yields net balances directly without decoding `DRCRK`. Recommended primary analytics table.
- **Native S/4HANA Key Note:** Production S/4HANA `ACDOCA` keys on `(RCLNT, RLDNR, RBUKRS, GJAHR, BELNR, DOCLN)`. `(RBUKRS, RYEAR, BELNR, DOCLN)` is uniquely identifying here because `RLDNR` is always `"0L"` and `GJAHR == RYEAR`.

| Column | Readable Name | Type | Key / FK | ECC Equivalent | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `RBUKRS` | `COMPANY_CODE` | `string` | PK, FK -> `dim_company_code.BUKRS` | `BUKRS` | SAP (`ACDOCA-RBUKRS`) | `HIERARCHY_KEY` | Company code. Values: `US10`, `EU20`, `AP30`. |
| `WERKS` | `PLANT_CODE` | `string` | FK -> `dim_plant.WERKS` | `WERKS` | SAP (`ACDOCA-WERKS`) | `HIERARCHY_KEY` | Plant / campus code. Values: `US01`, `US02`, `US03`, `BE01`, `NL01`, `IN01`. |
| `GJAHR` | `FISCAL_YEAR` | `int64` | — | `GJAHR` | SAP (`ACDOCA-GJAHR`) | `TIME_INDEX` | Fiscal year key (identical to `RYEAR`). |
| `RYEAR` | `LEDGER_FISCAL_YEAR` | `int64` | PK | `GJAHR` | SAP (`ACDOCA-RYEAR`) | `TIME_INDEX` | General Ledger fiscal year (`2022`–`2025`). |
| `POPER` | `POSTING_PERIOD` | `int64` | — | `MONAT` | SAP (`ACDOCA-POPER`) | `TIME_INDEX` | Posting period (`1–12` regular; `13–16` special audit; kept integer for numeric filtering). |
| `BELNR` | `DOCUMENT_NUMBER` | `string` | PK | `BELNR` | SAP (`ACDOCA-BELNR`) | `DOCUMENT_KEY` | 10-character chronological document number per `(RBUKRS, RYEAR)`. |
| `DOCLN` | `LINE_ITEM_NUMBER` | `string` | PK | `BUZEI` | SAP (`ACDOCA-DOCLN`) | `DOCUMENT_KEY` | Zero-padded 6-char line item. Values: `000001`, `000002`. |
| `BLDAT` | `DOCUMENT_DATE` | `date` | — | `BLDAT` | SAP (`ACDOCA-BLDAT`) | `AUDIT_ATTRIBUTE` | Document / invoice date. |
| `BUDAT` | `POSTING_DATE` | `date` | — | `BUDAT` | SAP (`ACDOCA-BUDAT`) | `AUDIT_ATTRIBUTE` | Posting date. |
| `CPUDT` | `ENTRY_DATE` | `date` | — | `CPUDT` | SAP (`BKPF-CPUDT`) | `PIT_CUTOFF_KEY` | System entry date (filter `CPUDT <= SNAPSHOT_DATE` for leak-free PIT backtesting). |
| `BLART` | `DOCUMENT_TYPE` | `string` | — | `BLART` | SAP (`ACDOCA-BLART`) | `AUDIT_ATTRIBUTE` | Document type. Values: `KR`, `PR`, `SA`, `ZA`, `AB`. |
| `USNAM` | `POSTING_USER` | `string` | — | `USNAM` | SAP (`ACDOCA-USNAM`) | `AUDIT_ATTRIBUTE` | Posting user or batch job ID. |
| `RACCT` | `GL_ACCOUNT_ID` | `string` | FK -> `dim_gl_account.RACCT` | `HKONT` | SAP (`ACDOCA-RACCT`) | `ENTITY_KEY` | GL account number. |
| `RCNTR` | `COST_CENTER_ID` | `string` | FK -> `dim_cost_center.KOSTL` | `KOSTL` | SAP (`ACDOCA-RCNTR`) | `ENTITY_KEY` | Cost center ID (blank string `""` on BS lines). |
| `PRCTR` | `PROFIT_CENTER_ID` | `string` | FK -> `dim_profit_center.PRCTR` | `PRCTR` | SAP (`ACDOCA-PRCTR`) | `HIERARCHY_KEY` | Profit center. |
| `RFAREA` | `FUNCTIONAL_AREA` | `string` | — | `RFAREA` | SAP (`ACDOCA-RFAREA`) | `HIERARCHY_KEY` | Functional area (blank `""` on BS lines so `GROUP BY RFAREA` never nets to `$0`). Values: `R&D`, `Commercial_SGA`, `Medical_Affairs`, `G&A`, `""` (empty string). |
| `PS_PSP_PNR` | `WBS_ELEMENT` | `string` | — | `WBS_ELEMENT` | SAP (`ACDOCA-PS_PSP_PNR`) | `HIERARCHY_KEY` | Project / clinical study WBS element. |
| `LIFNR` | `VENDOR_ID` | `string` | — | `LIFNR` | SAP (`ACDOCA-LIFNR`) | `AUDIT_ATTRIBUTE` | Vendor ID. |
| `EBELN` | `PO_NUMBER` | `string` | FK (optional) -> `fact_po_commitments.EBELN` | `EBELN` | SAP (`ACDOCA-EBELN`) | `AUDIT_ATTRIBUTE` | Purchase Order reference (`PO*` on PO-backed `KR` & `100%` of `ZA`/`AB` docs; `""` on non-PO, `650010` prepaid splits & Special Items). |
| `RWCUR` | `LOCAL_CURRENCY` | `string` | — | `WAERS` | SAP (`ACDOCA-RWCUR`) | `CURRENCY_KEY` | Local currency key. Values: `USD`, `EUR`, `INR`. |
| `DRCRK` | `DEBIT_CREDIT_FLAG` | `string` | — | `SHKZG` | SAP (`ACDOCA-DRCRK`) | `ACCOUNTING_SIGN` | Debit/Credit indicator. Values: `S`, `H`. |
| `BSCHL` | `POSTING_KEY` | `string` | — | `BSCHL` | SAP (`ACDOCA-BSCHL`) | `ACCOUNTING_SIGN` | Posting key. Values: `40`, `50`. |
| `HSL` | `AMOUNT_LOCAL_SIGNED` | `float64` | — | Signed `DMBTR` | SAP (`ACDOCA-HSL`) | `AMOUNT_LOCAL_SIGNED` | **Signed** local currency amount (`+` Debit, `-` Credit; sums to `0.00` per doc). |
| `KSL` | `AMOUNT_USD_REPORTED_SIGNED` | `float64` | — | Signed `KSL` | SAP (`ACDOCA-KSL`) | `TARGET_REPORTED_SIGNED` | **Signed** Reported Group USD (`HSL * ACTUAL_AVG`). |
| `KSL_BUDGET_FX` | `AMOUNT_USD_CONSTANT_FX_SIGNED` | `float64` | — | Signed `KSL_BUDGET_FX` | Analytical | `TARGET_OPERATIONAL_SIGNED` | **Signed** Operational Constant-Currency USD (`HSL * BUDGET_PLAN`; primary ML target). |
| `STBLG` | `REVERSAL_DOCUMENT_NUMBER` | `string` | — | `STBLG` | SAP (`BKPF-STBLG`) | `AUDIT_ATTRIBUTE` | Reversal document `BELNR` (`""` if unreversed, incl. `14` legitimate Dec 2025 open accruals). |
| `STJAH` | `REVERSAL_FISCAL_YEAR` | `int64` | — | `STJAH` | SAP (`BKPF-STJAH`) | `AUDIT_ATTRIBUTE` | Reversal fiscal year (`0` if `STBLG == ""`). |
| `SPECIAL_ITEM_FLAG` | `SPECIAL_ITEM_FLAG` | `string` | — | `SPECIAL_ITEM_FLAG` | Analytical | `FILTER_FLAG` | GAAP vs. Adjusted Operational tag (filter `== "NONE"` for recurring run-rate OPEX). |
| `SERVICE_START_DATE` | `SERVICE_START_DATE` | `date` | — | `SERVICE_START_DATE` | Analytical | `AUDIT_ATTRIBUTE` | Service start (full month regular, calendar quarter on `140000` prepaids, `BLDAT` default on `ZA`/`AB`/Special/P13–16). |
| `SERVICE_END_DATE` | `SERVICE_END_DATE` | `date` | — | `SERVICE_END_DATE` | Analytical | `AUDIT_ATTRIBUTE` | Service end (full month regular, calendar quarter end on `140000` prepaids, `BUDAT` default on `ZA`/`AB`/Special/P13–16). |
| `RLDNR` | `LEDGER_ID` | `string` | — | *(S/4HANA only)* | SAP (`ACDOCA-RLDNR`) | `METADATA` | SAP General Ledger ID (`0L` = Leading Ledger). |
| `VERSION` | `VERSION` | `int64` | — | `VERSION` | SAP (`ACDOCA-VERSN`) | `FILTER_FLAG` | SAP version (`0` = Actuals). |

### 4.12 `plan_versions.csv` (`49,392` rows · `16` cols)
- **Category:** `FACT_PLAN_AND_BENCHMARK` · **Primary Key:** `GJAHR, MONAT, KOSTL, HKONT, CYCLE_NAME` · **SAP Inspiration:** `ACDOCP`
- **Description:** Point-in-time, leak-free J&J planning snapshots across `196` active `(KOSTL, HKONT)` pairs covering `VERSION=1` (`BP`, FY22–FY26) and `VERSION=2` human rolling forecasts (`Q1_RF`, `JU`, `Q3_RF`, `NU`, FY22–FY25).
- **Benchmark & Static Driver Guardrails:** (1) Keep `VERSION == 2` strictly as an **evaluation benchmark**, never an ML input feature. (2) `PLAN_FTE`, `PLAN_ENROLLED_PATIENTS`, and `PLAN_HCP_EVENTS` on `VERSION == 2` rows carry the **static annual `BP` baseline** (not mid-year re-forecasted headcount/volume). (3) `DMBTR_PLAN` and `KSL_BUDGET` equal `0.0` on non-milestone months for `MILESTONE` GL `620030`.

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `GJAHR` | `FISCAL_YEAR` | `int64` | PK | SAP (`ACDOCP-RYEAR`) | `TIME_INDEX` | Fiscal year (`2022`–`2026` for `BP`; `2022`–`2025` for RFs). |
| `MONAT` | `POSTING_PERIOD` | `int64` | PK | SAP (`ACDOCP-POPER`) | `TIME_INDEX` | Fiscal month (`1`–`12`). |
| `BUKRS` | `COMPANY_CODE` | `string` | FK -> `dim_company_code.BUKRS` | SAP (`ACDOCP-RBUKRS`) | `HIERARCHY_KEY` | Company code. Values: `US10`, `EU20`, `AP30`. |
| `WERKS` | `PLANT_CODE` | `string` | FK -> `dim_plant.WERKS` | SAP (`ACDOCP-WERKS`) | `HIERARCHY_KEY` | Plant / campus code. Values: `US01`, `US02`, `US03`, `BE01`, `NL01`, `IN01`. |
| `KOSTL` | `COST_CENTER_ID` | `string` | PK, FK -> `dim_cost_center.KOSTL` | SAP (`ACDOCP-RCNTR`) | `ENTITY_KEY` | Cost center ID. |
| `HKONT` | `GL_ACCOUNT_ID` | `string` | PK, FK -> `dim_gl_account.HKONT` | SAP (`ACDOCP-RACCT`) | `ENTITY_KEY` | OPEX GL account. |
| `VERSION` | `VERSION` | `int64` | — | SAP (`ACDOCP-CATEGORY`) | `FILTER_FLAG` | `1` = Annual Business Plan (`BP`, point-in-time safe baseline); `2` = Human Rolling Forecast (benchmark only). Values: `1`, `2`. |
| `CYCLE_NAME` | `CYCLE_NAME` | `string` | PK | Analytical | `BENCHMARK_KEY` | J&J planning cycle. Values: `BP`, `Q1_RF`, `JU`, `Q3_RF`, `NU`. |
| `SNAPSHOT_DATE` | `SNAPSHOT_DATE` | `date` | — | Analytical | `PIT_CUTOFF_KEY` | Hard information boundary date for the plan snapshot. |
| `WAERS` | `LOCAL_CURRENCY` | `string` | — | SAP (`ACDOCP-RWCUR`) | `CURRENCY_KEY` | Local currency key. Values: `USD`, `EUR`, `INR`. |
| `PLAN_FX_RATE` | `PLAN_FX_RATE` | `float64` | — | Analytical | `FX_DRIVER` | Constant annual `BUDGET_PLAN` FX rate locked Oct `yr-1`. |
| `DMBTR_PLAN` | `PLAN_AMOUNT_LOCAL` | `float64` | — | SAP (`ACDOCP-HSL`) | `PLAN_OR_BENCHMARK` | Planned/forecasted spend in local currency (closed RF months copy realized clean `HSL`; `0.0` on non-milestone months for `620030`). |
| `KSL_BUDGET` | `PLAN_AMOUNT_USD` | `float64` | — | SAP (`ACDOCP-KSL`) | `PLAN_OR_BENCHMARK` | Planned/forecasted spend in Constant-Currency USD (`round(DMBTR_PLAN * PLAN_FX_RATE, 2)`). |
| `PLAN_FTE` | `PLAN_FTE` | `float64` | — | Analytical | `SAFE_FUTURE_PLAN_FEATURE` | Static annual BP FTE baseline (unchanged on `VERSION=2` rows). |
| `PLAN_ENROLLED_PATIENTS` | `PLAN_ENROLLED_PATIENTS` | `float64` | — | Analytical | `SAFE_FUTURE_PLAN_FEATURE` | Static annual BP enrolled patients baseline (unchanged on `VERSION=2` rows). |
| `PLAN_HCP_EVENTS` | `PLAN_HCP_EVENTS` | `int64` | — | Analytical | `SAFE_FUTURE_PLAN_FEATURE` | Static annual BP HCP events baseline (unchanged on `VERSION=2` rows). |

### 4.13 `scenario_assumptions.csv` (`1,440` rows · `25` cols)
- **Category:** `FACT_SCENARIOS` · **Primary Key:** `SCENARIO_ID, GJAHR, MONAT, KOSTL` · **Source Inspiration:** FP&A Driver-Based What-If Simulation Input
- **Description:** Deterministic FY2026 (`GJAHR == 2026`, `SNAPSHOT_DATE == "2025-12-31"`) driver and FX perturbations across `5` executive scenarios, preserving `100%` feature parity with `fact_opex_drivers.csv` for direct `model.predict()` scoring.
- **FY26 Observed vs. Schema Domain Note:** `TRIAL_PHASE` shares the 5-value schema domain (`NONE`, `P1`, `P2`, `P3`, `FILING`), while only `NONE`, `P2`, `P3`, `FILING` appear in FY26 (`P1` graduated to `P2` in FY25). Similarly, `LOE_FLAG` has schema domain `{0, 1}` and observed value `0` across all `1,440` FY26 rows; `WAGE_INFLATION_RATE` is `0.033` throughout FY26.

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `SCENARIO_ID` | `SCENARIO_ID` | `string` | PK | Analytical | `SCENARIO_KEY` | What-If scenario ID (see §7 for exact perturbation rules). Values: `BASE`, `HIRING_FREEZE_Q2`, `PHASE3_DELAY_1Q`, `USD_STRENGTHEN_10PCT`, `LAUNCH_ACCEL`. |
| `SNAPSHOT_DATE` | `SNAPSHOT_DATE` | `date` | — | Analytical | `AUDIT_CUTOFF` | Planning-date anchor (`2025-12-31`). |
| `GJAHR` ... `WAGE_INFLATION_RATE` *(21 cols)* | *(matches §4.7)* | *(matches §4.7)* | PK (`GJAHR, MONAT, KOSTL`) | *(matches §4.7; `WERKS` = Analytical (`Plant Link`))* | *(matches §4.7)* | Exact 21-column schema of `fact_opex_drivers.csv` projected deterministically for FY2026 (`PLANNED_HIRES` = deterministic planned monthly hires; `PRODUCT_LAUNCH_FLAG = 1` in `Immunology` `Commercial_SGA` M3–M9 under `LAUNCH_ACCEL`). |
| `FX_EUR_USD` | `FX_EUR_USD` | `float64` | — | Analytical (`TCURR_ABSTRACT`) | `FX_DRIVER` | Scenario-specific FY26 EUR -> USD forward rate (`0.90x` of `BASE` under `USD_STRENGTHEN_10PCT`). |
| `FX_INR_USD` | `FX_INR_USD` | `float64` | — | Analytical (`TCURR_ABSTRACT`) | `FX_DRIVER` | Scenario-specific FY26 INR -> USD forward rate (`0.90x` of `BASE` under `USD_STRENGTHEN_10PCT`). |

### 4.14 `eval_anomaly_labels.csv` (`104` rows · `11` cols — Physically Isolated Ground Truth)
- **Category:** `EVALUATION_ONLY` · **Primary Key:** `BUKRS, GJAHR, BELNR, BUZEI` · **Source Inspiration:** Out-of-Band Audit Benchmark
- **Description:** Physically isolated ground-truth labels for scoring unsupervised anomaly detection Precision, Recall, and F1. **NEVER join into training or inference features.**
- **Exact Anomaly Rate (`seed=42`):** `104` labeled anomalies = **`0.2763%` (`0.28%`) of `37,634` documents** and **`0.1382%` (`0.14%`) of `75,268` journal lines**: `16` `TRIAL_PAUSE_DROP`, `24` `Q4_BUDGET_FLUSH`, `28` `UNREVERSED_ACCRUAL`, and `36` `DUPLICATE_INVOICE` (`27` same-month + `9` cross-month). Remember to exclude Dec 2025 (`14` legitimate open `ZA` accruals) from unreversed-accrual detection rules.

| Column | Readable Name | Type | Key / FK | Provenance / SAP Ref | ML Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `BUKRS` | `COMPANY_CODE` | `string` | PK, FK -> `acdoca_actuals.BUKRS` | SAP (`BKPF-BUKRS`) | `EVAL_KEY` | Company code of anomalous document. |
| `GJAHR` | `FISCAL_YEAR` | `int64` | PK, FK -> `acdoca_actuals.GJAHR` | SAP (`BKPF-GJAHR`) | `EVAL_KEY` | Fiscal year (`2022`–`2025`). |
| `MONAT` | `POSTING_PERIOD` | `int64` | — | SAP (`BKPF-MONAT`) | `EVAL_METADATA` | Posting month (`1`–`12`). |
| `BELNR` | `DOCUMENT_NUMBER` | `string` | PK, FK -> `acdoca_actuals.BELNR` | SAP (`BKPF-BELNR`) | `EVAL_KEY` | Chronologically remapped `BELNR` matching `acdoca_actuals` / `acdoca_universal_journal`. |
| `BUZEI` | `LINE_ITEM_NUMBER` | `int64` | PK, FK -> `acdoca_actuals.BUZEI` | SAP (`BSEG-BUZEI`) | `EVAL_KEY` | Line item number (always `1` = OPEX line). |
| `KOSTL` | `COST_CENTER_ID` | `string` | FK -> `dim_cost_center.KOSTL` | SAP (`BSEG-KOSTL`) | `EVAL_METADATA` | Impacted cost center. |
| `HKONT` | `GL_ACCOUNT_ID` | `string` | FK -> `dim_gl_account.HKONT` | SAP (`BSEG-HKONT`) | `EVAL_METADATA` | Impacted OPEX GL account. |
| `ANOMALY_TAG` | `ANOMALY_TAG` | `string` | — | Evaluation Label | `EVAL_TARGET_ONLY` | Ground-truth class. Values: `DUPLICATE_INVOICE`, `UNREVERSED_ACCRUAL`, `Q4_BUDGET_FLUSH`, `TRIAL_PAUSE_DROP`. |
| `GRANULARITY` | `GRANULARITY` | `string` | — | Evaluation Label | `EVAL_METADATA` | Detection level (`DOCUMENT` for duplicates/accruals; `CELL_MONTH` for trial pauses/Q4 flushes). Values: `DOCUMENT`, `CELL_MONTH`. |
| `ESTIMATED_IMPACT_USD` | `ESTIMATED_IMPACT_USD` | `float64` | — | Evaluation Label | `EVAL_METADATA` | Signed USD P&L distortion (`-` for `TRIAL_PAUSE_DROP`; `+` for overstatements). |
| `DESCRIPTION` | `DESCRIPTION` | `string` | — | Evaluation Label | `EVAL_METADATA` | Human-readable audit description (distinguishes same-month vs. cross-month duplicates). |

---

## 5. CSV Ingestion, Nullability & Canonical Recurring OPEX Query

### 5.1 Mandatory `dtype=str` & Blank-String Spec

Always pass `dtype=str` on leading-zero / key columns and normalize empty strings so pandas does not strip leading zeros or cast blank SAP fields to `NaN`:

- **String-typed columns (`15`):** `BELNR`, `STBLG`, `DOCLN`, `HKONT`, `RACCT`, `EBELP`, `BSCHL`, `BUKRS`, `RBUKRS`, `WERKS`, `KOSTL`, `RCNTR`, `RFAREA`, `EBELN`, `USNAM`
- **Blank-string columns requiring `.fillna("")` (`5`):** `STBLG`, `KOSTL`, `RCNTR`, `RFAREA`, `EBELN`

```python
import pandas as pd

STR_COLS = {
    c: str for c in [
        "BELNR", "STBLG", "DOCLN", "HKONT", "RACCT", "EBELP", "BSCHL",
        "BUKRS", "RBUKRS", "WERKS", "KOSTL", "RCNTR", "RFAREA", "EBELN", "USNAM",
    ]
}
BLANK_STR_COLS = ["STBLG", "KOSTL", "RCNTR", "RFAREA", "EBELN"]

def read_pharma_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=STR_COLS, low_memory=False)
    for col in BLANK_STR_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df
```

### 5.2 Canonical 4-Condition Recurring OPEX Filter

```python
dim_gl = read_pharma_csv("./data/raw/pharma_v22/dim_gl_account.csv")
opex_gls = set(dim_gl.loc[dim_gl["ACCOUNT_TYPE"] == "OPEX", "HKONT"])

def get_recurring_opex_panel(uj: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    clean = uj[
        (uj["RACCT"].isin(opex_gls))            # 1. P&L OPEX accounts (610010-650040)
        & (uj["POPER"] <= 12)                   # 2. Regular fiscal periods only (exclude 13-16)
        & (uj["SPECIAL_ITEM_FLAG"] == "NONE")   # 3. Adjusted Operational (exclude GAAP specials)
        & (uj["CPUDT"] <= snapshot_date)        # 4. Point-in-time entry-date cutoff
    ]
    return clean.groupby(["RYEAR", "POPER", "RBUKRS", "WERKS", "RCNTR", "RACCT"], as_index=False)[
        ["HSL", "KSL", "KSL_BUDGET_FX"]
    ].sum()
```

### 5.3 Readable Column Mapping & Naming Contract

To make exploratory analysis and stakeholder charts easier to read without duplicating raw CSVs:
- **Core `src/opex_forecast/` pipelines** keep `load_pharma_table(..., readable=False)` (the default) and operate on canonical SAP column names.
- **Interactive `notebooks/` and charts** may opt into `UPPER_SNAKE` readable names via `load_pharma_table(table_name, readable=True)` or `rename_to_readable(df, table_name)`, and convert back via table-scoped `rename_to_sap(df, table_name)` from `src/opex_forecast/column_map.py`.
- See [00_sap_column_reference.ipynb](../notebooks/00_sap_column_reference.ipynb) for the interactive lookup guide and [sap_column_crosswalk.csv](sap_column_crosswalk.csv) for the flat 206-row mapping table.

---

## 6. Anti-Leakage & Modeling Guardrails Checklist

1. Apply all 4 canonical recurring OPEX filter conditions (`§5.2`) before monthly aggregation.
2. Filter `fact_po_commitments.csv` to a **single `SNAPSHOT_DATE == forecast_origin_month_end`** before aggregating `OPEN_COMMITMENT_USD` by `(KOSTL, HKONT, EXPECTED_GR_MONTH)`.
3. At forecast origin $t$, use only realized `fact_opex_drivers.csv` with `(GJAHR, MONAT) <= t` (lags) paired with point-in-time `plan_drivers.csv` for future horizons $t+h$.
4. Keep `plan_versions.csv` `VERSION == 2` (`Q1_RF`, `JU`, `Q3_RF`, `NU`) strictly as open-month evaluation benchmarks—never as ML features—and remember their `PLAN_FTE`/`PLAN_ENROLLED_PATIENTS`/`PLAN_HCP_EVENTS` reflect the static annual `BP` baseline.
5. Never join `eval_anomaly_labels.csv` into training or inference pipelines.
6. Exclude Dec 2025 (`GJAHR == 2025 & MONAT == 12`) from unreversed-accrual detection rules to avoid flagging the `14` legitimate year-end open `ZA` accruals.
7. Never use `dim_cost_center.COST_SCALE` as an ML feature (`DO_NOT_USE_IN_MODEL`).

---

## 7. Quick Reference: Code Enumerations, Accounting Conventions & What-If Scenarios

| Field | Allowed Values & Exact Business Semantics |
| :--- | :--- |
| **`BLART`** (Document Type) | `KR` = Vendor Invoice (offset `215000`, `USER001`–`USER039`); `PR` = Biweekly Payroll (`BATCH_PAYROLL`, offset `210000` Payroll Clearing); `SA` = Non-PO Cash (`113100`), Quarterly Prepaid Release (`BATCH_PREPAID`, `140000`, `~90d` service span), or Periods 13–16 Audit (`BATCH_AUDIT`, `215000`); `ZA` = Month-End Accrual (`215000`, `USER*`); `AB` = Auto-Reversal (`BATCH_REV`, `215000`, exact local-currency `DMBTR` clearing of paired `ZA`). |
| **`SHKZG` / `BSCHL`** | `S` = Debit (`BSCHL = "40"`, `+` signed `HSL`/`KSL`); `H` = Credit (`BSCHL = "50"`, `-` signed `HSL`/`KSL`). |
| **`COST_BEHAVIOR`** | `FIXED` (`620070`, `650030` rent/leases), `STEP_FIXED` (`610010`, `610020`, `650010` capacity tiers `ceil(FTE/25)`), `VARIABLE` (CRO, site/patient, lab, incentives, T&E, bonus), `DISCRETIONARY` (HCP, DTC marketing, HEOR, grants, IT consulting, legal/audit), `MILESTONE` (`620030`, `620060` lump-sum clinical/regulatory milestones). |
| **`SPECIAL_ITEM_FLAG`** | `NONE` (recurring operational OPEX), `ACQUIRED_IPRD` (`3` deals), `RESTRUCTURING` (`2` programs), `MA_INTEGRATION` (`1` program), `LITIGATION` (`2` settlements). |
| **`CYCLE_NAME`** | `BP` (`{yr-1}-10-15`, `0+12`), `Q1_RF` (`{yr}-04-10`, `3+9`), `JU` (`{yr}-06-15`, `5+7`), `Q3_RF` (`{yr}-10-10`, `9+3`), `NU` (`{yr}-11-15`, `10+2`). |
| **`SCENARIO_ID`** *(FY2026 What-If Perturbations from Dec 2025 State)* | • **`BASE`:** Deterministic projection from `2025-12-31` realized state + FY26 `FORWARD_CURVE` FX + scheduled milestones (`CC1010` M5 `DB_LOCK`, `CC2010` M6 `FPI`, `CC1030` M7 `FDA_FILING`, `CC3010` M10 `DB_LOCK`).<br>• **`HIRING_FREEZE_Q2`:** Sets `PLANNED_HIRES = 0` from April 2026 (`M4–M12`) and applies smooth `0.9%/month` natural attrition so `ACTIVE_FTE` and `FIELD_SALES_REPS` decay gradually without a cliff.<br>• **`PHASE3_DELAY_1Q`:** Holds Phase III (`TRIAL_PHASE == "P3"`) `ENROLLED_PATIENTS` and `ACTIVE_SITES` flat near March (`M3`) levels during Q2 (`M4–M6`, realistic enrollment pause retaining existing patients/sites), resumes the 3-month shifted ramp in `M7–M12`, and pushes scheduled `M4–M12` milestones out by `+3` months (`CC1010` M5->M8, `CC2010` M6->M9, `CC1030` M7->M10, `CC3010` M10->FY27).<br>• **`USD_STRENGTHEN_10PCT`:** Keeps all operational drivers identical to `BASE` while shocking `FX_EUR_USD` and `FX_INR_USD` by `0.90x` (`-10%`) to demonstrate Reported USD vs. Constant-Currency Operational divergence.<br>• **`LAUNCH_ACCEL`:** Activates `PRODUCT_LAUNCH_FLAG = 1` and boosts `HCP_EVENTS_COUNT` by `+30%` across `Immunology` `Commercial_SGA` cost centers during `M3–M9`. |