"""
database/schema_graph.py
─────────────────────────
Introspects retail.db and mimic.db at runtime and produces a structured
SchemaGraph that the prompt_engine uses to inject dual-DB schema context
directly into SLM prompts — with NO vector search or RAG.

Key outputs
-----------
  SchemaGraph.to_prompt_string(role)  → compact, LLM-readable schema block
  SchemaGraph.get_fk_graph()          → human-readable foreign-key paths
  SchemaGraph.get_sample_values(tbl)  → representative values per column
  SchemaGraph.get_join_paths(db1, db2)→ cross-DB join via bridge table
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RETAIL_DB = DATA_DIR / "retail.db"
MIMIC_DB = DATA_DIR / "mimic.db"

# ── RBAC-aware column masks (columns hidden per role) ────────────────────────
# This mirrors the full RBAC matrix but at the schema-display level.
# The actual data masking happens in engine/rbac_guardrail.py.
_MASKED_COLUMNS: dict[str, dict[str, list[str]]] = {
    "Regional_Sales_Manager": {
        "retail": ["customer_id"],       # can see aggregates, not individual IDs
    },
    "Customer_Support_Lead": {
        "retail": ["unit_price", "line_total"],  # no financial fields
    },
    "Clinical_Analyst": {
        "mimic": ["dod"],               # cannot see date-of-death
    },
    "Data_Engineer": {
        "retail": [],
        "mimic": [],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnInfo:
    name: str
    dtype: str
    is_pk: bool = False
    is_fk: bool = False
    fk_ref: Optional[str] = None      # "other_table.other_column"
    nullable: bool = True
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class TableInfo:
    name: str
    db_label: str                      # "retail" | "mimic"
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: int = 0
    description: str = ""


@dataclass
class ForeignKey:
    from_table: str
    from_col: str
    to_table: str
    to_col: str
    db_label: str


# ══════════════════════════════════════════════════════════════════════════════
#  Human-readable table descriptions (injected into prompts)
# ══════════════════════════════════════════════════════════════════════════════

_TABLE_DESCRIPTIONS: dict[str, str] = {
    # ── Retail ────────────────────────────────────────────────────────────────
    "customers": "Retail customers with country and business segment classification.",
    "products": "Product catalogue with stock codes, descriptions, categories, and prices.",
    "invoices": "Transaction-level invoice header; is_cancelled=1 means a return/refund.",
    "invoice_items": "Line items per invoice; quantity × unit_price = line_total (revenue).",
    "patient_customer_bridge": "Cross-DB identity link: maps retail CustomerID to MIMIC subject_id.",
    # ── MIMIC hosp ────────────────────────────────────────────────────────────
    "patients": "De-identified patient demographics (100 patients). anchor_age is age at anchor_year.",
    "admissions": "Hospital admission events. hospital_expire_flag=1 means in-hospital death.",
    "diagnoses_icd": "ICD-9/10 diagnosis codes assigned during each admission (seq_num=1 is primary).",
    "labevents": "Laboratory test results ordered during hospitalisations. flag='abnormal' is clinically significant.",
    "prescriptions": "Medication orders with drug name, dose, route, and administration window.",
    "procedures_icd": "ICD-9/10 procedure codes performed during each admission.",
    "icustays": "ICU admission records. los = length of stay in days.",
    "chartevents": "High-frequency physiological measurements (BP, HR, SpO2, GCS, etc.) from ICU monitors.",
    "inputevents": "Fluids and medications administered via IV in the ICU.",
    "outputevents": "Urine, drain, and other output measurements from ICU patients.",
    "transfers": "Patient movement events between care units within the hospital.",
    "services": "Medical services responsible for patient care (e.g., MED, SURG, NMED).",
    "drgcodes": "Diagnosis-Related Group codes used for hospital billing classification.",
    "omr": "Online Medical Records: outpatient measurements (weight, BMI, blood pressure).",
    "emar": "Electronic Medication Administration Record — nurse-charted actual drug deliveries.",
    "pharmacy": "Pharmacy dispensing records with drug formulations and dosing.",
    "poe": "Provider Order Entry — all clinician orders (medications, labs, procedures).",
    "d_labitems": "Reference dictionary: maps lab itemid to test name and fluid source.",
    "d_icd_diagnoses": "Reference dictionary: maps ICD code to long diagnosis description.",
    "d_icd_procedures": "Reference dictionary: maps ICD procedure code to long description.",
    "d_items": "Reference dictionary: maps ICU chart itemid to measurement name and unit.",
    "microbiologyevents": "Microbiology culture results (organism, sensitivity, test type).",
    "hcpcsevents": "HCPCS billing codes for procedures and supplies.",
    "datetimeevents": "Date/time-valued ICU clinical events (e.g., intubation time).",
    "procedureevents": "Structured ICU procedure records (e.g., arterial line, intubation).",
    "ingredientevents": "Ingredient-level detail for multi-component IV infusions.",
    "caregiver": "ICU caregiver identifiers associated with chart entries.",
    "provider": "De-identified provider identifiers referenced in orders and EMAR.",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SchemaGraph
# ══════════════════════════════════════════════════════════════════════════════

class SchemaGraph:
    """
    Extracts and caches full schema metadata for both databases.
    Provides methods to render the schema as structured prompt text.
    """

    def __init__(self) -> None:
        self.retail_tables: dict[str, TableInfo] = {}
        self.mimic_tables: dict[str, TableInfo] = {}
        self.foreign_keys: list[ForeignKey] = []
        self._loaded = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> "SchemaGraph":
        """Introspect both SQLite databases and populate internal structures."""
        if not RETAIL_DB.exists():
            raise FileNotFoundError(
                f"retail.db not found at {RETAIL_DB}.\n"
                "Run:  python database/db_setup.py"
            )
        if not MIMIC_DB.exists():
            raise FileNotFoundError(
                f"mimic.db not found at {MIMIC_DB}.\n"
                "Run:  python database/db_setup.py"
            )

        self.retail_tables = self._introspect_db(RETAIL_DB, "retail")
        self.mimic_tables = self._introspect_db(MIMIC_DB, "mimic")
        self._loaded = True
        return self

    def _introspect_db(
        self, db_path: Path, label: str
    ) -> dict[str, TableInfo]:
        tables: dict[str, TableInfo] = {}
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            table_names: list[str] = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            for tname in table_names:
                ti = self._introspect_table(conn, tname, label)
                tables[tname] = ti
        return tables

    def _introspect_table(
        self, conn: sqlite3.Connection, tname: str, label: str
    ) -> TableInfo:
        # Row count
        row_count: int = conn.execute(f"SELECT COUNT(*) FROM [{tname}]").fetchone()[0]

        # Column info (PRAGMA table_info)
        pk_cols: set[str] = set()
        columns_raw = conn.execute(f"PRAGMA table_info([{tname}])").fetchall()
        col_map: dict[str, ColumnInfo] = {}
        for col in columns_raw:
            name = col["name"]
            dtype = col["type"] or "TEXT"
            is_pk = bool(col["pk"])
            if is_pk:
                pk_cols.add(name)
            col_map[name] = ColumnInfo(
                name=name,
                dtype=dtype,
                is_pk=is_pk,
                nullable=not col["notnull"],
            )

        # Foreign key info (PRAGMA foreign_key_list)
        fk_rows = conn.execute(f"PRAGMA foreign_key_list([{tname}])").fetchall()
        for fk in fk_rows:
            from_col = fk["from"]
            to_table = fk["table"]
            to_col = fk["to"]
            if from_col in col_map:
                col_map[from_col].is_fk = True
                col_map[from_col].fk_ref = f"{to_table}.{to_col}"
            self.foreign_keys.append(
                ForeignKey(
                    from_table=tname,
                    from_col=from_col,
                    to_table=to_table,
                    to_col=to_col,
                    db_label=label,
                )
            )

        # Sample values (top 3 distinct non-null values per column)
        for col_name, ci in col_map.items():
            try:
                rows = conn.execute(
                    f"SELECT DISTINCT [{col_name}] FROM [{tname}] "
                    f"WHERE [{col_name}] IS NOT NULL LIMIT 3"
                ).fetchall()
                ci.sample_values = [r[0] for r in rows]
            except Exception:
                ci.sample_values = []

        return TableInfo(
            name=tname,
            db_label=label,
            columns=list(col_map.values()),
            row_count=row_count,
            description=_TABLE_DESCRIPTIONS.get(tname, ""),
        )

    # ── Prompt rendering ──────────────────────────────────────────────────────

    def to_prompt_string(
        self,
        role: str = "Executive",
        include_samples: bool = True,
        include_row_counts: bool = True,
    ) -> str:
        """
        Render both DB schemas as a compact, structured prompt block.
        Columns masked by RBAC for the given role are excluded.
        """
        self._assert_loaded()
        masked = _MASKED_COLUMNS.get(role, {})
        lines: list[str] = []

        # ── DB1: Retail ───────────────────────────────────────────────────────
        lines.append("=== DATABASE 1: Online Retail (retail.db) ===")
        lines.append("Domain: UK-based e-commerce transactions (Dec 2010 – Dec 2011)")
        lines.append("Engine: SQLite\n")

        for tname, ti in self.retail_tables.items():
            lines.extend(
                self._render_table(ti, masked.get("retail", []), include_samples, include_row_counts)
            )

        # ── DB2: MIMIC ────────────────────────────────────────────────────────
        lines.append("\n=== DATABASE 2: MIMIC-IV Clinical Demo (mimic.db) ===")
        lines.append("Domain: De-identified EHR for 100 ICU patients (Beth Israel Deaconess Medical Center)")
        lines.append("Engine: SQLite\n")

        # Surface the most query-relevant tables first
        priority_tables = [
            "patients", "admissions", "diagnoses_icd", "labevents",
            "prescriptions", "icustays", "chartevents", "d_labitems",
            "d_icd_diagnoses", "patient_customer_bridge",
        ]
        remaining = [t for t in self.mimic_tables if t not in priority_tables]
        ordered = priority_tables + remaining

        for tname in ordered:
            if tname not in self.mimic_tables:
                continue
            ti = self.mimic_tables[tname]
            lines.extend(
                self._render_table(ti, masked.get("mimic", []), include_samples, include_row_counts)
            )

        # ── Cross-DB join path ────────────────────────────────────────────────
        lines.append("\n=== CROSS-DATABASE JOIN PATH ===")
        lines.append(
            "retail.customers.customer_id "
            "→ patient_customer_bridge.customer_id "
            "→ patient_customer_bridge.subject_id "
            "→ mimic.patients.subject_id"
        )
        lines.append("Bridge table exists in BOTH databases.")
        lines.append(
            "To cross-join: query each DB separately, then merge on customer_id / subject_id in Python."
        )

        return "\n".join(lines)

    def _render_table(
        self,
        ti: TableInfo,
        masked_cols: list[str],
        include_samples: bool,
        include_row_counts: bool,
    ) -> list[str]:
        """Render a single table as a compact schema block."""
        lines: list[str] = []
        rc_str = f"  [{ti.row_count:,} rows]" if include_row_counts else ""
        lines.append(f"TABLE: {ti.name}{rc_str}")
        if ti.description:
            lines.append(f"  -- {ti.description}")
        for ci in ti.columns:
            if ci.name in masked_cols:
                lines.append(f"  {ci.name}  [MASKED — role lacks permission]")
                continue
            flags: list[str] = []
            if ci.is_pk:
                flags.append("PK")
            if ci.is_fk and ci.fk_ref:
                flags.append(f"FK→{ci.fk_ref}")
            if not ci.nullable:
                flags.append("NOT NULL")
            flag_str = f"  ({', '.join(flags)})" if flags else ""
            sample_str = ""
            if include_samples and ci.sample_values:
                sample_str = f"  e.g. {ci.sample_values}"
            lines.append(f"  {ci.name}  {ci.dtype}{flag_str}{sample_str}")
        lines.append("")
        return lines

    # ── FK graph ──────────────────────────────────────────────────────────────

    def get_fk_graph(self) -> str:
        """Return a human-readable foreign-key relationship summary."""
        self._assert_loaded()
        lines = ["=== FOREIGN KEY RELATIONSHIPS ===\n"]
        by_db: dict[str, list[ForeignKey]] = {}
        for fk in self.foreign_keys:
            by_db.setdefault(fk.db_label, []).append(fk)

        for db_label, fks in sorted(by_db.items()):
            lines.append(f"[{db_label.upper()}]")
            for fk in fks:
                lines.append(
                    f"  {fk.from_table}.{fk.from_col}  →  {fk.to_table}.{fk.to_col}"
                )
            lines.append("")
        return "\n".join(lines)

    # ── Table list (for UI) ───────────────────────────────────────────────────

    def get_table_list(self, db: str = "retail") -> list[dict]:
        """Return a list of table dicts for the Streamlit schema explorer."""
        self._assert_loaded()
        src = self.retail_tables if db == "retail" else self.mimic_tables
        return [
            {
                "table": ti.name,
                "rows": ti.row_count,
                "columns": len(ti.columns),
                "description": ti.description,
                "column_names": [c.name for c in ti.columns],
            }
            for ti in src.values()
        ]

    def get_all_tables(self) -> dict[str, dict[str, TableInfo]]:
        """Return both DB table dicts."""
        self._assert_loaded()
        return {"retail": self.retail_tables, "mimic": self.mimic_tables}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call SchemaGraph.load() before using this method.")

    def summary_stats(self) -> dict:
        """Quick stats dict for display."""
        self._assert_loaded()
        return {
            "retail_tables": len(self.retail_tables),
            "retail_rows": sum(t.row_count for t in self.retail_tables.values()),
            "mimic_tables": len(self.mimic_tables),
            "mimic_rows": sum(t.row_count for t in self.mimic_tables.values()),
            "foreign_keys": len(self.foreign_keys),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — quick inspection
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print schema graph for both DBs.")
    parser.add_argument(
        "--role",
        default="Executive",
        choices=["Executive", "Regional_Sales_Manager", "Customer_Support_Lead",
                 "Clinical_Analyst", "Data_Engineer"],
        help="Role to apply RBAC column masking for.",
    )
    parser.add_argument("--fk", action="store_true", help="Show FK graph only.")
    args = parser.parse_args()

    sg = SchemaGraph().load()
    stats = sg.summary_stats()
    print(f"\nSchema loaded: {stats['retail_tables']} retail tables "
          f"({stats['retail_rows']:,} rows)  |  "
          f"{stats['mimic_tables']} MIMIC tables "
          f"({stats['mimic_rows']:,} rows)  |  "
          f"{stats['foreign_keys']} FK relationships\n")

    if args.fk:
        print(sg.get_fk_graph())
    else:
        print(sg.to_prompt_string(role=args.role))
