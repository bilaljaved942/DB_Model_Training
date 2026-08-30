"""
database/db_setup.py
────────────────────
Initialises two SQLite databases from real source files:

  retail.db  ← Online Retail.xlsx  (UCI, 541 909 rows)
  mimic.db   ← MIMIC-IV demo CSVs  (100 patients, 22 hosp + 9 icu tables)

The Online Retail flat file is normalised into 4 relational tables:
  customers · products · invoices · invoice_items

MIMIC-IV tables are loaded as-is (they are already relational).

A cross-database bridge table is written into *both* databases so that
executors can join retail customers to MIMIC patients by subject_id.

Usage
-----
    python database/db_setup.py          # creates data/retail.db & data/mimic.db
    python database/db_setup.py --force  # drops & recreates if DBs already exist
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RETAIL_XLSX = DATA_DIR / "online-retail-database" / "Online Retail.xlsx"
MIMIC_DIR = DATA_DIR / "mimic-iv-clinical-database"
HOSP_DIR = MIMIC_DIR / "hosp"
ICU_DIR = MIMIC_DIR / "icu"

RETAIL_DB = DATA_DIR / "retail.db"
MIMIC_DB = DATA_DIR / "mimic.db"

# ── Colour helpers ────────────────────────────────────────────────────────────
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"{_GREEN}✔{_RESET}  {msg}")


def _info(msg: str) -> None:
    print(f"{_YELLOW}→{_RESET}  {msg}")


def _err(msg: str) -> None:
    print(f"{_RED}✘{_RESET}  {msg}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
#  RETAIL DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def _load_retail_raw() -> pd.DataFrame:
    """Read the Online Retail Excel file and return a clean DataFrame."""
    _info(f"Reading {RETAIL_XLSX.name} …  (23 MB — this takes ~20s)")
    df = pd.read_excel(RETAIL_XLSX, dtype={"CustomerID": str, "InvoiceNo": str})
    _ok(f"Loaded {len(df):,} raw rows")

    # ── Basic cleaning ────────────────────────────────────────────────────────
    df["InvoiceNo"] = df["InvoiceNo"].str.strip()
    df["StockCode"] = df["StockCode"].str.strip()
    df["CustomerID"] = df["CustomerID"].str.strip()
    df["Description"] = df["Description"].fillna("UNKNOWN").str.strip().str.upper()
    df["Country"] = df["Country"].str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").fillna(0.0)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Drop rows where InvoiceDate is null (data quality)
    df = df.dropna(subset=["InvoiceDate"])
    _ok(f"After cleaning: {len(df):,} rows")
    return df


def _derive_segment(country: str) -> str:
    """Simple rule-based customer segment by country."""
    uk_countries = {"United Kingdom"}
    eu_countries = {
        "Germany", "France", "Belgium", "Netherlands", "Spain",
        "Portugal", "Switzerland", "Norway", "Sweden", "Denmark",
        "Finland", "Italy", "Austria", "Poland", "Czech Republic",
        "Greece", "Cyprus", "Malta",
    }
    if country in uk_countries:
        return "Domestic_UK"
    if country in eu_countries:
        return "EU_Wholesale"
    return "International"


def _build_retail_db(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Normalise flat DataFrame into 4 relational tables and write to SQLite."""
    cur = conn.cursor()

    # ── customers ─────────────────────────────────────────────────────────────
    _info("Building customers table …")
    customers_df = (
        df[df["CustomerID"].notna()]
        .groupby("CustomerID")
        .agg(country=("Country", "first"))
        .reset_index()
    )
    customers_df.rename(columns={"CustomerID": "customer_id"}, inplace=True)
    customers_df["segment"] = customers_df["country"].apply(_derive_segment)
    customers_df["registration_date"] = "2010-12-01"  # earliest date in dataset

    cur.execute("DROP TABLE IF EXISTS customers")
    cur.execute("""
        CREATE TABLE customers (
            customer_id       TEXT PRIMARY KEY,
            country           TEXT NOT NULL,
            segment           TEXT NOT NULL,
            registration_date TEXT
        )
    """)
    customers_df.to_sql("customers", conn, if_exists="append", index=False)
    _ok(f"  customers: {len(customers_df):,} rows")

    # ── products ──────────────────────────────────────────────────────────────
    _info("Building products table …")
    products_df = (
        df.groupby("StockCode")
        .agg(
            description=("Description", lambda x: x.mode()[0] if not x.mode().empty else "UNKNOWN"),
            unit_price=("UnitPrice", "median"),
        )
        .reset_index()
    )
    products_df.rename(columns={"StockCode": "stock_code"}, inplace=True)

    # Infer category from description keywords
    def _category(desc: str) -> str:
        desc = desc.upper()
        if any(k in desc for k in ("HEART", "LOVE", "ROSE", "FLOWER")):
            return "Gifts_Romantic"
        if any(k in desc for k in ("CHRISTMAS", "XMAS", "SANTA", "HOLLY")):
            return "Seasonal_Christmas"
        if any(k in desc for k in ("BAG", "POUCH", "HOLDER", "BOX", "TIN")):
            return "Storage_Bags"
        if any(k in desc for k in ("CANDLE", "LIGHT", "LAMP")):
            return "Lighting"
        if any(k in desc for k in ("CARD", "SIGN", "PRINT", "FRAME")):
            return "Stationery_Decor"
        if any(k in desc for k in ("LUNCH", "KITCHEN", "MUG", "CUP", "BOTTLE")):
            return "Kitchenware"
        return "General_Gifts"

    products_df["category"] = products_df["description"].apply(_category)

    cur.execute("DROP TABLE IF EXISTS products")
    cur.execute("""
        CREATE TABLE products (
            stock_code  TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            category    TEXT NOT NULL,
            unit_price  REAL NOT NULL
        )
    """)
    products_df.to_sql("products", conn, if_exists="append", index=False)
    _ok(f"  products: {len(products_df):,} rows")

    # ── invoices ──────────────────────────────────────────────────────────────
    _info("Building invoices table …")
    invoices_df = (
        df.groupby("InvoiceNo")
        .agg(
            customer_id=("CustomerID", "first"),
            invoice_date=("InvoiceDate", "first"),
        )
        .reset_index()
    )
    invoices_df.rename(columns={"InvoiceNo": "invoice_no"}, inplace=True)
    invoices_df["is_cancelled"] = invoices_df["invoice_no"].str.startswith("C").astype(int)
    invoices_df["invoice_date"] = invoices_df["invoice_date"].astype(str)

    cur.execute("DROP TABLE IF EXISTS invoices")
    cur.execute("""
        CREATE TABLE invoices (
            invoice_no   TEXT PRIMARY KEY,
            customer_id  TEXT REFERENCES customers(customer_id),
            invoice_date TEXT NOT NULL,
            is_cancelled INTEGER NOT NULL DEFAULT 0
        )
    """)
    invoices_df.to_sql("invoices", conn, if_exists="append", index=False)
    _ok(f"  invoices: {len(invoices_df):,} rows  "
        f"({invoices_df['is_cancelled'].sum():,} cancellations)")

    # ── invoice_items ─────────────────────────────────────────────────────────
    _info("Building invoice_items table …")
    items_df = df[["InvoiceNo", "StockCode", "Quantity", "UnitPrice"]].copy()
    items_df.rename(
        columns={
            "InvoiceNo": "invoice_no",
            "StockCode": "stock_code",
            "Quantity": "quantity",
            "UnitPrice": "unit_price",
        },
        inplace=True,
    )
    # Drop rows with null stock_code or invoice_no (data quality — some rows lack these)
    before = len(items_df)
    items_df = items_df.dropna(subset=["stock_code", "invoice_no"])
    items_df = items_df[items_df["stock_code"].str.strip() != ""]
    dropped = before - len(items_df)
    if dropped:
        _info(f"  Dropped {dropped:,} invoice_item rows with null/empty stock_code")

    items_df["line_total"] = (items_df["quantity"] * items_df["unit_price"]).round(2)
    items_df = items_df.reset_index(drop=True)
    items_df.index.name = "item_id"

    cur.execute("DROP TABLE IF EXISTS invoice_items")
    cur.execute("""
        CREATE TABLE invoice_items (
            item_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_no  TEXT    NOT NULL REFERENCES invoices(invoice_no),
            stock_code  TEXT    NOT NULL REFERENCES products(stock_code),
            quantity    INTEGER NOT NULL,
            unit_price  REAL    NOT NULL,
            line_total  REAL    NOT NULL
        )
    """)
    items_df.to_sql("invoice_items", conn, if_exists="append", index=False)
    _ok(f"  invoice_items: {len(items_df):,} rows")

    # ── Indexes for join performance ──────────────────────────────────────────
    cur.execute("CREATE INDEX IF NOT EXISTS idx_inv_customer ON invoices(customer_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_items_invoice ON invoice_items(invoice_no)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_items_stock ON invoice_items(stock_code)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_items_total ON invoice_items(line_total)")
    conn.commit()


def setup_retail_db(force: bool = False) -> None:
    """Entry point: create retail.db from Online Retail.xlsx."""
    if RETAIL_DB.exists() and not force:
        _ok(f"retail.db already exists — skipping (use --force to rebuild)")
        return

    if not RETAIL_XLSX.exists():
        _err(f"Online Retail.xlsx not found at:\n  {RETAIL_XLSX}")
        sys.exit(1)

    _info(f"{'Rebuilding' if force else 'Creating'} retail.db …")
    t0 = time.time()
    df = _load_retail_raw()
    with sqlite3.connect(RETAIL_DB) as conn:
        _build_retail_db(df, conn)
    elapsed = time.time() - t0
    _ok(f"retail.db ready  ({elapsed:.1f}s)  →  {RETAIL_DB}")


# ══════════════════════════════════════════════════════════════════════════════
#  MIMIC-IV DATABASE
# ══════════════════════════════════════════════════════════════════════════════

# Tables to load from hosp/ and icu/ — (filename_stem, is_large)
HOSP_TABLES: list[tuple[str, bool]] = [
    ("admissions", False),
    ("patients", False),
    ("diagnoses_icd", False),
    ("labevents", True),       # 107 k rows
    ("prescriptions", True),   # 18 k rows
    ("procedures_icd", False),
    ("drgcodes", False),
    ("services", False),
    ("transfers", False),
    ("omr", False),
    ("d_labitems", False),
    ("d_icd_diagnoses", False),
    ("d_icd_procedures", False),
    ("emar", True),
    ("pharmacy", True),
    ("poe", True),
    ("microbiologyevents", False),
    ("hcpcsevents", False),
]

ICU_TABLES: list[tuple[str, bool]] = [
    ("icustays", False),
    ("chartevents", True),     # 668 k rows — largest table
    ("datetimeevents", False),
    ("inputevents", True),
    ("outputevents", False),
    ("procedureevents", False),
    ("ingredientevents", True),
    ("d_items", False),
    ("caregiver", False),
]


def _load_csv(path: Path, chunksize: Optional[int] = None) -> pd.DataFrame:
    """Load a plain CSV, handling .csv extension."""
    if not path.exists():
        return pd.DataFrame()  # return empty if missing
    return pd.read_csv(path, low_memory=False)


def _ingest_table(
    conn: sqlite3.Connection,
    table_name: str,
    df: pd.DataFrame,
) -> int:
    """Write a DataFrame to SQLite, replacing if exists. Returns row count."""
    if df.empty:
        return 0
    df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=10_000)
    return len(df)


def setup_mimic_db(force: bool = False) -> None:
    """Entry point: load all MIMIC-IV CSV tables into mimic.db."""
    if MIMIC_DB.exists() and not force:
        _ok(f"mimic.db already exists — skipping (use --force to rebuild)")
        return

    if not MIMIC_DIR.exists():
        _err(f"MIMIC-IV directory not found at:\n  {MIMIC_DIR}")
        sys.exit(1)

    _info(f"{'Rebuilding' if force else 'Creating'} mimic.db …")
    t0 = time.time()

    with sqlite3.connect(MIMIC_DB) as conn:
        total_rows = 0

        # ── hosp tables ───────────────────────────────────────────────────────
        print(f"\n{_BOLD}  Loading hosp/ tables:{_RESET}")
        for stem, is_large in tqdm(HOSP_TABLES, desc="  hosp", unit="table"):
            path = HOSP_DIR / f"{stem}.csv"
            df = _load_csv(path)
            n = _ingest_table(conn, stem, df)
            total_rows += n
            _ok(f"    hosp/{stem}: {n:,} rows")

        # ── icu tables ────────────────────────────────────────────────────────
        print(f"\n{_BOLD}  Loading icu/ tables:{_RESET}")
        for stem, is_large in tqdm(ICU_TABLES, desc="  icu ", unit="table"):
            path = ICU_DIR / f"{stem}.csv"
            df = _load_csv(path)
            n = _ingest_table(conn, stem, df)
            total_rows += n
            _ok(f"    icu/{stem}: {n:,} rows")

        # ── Indexes on key join columns ───────────────────────────────────────
        _info("Creating indexes …")
        index_sql = [
            "CREATE INDEX IF NOT EXISTS idx_adm_subject ON admissions(subject_id)",
            "CREATE INDEX IF NOT EXISTS idx_adm_hadm    ON admissions(hadm_id)",
            "CREATE INDEX IF NOT EXISTS idx_dx_hadm     ON diagnoses_icd(hadm_id)",
            "CREATE INDEX IF NOT EXISTS idx_lab_hadm    ON labevents(hadm_id)",
            "CREATE INDEX IF NOT EXISTS idx_lab_subject ON labevents(subject_id)",
            "CREATE INDEX IF NOT EXISTS idx_rx_hadm     ON prescriptions(hadm_id)",
            "CREATE INDEX IF NOT EXISTS idx_icu_subject ON icustays(subject_id)",
            "CREATE INDEX IF NOT EXISTS idx_icu_hadm    ON icustays(hadm_id)",
            "CREATE INDEX IF NOT EXISTS idx_chart_stay  ON chartevents(stay_id)",
        ]
        for sql in index_sql:
            conn.execute(sql)
        conn.commit()

    elapsed = time.time() - t0
    _ok(f"mimic.db ready  ({elapsed:.1f}s, {total_rows:,} total rows)  →  {MIMIC_DB}")


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-DATABASE BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

def setup_bridge_table(force: bool = False) -> None:
    """
    Create a patient_customer_bridge table in BOTH databases.

    Maps retail CustomerID (TEXT) ←→ MIMIC subject_id (INTEGER).
    Since the two datasets are from different domains, the bridge is
    constructed deterministically: we use a reproducible random mapping
    so that the system can demonstrate cross-DB joins without fabricating
    clinical meaning.

    In a real deployment this table would come from an identity resolution
    system or a consent-based data linkage.
    """
    import hashlib, random

    _info("Building cross-DB bridge table …")

    # Read customer IDs from retail DB
    with sqlite3.connect(RETAIL_DB) as rconn:
        cids = pd.read_sql("SELECT DISTINCT customer_id FROM customers", rconn)

    # Read subject IDs from mimic DB
    with sqlite3.connect(MIMIC_DB) as mconn:
        sids = pd.read_sql("SELECT DISTINCT subject_id FROM patients", mconn)

    customer_ids: list[str] = cids["customer_id"].dropna().tolist()
    subject_ids: list[int] = sids["subject_id"].dropna().astype(int).tolist()

    # Deterministic pseudo-random assignment (seed = project hash)
    rng = random.Random(42)
    # Assign each subject_id to a customer_id  (n_patients ≤ n_customers)
    sampled_customers = rng.sample(customer_ids, min(len(subject_ids), len(customer_ids)))

    bridge_df = pd.DataFrame(
        {
            "customer_id": sampled_customers[: len(subject_ids)],
            "subject_id": subject_ids[: len(sampled_customers)],
            "link_confidence": 1.0,  # deterministic mapping — confidence = 100%
            "link_method": "demo_assignment",
        }
    )

    for db_path in (RETAIL_DB, MIMIC_DB):
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS patient_customer_bridge")
            conn.execute("""
                CREATE TABLE patient_customer_bridge (
                    customer_id      TEXT    NOT NULL,
                    subject_id       INTEGER NOT NULL,
                    link_confidence  REAL    DEFAULT 1.0,
                    link_method      TEXT    DEFAULT 'demo_assignment',
                    PRIMARY KEY (customer_id, subject_id)
                )
            """)
            bridge_df.to_sql(
                "patient_customer_bridge", conn, if_exists="append", index=False
            )
            conn.commit()

    _ok(f"Bridge table: {len(bridge_df):,} customer ↔ patient links (in both DBs)")


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_databases() -> None:
    """Print row counts for every table in both databases."""
    print(f"\n{_BOLD}{'═'*60}{_RESET}")
    print(f"{_BOLD}  DATABASE VERIFICATION{_RESET}")
    print(f"{_BOLD}{'═'*60}{_RESET}")

    for label, db_path in [("retail.db", RETAIL_DB), ("mimic.db", MIMIC_DB)]:
        print(f"\n{_BOLD}  {label}{_RESET}")
        with sqlite3.connect(db_path) as conn:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
            )
            for tbl in tables["name"]:
                cnt = conn.execute(f"SELECT COUNT(*) FROM [{tbl}]").fetchone()[0]
                bar = "█" * min(40, cnt // 2000 + 1)
                print(f"    {tbl:<35} {cnt:>10,}  {bar}")

    print(f"\n{_GREEN}{'═'*60}{_RESET}")
    print(f"{_GREEN}  All databases ready.{_RESET}")
    print(f"{_GREEN}{'═'*60}{_RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialise retail.db and mimic.db from raw source files."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate databases even if they already exist.",
    )
    parser.add_argument(
        "--retail-only", action="store_true", help="Only process Online Retail database."
    )
    parser.add_argument(
        "--mimic-only", action="store_true", help="Only process MIMIC-IV database."
    )
    args = parser.parse_args()

    print(f"\n{_BOLD}{'═'*60}{_RESET}")
    print(f"{_BOLD}  Multi-DB SLM — Database Setup{_RESET}")
    print(f"{_BOLD}{'═'*60}{_RESET}\n")

    if not args.mimic_only:
        setup_retail_db(force=args.force)

    if not args.retail_only:
        setup_mimic_db(force=args.force)

    if not args.retail_only and not args.mimic_only:
        setup_bridge_table(force=args.force)

    verify_databases()


if __name__ == "__main__":
    main()
