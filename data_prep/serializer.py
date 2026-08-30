"""
data_prep/serializer.py
────────────────────────
Converts both SQLite databases into natural language text
suitable for LLM fine-tuning.

Three conversion strategies:
  1. Schema narration   — describes each table's purpose and columns in prose
  2. Row serialization  — converts individual rows to natural language sentences
  3. Aggregate facts    — runs real SQL and converts results to factual statements

These text blocks become the raw material that qa_generator.py
turns into instruction-tuning (question, answer) training pairs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RETAIL_DB = ROOT / "data" / "retail.db"
MIMIC_DB  = ROOT / "data" / "mimic.db"


def _conn(db: str) -> sqlite3.Connection:
    path = RETAIL_DB if db == "retail" else MIMIC_DB
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _sql(db: str, query: str) -> pd.DataFrame:
    with _conn(db) as c:
        return pd.read_sql_query(query, c)


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 1 — Schema Narration
# ══════════════════════════════════════════════════════════════════════════════

RETAIL_SCHEMA_NARRATION = """
The Online Retail database (retail.db) contains one year of UK e-commerce
transactions from December 2010 to December 2011.

TABLE: customers
  Stores unique retail customers identified by customer_id (text).
  Each customer has a country of residence and a segment classification:
  Domestic_UK (United Kingdom), EU_Wholesale (European countries), or
  International (rest of world). The registration_date is the earliest
  known transaction date for that customer.

TABLE: products
  Product catalogue with stock_code as the primary key (a 5-character code).
  Each product has a description, a category (e.g. Gifts_Romantic,
  Seasonal_Christmas, Lighting, Kitchenware), and a unit_price in GBP sterling.

TABLE: invoices
  Transaction header for each order. invoice_no is the primary key.
  customer_id links to the customers table.
  is_cancelled = 1 means the order was a return or refund
  (these invoice numbers start with the letter C).

TABLE: invoice_items
  Line items for each invoice. Each row is one product on one invoice:
  quantity units at unit_price each, giving line_total = quantity × unit_price.
  A single invoice can contain multiple products (multiple rows).

TABLE: patient_customer_bridge
  Cross-database link table. Maps retail customer_id to MIMIC patient subject_id.
  Used to join the retail and clinical databases for cross-domain analysis.

RELATIONSHIPS:
  invoice_items.invoice_no  →  invoices.invoice_no
  invoice_items.stock_code  →  products.stock_code
  invoices.customer_id      →  customers.customer_id
  patient_customer_bridge.customer_id → customers.customer_id  (retail side)
  patient_customer_bridge.subject_id  → patients.subject_id    (MIMIC side)
""".strip()

MIMIC_SCHEMA_NARRATION = """
The MIMIC-IV Clinical Demo database (mimic.db) contains de-identified electronic
health records for 100 patients admitted to Beth Israel Deaconess Medical Center.
Data spans both hospital (hosp) and intensive care unit (icu) modules.

HOSPITAL MODULE (hosp):

TABLE: patients
  Demographics for 100 de-identified patients. subject_id is the unique patient key.
  gender is 'M' or 'F'. anchor_age is the patient's age at anchor_year.
  anchor_year_group places patients in a 3-year cohort (e.g. '2011 - 2013').
  dod is date of death (empty if still alive at data collection).

TABLE: admissions
  Hospital admission records. hadm_id is the unique admission identifier.
  subject_id links to the patients table. One patient can have multiple admissions.
  admittime and dischtime record arrival and departure timestamps.
  admission_type includes URGENT, ELECTIVE, EW EMER (emergency), DIRECT EMER, OBSERVATION ADMIT.
  admission_location and discharge_location describe patient origin and destination.
  insurance is the payer type (Medicare, Medicaid, Other).
  hospital_expire_flag = 1 means the patient died during this hospital admission.
  edregtime and edouttime record emergency department entry and exit times.

TABLE: diagnoses_icd
  ICD-9 or ICD-10 diagnosis codes assigned during each admission.
  seq_num = 1 is the primary (most important) diagnosis.
  icd_code is the diagnosis code; icd_version is 9 or 10.

TABLE: labevents
  Laboratory test results. Each row is one test result for one patient admission.
  itemid links to d_labitems for the test name.
  value and valuenum are the result (text and numeric forms).
  flag = 'abnormal' means the result is outside reference range.
  ref_range_lower and ref_range_upper define the normal range.

TABLE: prescriptions
  Medication orders. drug is the medication name.
  dose_val_rx and dose_unit_rx describe the prescribed dose.
  route is the administration route (PO = oral, IV = intravenous, etc.).
  starttime and stoptime define the prescription window.

TABLE: procedures_icd
  ICD-coded procedures performed during each admission.

TABLE: d_labitems
  Reference dictionary: maps itemid to lab test label and fluid source.

TABLE: d_icd_diagnoses
  Reference dictionary: maps ICD code + version to long diagnosis description.

TABLE: d_icd_procedures
  Reference dictionary: maps ICD procedure code + version to description.

ICU MODULE (icu):

TABLE: icustays
  ICU admission records. stay_id is unique per ICU stay.
  first_careunit is the initial ICU care unit (e.g. 'Medical Intensive Care Unit (MICU)',
  'Neuro Stepdown', 'Cardiac Vascular Intensive Care Unit (CVICU)').
  los is length of stay in days (decimal).

TABLE: chartevents
  High-frequency physiological measurements from ICU monitors.
  Each row is one measurement at one charttime for one patient.
  itemid links to d_items for the measurement name (e.g. Heart Rate, Blood Pressure).
  value and valuenum store the recorded measurement.
  warning = 1 means the value triggered a clinical alert.

TABLE: inputevents
  Fluids and medications administered via IV in the ICU.

TABLE: outputevents
  Urine output and drain measurements from ICU patients.

TABLE: d_items
  Reference dictionary: maps ICU itemid to measurement name, unit, and category.

RELATIONSHIPS:
  admissions.subject_id   →  patients.subject_id
  diagnoses_icd.hadm_id   →  admissions.hadm_id
  labevents.hadm_id       →  admissions.hadm_id
  prescriptions.hadm_id   →  admissions.hadm_id
  icustays.hadm_id        →  admissions.hadm_id
  chartevents.stay_id     →  icustays.stay_id
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 2 — Row Serialization
# ══════════════════════════════════════════════════════════════════════════════

def serialize_retail_rows(n_customers: int = 20, n_products: int = 30,
                           n_invoices: int = 20) -> list[str]:
    """Convert sample rows from retail tables into natural language sentences."""
    texts: list[str] = []

    # Customers
    df = _sql("retail", f"SELECT * FROM customers LIMIT {n_customers}")
    for _, row in df.iterrows():
        texts.append(
            f"Customer {row['customer_id']} is located in {row['country']} "
            f"and classified as a {row['segment'].replace('_', ' ')} customer."
        )

    # Products
    df = _sql("retail", f"SELECT * FROM products ORDER BY unit_price DESC LIMIT {n_products}")
    for _, row in df.iterrows():
        texts.append(
            f"Product {row['stock_code']} is '{row['description']}' in the "
            f"{row['category'].replace('_', ' ')} category, priced at "
            f"£{row['unit_price']:.2f} per unit."
        )

    # Invoices with revenue
    df = _sql("retail", f"""
        SELECT i.invoice_no, i.customer_id, i.invoice_date,
               i.is_cancelled, ROUND(SUM(ii.line_total),2) AS total_value
        FROM invoices i
        JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
        GROUP BY i.invoice_no
        ORDER BY total_value DESC
        LIMIT {n_invoices}
    """)
    for _, row in df.iterrows():
        status = "was cancelled (return/refund)" if row["is_cancelled"] else "was a confirmed purchase"
        texts.append(
            f"Invoice {row['invoice_no']} placed on {str(row['invoice_date'])[:10]} "
            f"by customer {row['customer_id'] or 'guest'} {status} "
            f"with a total value of £{row['total_value']:.2f}."
        )

    return texts


def serialize_mimic_rows(n_patients: int = 30, n_admissions: int = 30,
                          n_diagnoses: int = 30) -> list[str]:
    """Convert sample rows from MIMIC tables into natural language sentences."""
    texts: list[str] = []

    # Patients
    df = _sql("mimic", f"SELECT * FROM patients LIMIT {n_patients}")
    for _, row in df.iterrows():
        dod = f", died on {row['dod']}" if row["dod"] else ""
        texts.append(
            f"Patient {row['subject_id']} is {row['gender']} (gender code), "
            f"aged {row['anchor_age']} at their reference year, "
            f"in the {row['anchor_year_group']} cohort{dod}."
        )

    # Admissions
    df = _sql("mimic", f"""
        SELECT a.*, p.gender, p.anchor_age
        FROM admissions a JOIN patients p ON a.subject_id = p.subject_id
        LIMIT {n_admissions}
    """)
    for _, row in df.iterrows():
        outcome = "died during this admission" if row["hospital_expire_flag"] else "survived and was discharged"
        texts.append(
            f"Admission {row['hadm_id']}: patient {row['subject_id']} "
            f"({row['gender']}, age ~{row['anchor_age']}) was admitted as "
            f"{row['admission_type']} from {row['admission_location']} and {outcome} "
            f"to {row['discharge_location']}. Insurance: {row['insurance']}."
        )

    # Diagnoses with descriptions
    df = _sql("mimic", f"""
        SELECT d.subject_id, d.hadm_id, d.icd_code, d.icd_version,
               d.seq_num, dd.long_title
        FROM diagnoses_icd d
        LEFT JOIN d_icd_diagnoses dd
          ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
        WHERE d.seq_num <= 2
        LIMIT {n_diagnoses}
    """)
    for _, row in df.iterrows():
        rank = "primary" if row["seq_num"] == 1 else "secondary"
        title = row["long_title"] or "unknown diagnosis"
        texts.append(
            f"Patient {row['subject_id']}, admission {row['hadm_id']}: "
            f"{rank} ICD-{row['icd_version']} diagnosis is "
            f"'{title}' (code {row['icd_code']})."
        )

    return texts


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 3 — Aggregate Facts (from real SQL execution)
# ══════════════════════════════════════════════════════════════════════════════

def generate_retail_facts() -> list[str]:
    """Generate factual statements about the retail DB from actual SQL results."""
    facts: list[str] = []

    # Total stats
    r = _sql("retail", """
        SELECT COUNT(DISTINCT c.customer_id) AS customers,
               COUNT(DISTINCT i.invoice_no) AS invoices,
               COUNT(DISTINCT ii.stock_code) AS products,
               COUNT(DISTINCT c.country) AS countries
        FROM invoices i
        JOIN customers c ON i.customer_id = c.customer_id
        JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
    """).iloc[0]
    facts.append(
        f"The Online Retail database contains {r['customers']:,} registered customers, "
        f"{r['invoices']:,} invoices, {r['products']:,} distinct products, "
        f"and spans {r['countries']} countries."
    )

    # Revenue
    r = _sql("retail", """
        SELECT ROUND(SUM(ii.line_total),2) AS revenue,
               SUM(i.is_cancelled) AS cancellations,
               COUNT(*) AS total
        FROM invoices i
        JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
    """).iloc[0]
    facts.append(
        f"Total gross line value across all transactions is £{r['revenue']:,.2f}. "
        f"There are {r['cancellations']:,} cancelled invoices out of {r['total']:,} total "
        f"({100*r['cancellations']/r['total']:.1f}% cancellation rate)."
    )

    # Confirmed revenue only
    r = _sql("retail", """
        SELECT ROUND(SUM(ii.line_total),2) AS confirmed_revenue
        FROM invoices i
        JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
        WHERE i.is_cancelled = 0
    """).iloc[0]
    facts.append(
        f"Confirmed (non-cancelled) order revenue totals £{r['confirmed_revenue']:,.2f}."
    )

    # Top 5 countries
    df = _sql("retail", """
        SELECT c.country,
               COUNT(DISTINCT c.customer_id) AS num_customers,
               ROUND(SUM(ii.line_total),2) AS revenue
        FROM customers c
        JOIN invoices i ON c.customer_id = i.customer_id
        JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
        WHERE i.is_cancelled = 0
        GROUP BY c.country
        ORDER BY revenue DESC LIMIT 5
    """)
    country_str = "; ".join(
        f"{row['country']} (£{row['revenue']:,.0f}, {row['num_customers']} customers)"
        for _, row in df.iterrows()
    )
    facts.append(f"Top 5 countries by confirmed revenue: {country_str}.")

    # Top product categories
    df = _sql("retail", """
        SELECT p.category, ROUND(SUM(ii.line_total),2) AS revenue
        FROM invoice_items ii
        JOIN products p ON ii.stock_code = p.stock_code
        JOIN invoices i ON ii.invoice_no = i.invoice_no
        WHERE i.is_cancelled = 0
        GROUP BY p.category ORDER BY revenue DESC LIMIT 5
    """)
    cat_str = "; ".join(f"{r['category'].replace('_',' ')} £{r['revenue']:,.0f}" for _, r in df.iterrows())
    facts.append(f"Top product categories by revenue: {cat_str}.")

    # Monthly trend
    df = _sql("retail", """
        SELECT SUBSTR(i.invoice_date,1,7) AS month,
               ROUND(SUM(ii.line_total),2) AS revenue
        FROM invoices i JOIN invoice_items ii ON i.invoice_no = ii.invoice_no
        WHERE i.is_cancelled = 0
        GROUP BY month ORDER BY revenue DESC LIMIT 3
    """)
    month_str = "; ".join(f"{r['month']} £{r['revenue']:,.0f}" for _, r in df.iterrows())
    facts.append(f"Highest revenue months: {month_str}.")

    # Customer segments
    df = _sql("retail", """
        SELECT segment, COUNT(*) AS cnt FROM customers GROUP BY segment ORDER BY cnt DESC
    """)
    seg_str = "; ".join(f"{r['segment'].replace('_',' ')}: {r['cnt']:,}" for _, r in df.iterrows())
    facts.append(f"Customer segment breakdown: {seg_str}.")

    return facts


def generate_mimic_facts() -> list[str]:
    """Generate factual statements about the MIMIC DB from actual SQL results."""
    facts: list[str] = []

    # Patient overview
    r = _sql("mimic", """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN gender='F' THEN 1 ELSE 0 END) AS female,
               SUM(CASE WHEN gender='M' THEN 1 ELSE 0 END) AS male,
               ROUND(AVG(anchor_age),1) AS avg_age,
               SUM(CASE WHEN dod IS NOT NULL AND dod != '' THEN 1 ELSE 0 END) AS deceased
        FROM patients
    """).iloc[0]
    facts.append(
        f"The MIMIC-IV demo contains {r['total']} patients: "
        f"{r['female']} female and {r['male']} male, "
        f"with a mean age of {r['avg_age']} years at their reference year. "
        f"{r['deceased']} patients have a recorded date of death."
    )

    # Admissions
    r = _sql("mimic", """
        SELECT COUNT(*) AS total,
               SUM(hospital_expire_flag) AS deaths,
               ROUND(100.0*SUM(hospital_expire_flag)/COUNT(*),1) AS mortality_pct
        FROM admissions
    """).iloc[0]
    facts.append(
        f"There are {r['total']} hospital admissions. "
        f"{r['deaths']} resulted in in-hospital death "
        f"({r['mortality_pct']}% in-hospital mortality rate)."
    )

    # Admission types
    df = _sql("mimic", """
        SELECT admission_type, COUNT(*) AS cnt FROM admissions
        GROUP BY admission_type ORDER BY cnt DESC
    """)
    adm_str = "; ".join(f"{r['admission_type']}: {r['cnt']}" for _, r in df.iterrows())
    facts.append(f"Admission types breakdown: {adm_str}.")

    # Insurance
    df = _sql("mimic", """
        SELECT insurance, COUNT(*) AS cnt FROM admissions
        GROUP BY insurance ORDER BY cnt DESC
    """)
    ins_str = "; ".join(f"{r['insurance']}: {r['cnt']}" for _, r in df.iterrows())
    facts.append(f"Insurance coverage breakdown: {ins_str}.")

    # ICU
    r = _sql("mimic", """
        SELECT COUNT(*) AS stays, COUNT(DISTINCT subject_id) AS patients,
               ROUND(AVG(los),2) AS avg_los, ROUND(MAX(los),1) AS max_los,
               ROUND(MIN(los),2) AS min_los
        FROM icustays
    """).iloc[0]
    facts.append(
        f"There are {r['stays']} ICU stays across {r['patients']} patients. "
        f"Average ICU length of stay is {r['avg_los']} days "
        f"(range: {r['min_los']} to {r['max_los']} days)."
    )

    # ICU units
    df = _sql("mimic", """
        SELECT first_careunit, COUNT(*) AS cnt, ROUND(AVG(los),2) AS avg_los
        FROM icustays GROUP BY first_careunit ORDER BY cnt DESC
    """)
    for _, row in df.iterrows():
        facts.append(
            f"ICU care unit '{row['first_careunit']}' had {row['cnt']} stay(s) "
            f"with average length of stay {row['avg_los']} days."
        )

    # Top diagnoses
    df = _sql("mimic", """
        SELECT d.icd_code, dd.long_title, COUNT(*) AS freq
        FROM diagnoses_icd d
        LEFT JOIN d_icd_diagnoses dd ON d.icd_code=dd.icd_code AND d.icd_version=dd.icd_version
        WHERE d.seq_num=1
        GROUP BY d.icd_code ORDER BY freq DESC LIMIT 5
    """)
    dx_str = "; ".join(
        f"'{r['long_title'] or r['icd_code']}' ({r['freq']} cases)"
        for _, r in df.iterrows()
    )
    facts.append(f"Top 5 primary diagnoses: {dx_str}.")

    # Lab events
    r = _sql("mimic", """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN flag='abnormal' THEN 1 ELSE 0 END) AS abnormal
        FROM labevents
    """).iloc[0]
    pct = 100 * r['abnormal'] / max(r['total'], 1)
    facts.append(
        f"There are {r['total']:,} laboratory results. "
        f"{r['abnormal']:,} ({pct:.1f}%) are flagged as abnormal."
    )

    # Top labs
    df = _sql("mimic", """
        SELECT d.label, COUNT(*) AS cnt
        FROM labevents l JOIN d_labitems d ON l.itemid=d.itemid
        GROUP BY l.itemid ORDER BY cnt DESC LIMIT 5
    """)
    lab_str = "; ".join(f"{r['label']} ({r['cnt']:,})" for _, r in df.iterrows())
    facts.append(f"Most frequently ordered lab tests: {lab_str}.")

    # Chartevents
    r = _sql("mimic", "SELECT COUNT(*) AS total FROM chartevents").iloc[0]
    facts.append(
        f"The ICU chart contains {r['total']:,} clinical measurements "
        f"(vital signs, physiological parameters) across all ICU stays."
    )

    # Prescriptions
    r = _sql("mimic", "SELECT COUNT(*) AS total, COUNT(DISTINCT drug) AS drugs FROM prescriptions").iloc[0]
    facts.append(
        f"There are {r['total']:,} prescription orders covering {r['drugs']:,} distinct medications."
    )

    # Cross-DB bridge
    r = _sql("retail", "SELECT COUNT(*) AS links FROM patient_customer_bridge").iloc[0]
    facts.append(
        f"The cross-database bridge table links {r['links']} retail customers "
        f"to MIMIC-IV patient records, enabling combined retail-clinical analysis."
    )

    return facts


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== RETAIL SCHEMA ===")
    print(RETAIL_SCHEMA_NARRATION[:500])
    print("\n=== RETAIL FACTS ===")
    for f in generate_retail_facts():
        print(" •", f)
    print("\n=== MIMIC FACTS ===")
    for f in generate_mimic_facts():
        print(" •", f)
    print("\n=== SAMPLE RETAIL ROWS ===")
    for t in serialize_retail_rows(n_customers=3, n_products=3, n_invoices=3):
        print(" •", t)
    print("\n=== SAMPLE MIMIC ROWS ===")
    for t in serialize_mimic_rows(n_patients=3, n_admissions=3, n_diagnoses=3):
        print(" •", t)
