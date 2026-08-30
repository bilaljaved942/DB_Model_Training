"""
data_prep/qa_generator.py
──────────────────────────
Generates instruction-tuning (question, answer) pairs by:
  1. Running real SQL queries against both databases
  2. Formatting results as natural language answers
  3. Pairing each with a natural language question

The output is used as training data for supervised fine-tuning (SFT).

Format of each example:
  {
    "instruction": "How many patients died in hospital?",
    "input": "",
    "output": "Based on the MIMIC-IV clinical demo database, 13 patients died
               during their hospital admission out of 275 total admissions,
               giving an in-hospital mortality rate of 4.7%.",
    "source": "mimic",
    "category": "aggregate"
  }

Categories:
  - schema      : questions about table/column structure
  - aggregate   : COUNT, AVG, SUM type questions
  - top_n       : ranking questions (top 5, highest, etc.)
  - filter      : questions about specific subsets
  - comparison  : comparing two groups or time periods
  - cross_db    : questions spanning both databases
  - business    : business logic / derived insight questions
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RETAIL_DB = ROOT / "data" / "retail.db"
MIMIC_DB = ROOT / "data" / "mimic.db"


def _sql(db: str, query: str) -> pd.DataFrame:
    path = RETAIL_DB if db == "retail" else MIMIC_DB
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(query, conn)


def _v(df: pd.DataFrame, row: int = 0, col: int = 0):
    """Get a single value from a query result safely."""
    try:
        return df.iloc[row, col]
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Retail QA generators
# ══════════════════════════════════════════════════════════════════════════════

def _retail_schema_qa() -> list[dict]:
    return [
        {
            "instruction": "What tables are in the Online Retail database?",
            "input": "",
            "output": (
                "The Online Retail database contains five tables: "
                "customers (customer profiles with country and segment), "
                "products (product catalogue with stock codes, descriptions, categories, and prices), "
                "invoices (transaction headers with date, customer link, and cancellation flag), "
                "invoice_items (line items linking invoices to products with quantity and price), "
                "and patient_customer_bridge (cross-database link to MIMIC patient records)."
            ),
            "source": "retail",
            "category": "schema",
        },
        {
            "instruction": "What does the is_cancelled field in the invoices table mean?",
            "input": "",
            "output": (
                "The is_cancelled field in the invoices table is a binary flag where 1 means the "
                "invoice is a cancellation (return or refund) and 0 means it is a confirmed purchase. "
                "Cancelled invoices have invoice numbers that begin with the letter C."
            ),
            "source": "retail",
            "category": "schema",
        },
        {
            "instruction": "How are invoices and products linked in the retail database?",
            "input": "",
            "output": (
                "Invoices and products are linked through the invoice_items table. "
                "Each row in invoice_items contains an invoice_no (linking to the invoices table) "
                "and a stock_code (linking to the products table), along with the quantity "
                "and unit_price for that specific product on that specific order. "
                "This many-to-many relationship means one invoice can contain multiple products "
                "and one product can appear on many invoices."
            ),
            "source": "retail",
            "category": "schema",
        },
        {
            "instruction": "What is the customer segment classification in the retail database?",
            "input": "",
            "output": (
                "Customers in the Online Retail database are classified into three segments: "
                "Domestic_UK for customers based in the United Kingdom, "
                "EU_Wholesale for European customers outside the UK, "
                "and International for customers from the rest of the world. "
                "The majority of revenue comes from the United Kingdom (Domestic_UK segment)."
            ),
            "source": "retail",
            "category": "schema",
        },
        {
            "instruction": "What does line_total represent in the invoice_items table?",
            "input": "",
            "output": (
                "The line_total column in invoice_items represents the total monetary value "
                "of that specific line item, calculated as quantity multiplied by unit_price. "
                "It is stored in British pounds sterling (GBP). "
                "Summing all line_total values across confirmed (non-cancelled) invoices "
                "gives the total gross revenue."
            ),
            "source": "retail",
            "category": "schema",
        },
    ]


def _retail_aggregate_qa() -> list[dict]:
    qa: list[dict] = []

    # Customer count
    r = _sql("retail", "SELECT COUNT(DISTINCT customer_id) AS n FROM customers").iloc[0]
    qa.append({
        "instruction": "How many unique customers are in the retail database?",
        "input": "",
        "output": f"The Online Retail database contains {r['n']:,} unique registered customers.",
        "source": "retail", "category": "aggregate",
    })

    # Product count
    r = _sql("retail", "SELECT COUNT(*) AS n FROM products").iloc[0]
    qa.append({
        "instruction": "How many distinct products are in the retail catalogue?",
        "input": "",
        "output": f"The retail product catalogue contains {r['n']:,} distinct products identified by unique stock codes.",
        "source": "retail", "category": "aggregate",
    })

    # Invoice stats
    r = _sql("retail", """
        SELECT COUNT(*) AS total, SUM(is_cancelled) AS cancelled,
               COUNT(*)-SUM(is_cancelled) AS confirmed
        FROM invoices
    """).iloc[0]
    pct = 100 * r['cancelled'] / max(r['total'], 1)
    qa.append({
        "instruction": "How many invoices are there in total, and how many are cancellations?",
        "input": "",
        "output": (
            f"There are {r['total']:,} total invoices in the retail database. "
            f"{r['cancelled']:,} are cancellations ({pct:.1f}%), "
            f"and {r['confirmed']:,} are confirmed purchases."
        ),
        "source": "retail", "category": "aggregate",
    })

    # Total revenue
    r = _sql("retail", """
        SELECT ROUND(SUM(ii.line_total),2) AS total_rev,
               ROUND(SUM(CASE WHEN i.is_cancelled=0 THEN ii.line_total ELSE 0 END),2) AS confirmed_rev
        FROM invoices i JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
    """).iloc[0]
    qa.append({
        "instruction": "What is the total revenue from the Online Retail database?",
        "input": "",
        "output": (
            f"The gross total of all line items (including cancellations) is £{r['total_rev']:,.2f}. "
            f"Excluding cancellations, confirmed order revenue totals £{r['confirmed_rev']:,.2f}."
        ),
        "source": "retail", "category": "aggregate",
    })

    # Country count
    r = _sql("retail", "SELECT COUNT(DISTINCT country) AS n FROM customers").iloc[0]
    qa.append({
        "instruction": "How many countries are represented in the retail customer base?",
        "input": "",
        "output": f"Customers in the Online Retail database come from {r['n']} different countries.",
        "source": "retail", "category": "aggregate",
    })

    # Line items count
    r = _sql("retail", "SELECT COUNT(*) AS n FROM invoice_items").iloc[0]
    qa.append({
        "instruction": "How many individual line items (product-invoice combinations) are there?",
        "input": "",
        "output": f"There are {r['n']:,} individual line items across all invoices in the retail database.",
        "source": "retail", "category": "aggregate",
    })

    return qa


def _retail_topn_qa() -> list[dict]:
    qa: list[dict] = []

    # Top 5 countries by revenue
    df = _sql("retail", """
        SELECT c.country, ROUND(SUM(ii.line_total),2) AS rev
        FROM customers c JOIN invoices i ON c.customer_id=i.customer_id
        JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
        WHERE i.is_cancelled=0 GROUP BY c.country ORDER BY rev DESC LIMIT 5
    """)
    rows = [f"{i+1}. {r['country']} (£{r['rev']:,.0f})" for i, (_, r) in enumerate(df.iterrows())]
    qa.append({
        "instruction": "Which countries generate the most revenue in the retail database?",
        "input": "",
        "output": "Top 5 countries by confirmed order revenue: " + "; ".join(rows) + ".",
        "source": "retail", "category": "top_n",
    })

    # Top 5 products
    df = _sql("retail", """
        SELECT p.description, ROUND(SUM(ii.line_total),2) AS rev
        FROM invoice_items ii JOIN products p ON ii.stock_code=p.stock_code
        JOIN invoices i ON ii.invoice_no=i.invoice_no
        WHERE i.is_cancelled=0 GROUP BY ii.stock_code ORDER BY rev DESC LIMIT 5
    """)
    rows = [f"{i+1}. {r['description']} (£{r['rev']:,.0f})" for i, (_, r) in enumerate(df.iterrows())]
    qa.append({
        "instruction": "What are the top 5 best-selling products by revenue?",
        "input": "",
        "output": "Top 5 products by confirmed revenue: " + "; ".join(rows) + ".",
        "source": "retail", "category": "top_n",
    })

    # Top product categories
    df = _sql("retail", """
        SELECT p.category, ROUND(SUM(ii.line_total),2) AS rev
        FROM invoice_items ii JOIN products p ON ii.stock_code=p.stock_code
        JOIN invoices i ON ii.invoice_no=i.invoice_no
        WHERE i.is_cancelled=0 GROUP BY p.category ORDER BY rev DESC LIMIT 5
    """)
    rows = [f"{r['category'].replace('_',' ')} (£{r['rev']:,.0f})" for _, r in df.iterrows()]
    qa.append({
        "instruction": "Which product categories generate the most revenue?",
        "input": "",
        "output": "Top product categories by confirmed revenue: " + "; ".join(rows) + ".",
        "source": "retail", "category": "top_n",
    })

    # Best month
    df = _sql("retail", """
        SELECT SUBSTR(i.invoice_date,1,7) AS month, ROUND(SUM(ii.line_total),2) AS rev
        FROM invoices i JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
        WHERE i.is_cancelled=0 GROUP BY month ORDER BY rev DESC LIMIT 3
    """)
    best = df.iloc[0]
    qa.append({
        "instruction": "Which month had the highest sales in the retail database?",
        "input": "",
        "output": (
            f"The highest revenue month was {best['month']} with £{best['rev']:,.2f} in confirmed sales. "
            f"Top 3 months: " + "; ".join(f"{r['month']} £{r['rev']:,.0f}" for _, r in df.iterrows()) + "."
        ),
        "source": "retail", "category": "top_n",
    })

    # Customers with most orders
    df = _sql("retail", """
        SELECT customer_id, COUNT(DISTINCT invoice_no) AS orders
        FROM invoices WHERE is_cancelled=0
        GROUP BY customer_id ORDER BY orders DESC LIMIT 5
    """)
    rows = [f"Customer {r['customer_id']} ({r['orders']} orders)" for _, r in df.iterrows()]
    qa.append({
        "instruction": "Which customers placed the most orders?",
        "input": "",
        "output": "Top 5 most active customers by confirmed order count: " + "; ".join(rows) + ".",
        "source": "retail", "category": "top_n",
    })

    return qa


def _retail_business_qa() -> list[dict]:
    qa: list[dict] = []

    # Average order value
    r = _sql("retail", """
        SELECT ROUND(AVG(order_total),2) AS aov FROM (
            SELECT i.invoice_no, SUM(ii.line_total) AS order_total
            FROM invoices i JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
            WHERE i.is_cancelled=0 GROUP BY i.invoice_no
        )
    """).iloc[0]
    qa.append({
        "instruction": "What is the average order value (basket size) in the retail database?",
        "input": "",
        "output": f"The average confirmed order value is £{r['aov']:,.2f} per invoice.",
        "source": "retail", "category": "business",
    })

    # Segment revenue share
    df = _sql("retail", """
        SELECT c.segment, ROUND(SUM(ii.line_total),2) AS rev
        FROM customers c JOIN invoices i ON c.customer_id=i.customer_id
        JOIN invoice_items ii ON i.invoice_no=ii.invoice_no
        WHERE i.is_cancelled=0 GROUP BY c.segment ORDER BY rev DESC
    """)
    total = df["rev"].sum()
    rows = [f"{r['segment'].replace('_',' ')} £{r['rev']:,.0f} ({100*r['rev']/total:.1f}%)"
            for _, r in df.iterrows()]
    qa.append({
        "instruction": "How does revenue break down by customer segment in the retail database?",
        "input": "",
        "output": "Confirmed revenue by segment: " + "; ".join(rows) + ".",
        "source": "retail", "category": "business",
    })

    # Date range
    r = _sql("retail", "SELECT MIN(invoice_date) AS mn, MAX(invoice_date) AS mx FROM invoices").iloc[0]
    qa.append({
        "instruction": "What date range does the retail transaction data cover?",
        "input": "",
        "output": (
            f"The retail transaction data spans from {str(r['mn'])[:10]} to {str(r['mx'])[:10]}, "
            f"covering approximately one full year from late 2010 to late 2011."
        ),
        "source": "retail", "category": "business",
    })

    return qa


# ══════════════════════════════════════════════════════════════════════════════
#  MIMIC QA generators
# ══════════════════════════════════════════════════════════════════════════════

def _mimic_schema_qa() -> list[dict]:
    return [
        {
            "instruction": "What is the MIMIC-IV clinical database?",
            "input": "",
            "output": (
                "MIMIC-IV (Medical Information Mart for Intensive Care) is a de-identified "
                "electronic health records database from Beth Israel Deaconess Medical Center. "
                "The demo version used here contains records for 100 patients with ICU admissions. "
                "It covers two modules: hosp (hospital-wide data including admissions, diagnoses, "
                "lab results, prescriptions, and procedures) and icu (intensive care data including "
                "ICU stays, vital sign measurements, fluid inputs, and outputs)."
            ),
            "source": "mimic", "category": "schema",
        },
        {
            "instruction": "What does hospital_expire_flag mean in the MIMIC admissions table?",
            "input": "",
            "output": (
                "The hospital_expire_flag is a binary indicator in the admissions table. "
                "A value of 1 means the patient died during that hospital admission "
                "(in-hospital mortality). A value of 0 means the patient survived and "
                "was discharged alive. This flag is commonly used to calculate in-hospital "
                "mortality rates for clinical research."
            ),
            "source": "mimic", "category": "schema",
        },
        {
            "instruction": "How are diagnoses coded in the MIMIC database?",
            "input": "",
            "output": (
                "Diagnoses in MIMIC are coded using the International Classification of Diseases (ICD). "
                "The diagnoses_icd table stores both ICD-9 and ICD-10 codes (indicated by the "
                "icd_version column, which is either 9 or 10). The seq_num column indicates "
                "the priority: seq_num = 1 is the primary (principal) diagnosis, "
                "while higher numbers are secondary diagnoses. "
                "The d_icd_diagnoses reference table maps codes to their full descriptions."
            ),
            "source": "mimic", "category": "schema",
        },
        {
            "instruction": "What is the relationship between admissions, icustays, and chartevents?",
            "input": "",
            "output": (
                "These three tables form a hierarchy in MIMIC: "
                "A patient (subject_id) can have multiple hospital admissions (hadm_id in the "
                "admissions table). Each admission can include one or more ICU stays (stay_id in "
                "the icustays table, linked via hadm_id). Each ICU stay generates thousands of "
                "clinical measurements stored in chartevents (linked via stay_id). "
                "So the join chain is: patients → admissions → icustays → chartevents."
            ),
            "source": "mimic", "category": "schema",
        },
        {
            "instruction": "What does the anchor_age column represent in the patients table?",
            "input": "",
            "output": (
                "The anchor_age column in the patients table represents the patient's age in years "
                "at the time of their anchor_year. Because MIMIC dates are shifted to remove "
                "identifying temporal information, anchor_year is a reference year (not the actual "
                "calendar year). The anchor_year_group provides a 3-year cohort window "
                "(e.g., '2011 - 2013') indicating approximately when the patient was treated."
            ),
            "source": "mimic", "category": "schema",
        },
    ]


def _mimic_aggregate_qa() -> list[dict]:
    qa: list[dict] = []

    # Patients
    r = _sql("mimic", """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN gender='F' THEN 1 ELSE 0 END) AS female,
               SUM(CASE WHEN gender='M' THEN 1 ELSE 0 END) AS male,
               ROUND(AVG(anchor_age),1) AS avg_age
        FROM patients
    """).iloc[0]
    qa.append({
        "instruction": "How many patients are in the MIMIC-IV demo database, and what is their gender breakdown?",
        "input": "",
        "output": (
            f"The MIMIC-IV demo database contains {r['total']} patients: "
            f"{r['female']} female and {r['male']} male, "
            f"with a mean anchor age of {r['avg_age']} years."
        ),
        "source": "mimic", "category": "aggregate",
    })

    # Admissions mortality
    r = _sql("mimic", """
        SELECT COUNT(*) AS total, SUM(hospital_expire_flag) AS deaths,
               ROUND(100.0*SUM(hospital_expire_flag)/COUNT(*),1) AS pct
        FROM admissions
    """).iloc[0]
    qa.append({
        "instruction": "What is the in-hospital mortality rate in the MIMIC demo?",
        "input": "",
        "output": (
            f"Out of {r['total']} hospital admissions, {r['deaths']} patients died "
            f"in-hospital, giving an in-hospital mortality rate of {r['pct']}%."
        ),
        "source": "mimic", "category": "aggregate",
    })

    # ICU LOS
    r = _sql("mimic", """
        SELECT COUNT(*) AS stays, COUNT(DISTINCT subject_id) AS patients,
               ROUND(AVG(los),2) AS avg_los, ROUND(MEDIAN(los),2) AS median_los
        FROM icustays
    """).iloc[0]
    qa.append({
        "instruction": "What is the average ICU length of stay in MIMIC?",
        "input": "",
        "output": (
            f"There are {r['stays']} ICU stays across {r['patients']} patients. "
            f"The average ICU length of stay is {r['avg_los']} days, "
            f"with a median of {r['median_los']} days."
        ),
        "source": "mimic", "category": "aggregate",
    })

    # Lab events
    r = _sql("mimic", """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN flag='abnormal' THEN 1 ELSE 0 END) AS abnormal
        FROM labevents
    """).iloc[0]
    pct = 100 * r['abnormal'] / max(r['total'], 1)
    qa.append({
        "instruction": "How many lab results are in MIMIC and what proportion are abnormal?",
        "input": "",
        "output": (
            f"The MIMIC database contains {r['total']:,} laboratory results. "
            f"{r['abnormal']:,} ({pct:.1f}%) are flagged as abnormal, "
            f"indicating results outside the clinical reference range."
        ),
        "source": "mimic", "category": "aggregate",
    })

    # Prescriptions
    r = _sql("mimic", "SELECT COUNT(*) AS n, COUNT(DISTINCT drug) AS drugs FROM prescriptions").iloc[0]
    qa.append({
        "instruction": "How many prescriptions are recorded in MIMIC?",
        "input": "",
        "output": (
            f"There are {r['n']:,} prescription orders in MIMIC, "
            f"covering {r['drugs']:,} distinct medications."
        ),
        "source": "mimic", "category": "aggregate",
    })

    # Deceased
    r = _sql("mimic", """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN dod IS NOT NULL AND dod!='' THEN 1 ELSE 0 END) AS deceased
        FROM patients
    """).iloc[0]
    qa.append({
        "instruction": "How many patients in MIMIC have a recorded date of death?",
        "input": "",
        "output": (
            f"Out of {r['total']} patients in the MIMIC demo, "
            f"{r['deceased']} have a recorded date of death (dod field is not empty)."
        ),
        "source": "mimic", "category": "aggregate",
    })

    return qa


def _mimic_topn_qa() -> list[dict]:
    qa: list[dict] = []

    # Top diagnoses
    df = _sql("mimic", """
        SELECT d.icd_code, d.icd_version, dd.long_title, COUNT(*) AS freq
        FROM diagnoses_icd d
        LEFT JOIN d_icd_diagnoses dd ON d.icd_code=dd.icd_code AND d.icd_version=dd.icd_version
        WHERE d.seq_num=1
        GROUP BY d.icd_code ORDER BY freq DESC LIMIT 5
    """)
    rows = [f"{r['long_title'] or r['icd_code']} ({r['freq']} cases)" for _, r in df.iterrows()]
    qa.append({
        "instruction": "What are the most common primary diagnoses in the MIMIC demo?",
        "input": "",
        "output": "Top 5 primary diagnoses: " + "; ".join(rows) + ".",
        "source": "mimic", "category": "top_n",
    })

    # Top labs
    df = _sql("mimic", """
        SELECT d.label, COUNT(*) AS n FROM labevents l
        JOIN d_labitems d ON l.itemid=d.itemid
        GROUP BY l.itemid ORDER BY n DESC LIMIT 5
    """)
    rows = [f"{r['label']} ({r['n']:,})" for _, r in df.iterrows()]
    qa.append({
        "instruction": "What are the most frequently ordered lab tests in MIMIC?",
        "input": "",
        "output": "Top 5 most ordered laboratory tests: " + "; ".join(rows) + ".",
        "source": "mimic", "category": "top_n",
    })

    # ICU units by stay count
    df = _sql("mimic", """
        SELECT first_careunit, COUNT(*) AS n, ROUND(AVG(los),2) AS avg_los
        FROM icustays GROUP BY first_careunit ORDER BY n DESC
    """)
    rows = [f"{r['first_careunit']} ({r['n']} stays, avg {r['avg_los']} days)" for _, r in df.iterrows()]
    qa.append({
        "instruction": "Which ICU care units are used in MIMIC and how many stays does each have?",
        "input": "",
        "output": "ICU care units by stay count: " + "; ".join(rows) + ".",
        "source": "mimic", "category": "top_n",
    })

    # Admission types
    df = _sql("mimic", """
        SELECT admission_type, COUNT(*) AS n FROM admissions GROUP BY admission_type ORDER BY n DESC
    """)
    rows = [f"{r['admission_type']} ({r['n']})" for _, r in df.iterrows()]
    qa.append({
        "instruction": "What types of hospital admissions are in MIMIC and how frequent are they?",
        "input": "",
        "output": "Admission types: " + "; ".join(rows) + ".",
        "source": "mimic", "category": "top_n",
    })

    # Top medications
    df = _sql("mimic", "SELECT drug, COUNT(*) AS n FROM prescriptions GROUP BY drug ORDER BY n DESC LIMIT 5")
    rows = [f"{r['drug']} ({r['n']} orders)" for _, r in df.iterrows()]
    qa.append({
        "instruction": "What are the most commonly prescribed medications in MIMIC?",
        "input": "",
        "output": "Top 5 most prescribed medications: " + "; ".join(rows) + ".",
        "source": "mimic", "category": "top_n",
    })

    return qa


def _mimic_clinical_qa() -> list[dict]:
    qa: list[dict] = []

    # Insurance
    df = _sql("mimic", "SELECT insurance, COUNT(*) AS n FROM admissions GROUP BY insurance ORDER BY n DESC")
    rows = [f"{r['insurance']} ({r['n']})" for _, r in df.iterrows()]
    qa.append({
        "instruction": "What insurance types cover patients in the MIMIC database?",
        "input": "",
        "output": "Insurance breakdown across admissions: " + "; ".join(rows) + ".",
        "source": "mimic", "category": "filter",
    })

    # Discharge locations
    df = _sql("mimic", """
        SELECT discharge_location, COUNT(*) AS n FROM admissions
        GROUP BY discharge_location ORDER BY n DESC LIMIT 6
    """)
    rows = [f"{r['discharge_location']} ({r['n']})" for _, r in df.iterrows()]
    qa.append({
        "instruction": "Where are patients typically discharged to after a MIMIC hospital admission?",
        "input": "",
        "output": "Top discharge locations: " + "; ".join(rows) + ".",
        "source": "mimic", "category": "filter",
    })

    # Patients with multiple admissions
    r = _sql("mimic", """
        SELECT COUNT(*) AS n FROM (
            SELECT subject_id FROM admissions
            GROUP BY subject_id HAVING COUNT(*)>1
        )
    """).iloc[0]
    qa.append({
        "instruction": "How many MIMIC patients had more than one hospital admission?",
        "input": "",
        "output": (
            f"{r['n']} patients in the MIMIC demo had more than one hospital admission, "
            f"indicating readmissions or multiple separate hospital encounters."
        ),
        "source": "mimic", "category": "filter",
    })

    # Long ICU stays
    r = _sql("mimic", "SELECT COUNT(*) AS n FROM icustays WHERE los > 7").iloc[0]
    qa.append({
        "instruction": "How many ICU stays lasted more than 7 days in MIMIC?",
        "input": "",
        "output": (
            f"{r['n']} ICU stays in the MIMIC demo lasted more than 7 days, "
            f"which are typically considered prolonged ICU admissions."
        ),
        "source": "mimic", "category": "filter",
    })

    # Abnormal labs
    df = _sql("mimic", """
        SELECT d.label, COUNT(*) AS n FROM labevents l
        JOIN d_labitems d ON l.itemid=d.itemid
        WHERE l.flag='abnormal'
        GROUP BY l.itemid ORDER BY n DESC LIMIT 5
    """)
    rows = [f"{r['label']} ({r['n']} abnormal)" for _, r in df.iterrows()]
    qa.append({
        "instruction": "Which lab tests have the most abnormal results in MIMIC?",
        "input": "",
        "output": "Lab tests with highest count of abnormal results: " + "; ".join(rows) + ".",
        "source": "mimic", "category": "filter",
    })

    return qa


# ══════════════════════════════════════════════════════════════════════════════
#  Cross-DB QA generators
# ══════════════════════════════════════════════════════════════════════════════

def _cross_db_qa() -> list[dict]:
    qa: list[dict] = []

    # Bridge count
    r = _sql("retail", "SELECT COUNT(*) AS n FROM patient_customer_bridge").iloc[0]
    qa.append({
        "instruction": "How many retail customers are linked to MIMIC patient records?",
        "input": "",
        "output": (
            f"{r['n']} retail customers are linked to MIMIC-IV clinical patient records "
            f"through the patient_customer_bridge table. This cross-database link enables "
            f"combined retail and clinical analysis."
        ),
        "source": "cross", "category": "cross_db",
    })

    qa.append({
        "instruction": "What is the relationship between the retail and MIMIC databases?",
        "input": "",
        "output": (
            "The Online Retail and MIMIC-IV databases are linked through a bridge table "
            "called patient_customer_bridge, which maps retail customer_id values to "
            "MIMIC patient subject_id values. This allows analysis that combines "
            "e-commerce purchasing behaviour with clinical health data. "
            "For example, one could ask whether patients with certain diagnoses "
            "show different purchasing patterns."
        ),
        "source": "cross", "category": "cross_db",
    })

    qa.append({
        "instruction": "What two databases are used in this system and what domain does each cover?",
        "input": "",
        "output": (
            "This system uses two databases: "
            "1) Online Retail (retail.db) — UK-based e-commerce transaction data from "
            "December 2010 to December 2011, covering customers, products, orders, and revenue. "
            "2) MIMIC-IV Clinical Demo (mimic.db) — de-identified electronic health records "
            "for 100 ICU patients from Beth Israel Deaconess Medical Center, covering "
            "hospital admissions, diagnoses, lab results, medications, and ICU monitoring data."
        ),
        "source": "cross", "category": "cross_db",
    })

    qa.append({
        "instruction": "How would you combine the retail and MIMIC databases to answer a cross-domain question?",
        "input": "",
        "output": (
            "To combine data from both databases, you use the patient_customer_bridge table "
            "which exists in both databases. In retail.db, join customers.customer_id to "
            "patient_customer_bridge.customer_id to get retail purchasing data. "
            "In mimic.db, join patients.subject_id to patient_customer_bridge.subject_id "
            "to get clinical data. Then merge the results from both queries using the "
            "shared customer_id/subject_id keys."
        ),
        "source": "cross", "category": "cross_db",
    })

    return qa


# ══════════════════════════════════════════════════════════════════════════════
#  Master generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_qa_pairs() -> list[dict]:
    """Generate all QA training pairs from both databases."""
    all_qa: list[dict] = []

    print("Generating retail schema QA…")
    all_qa.extend(_retail_schema_qa())

    print("Generating retail aggregate QA…")
    all_qa.extend(_retail_aggregate_qa())

    print("Generating retail top-N QA…")
    all_qa.extend(_retail_topn_qa())

    print("Generating retail business QA…")
    all_qa.extend(_retail_business_qa())

    print("Generating MIMIC schema QA…")
    all_qa.extend(_mimic_schema_qa())

    print("Generating MIMIC aggregate QA…")
    all_qa.extend(_mimic_aggregate_qa())

    print("Generating MIMIC top-N QA…")
    all_qa.extend(_mimic_topn_qa())

    print("Generating MIMIC clinical QA…")
    all_qa.extend(_mimic_clinical_qa())

    print("Generating cross-DB QA…")
    all_qa.extend(_cross_db_qa())

    print(f"\nTotal QA pairs generated: {len(all_qa)}")
    by_cat = {}
    for qa in all_qa:
        by_cat.setdefault(qa["category"], 0)
        by_cat[qa["category"]] += 1
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<15} {n:3d}")

    return all_qa


if __name__ == "__main__":
    pairs = generate_all_qa_pairs()
    # Preview a few
    import random
    random.shuffle(pairs)
    for p in pairs[:3]:
        print(f"\n[{p['source']}/{p['category']}]")
        print(f"Q: {p['instruction']}")
        print(f"A: {p['output'][:200]}…")
