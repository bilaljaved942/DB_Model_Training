"""
data_prep/dataset_builder.py
─────────────────────────────
Assembles the complete fine-tuning dataset from:
  1. Schema narration blocks (retail + MIMIC)
  2. Row-level serialized sentences
  3. QA pairs from qa_generator.py

Output format: JSONL where each line is:
  {
    "messages": [
      {"role": "system", "content": "<combined schema context>"},
      {"role": "user", "content": "<question>"},
      {"role": "assistant", "content": "<answer>"}
    ]
  }

This ChatML format works with most instruction-tuned models
(Qwen2.5, Phi-3.5, LLaMA-3, Mistral, etc.) via TRL SFTTrainer.

Also exports:
  - Alpaca format (instruction/input/output) for compatibility
  - A 90/10 train/validation split
  - A human-readable dataset_report.txt summarising what was generated

Usage
─────
    python data_prep/dataset_builder.py
    # → data/training_dataset.jsonl
    # → data/validation_dataset.jsonl
    # → data/dataset_report.txt
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

random.seed(42)

# ── Combined schema context (injected as system message) ─────────────────────

SYSTEM_CONTEXT = """You are a knowledgeable AI assistant trained on two relational databases.

DATABASE 1 — Online Retail (UK e-commerce, Dec 2010 – Dec 2011):
Tables: customers (4,372 rows) | products (1,124 rows) | invoices (25,900 rows) | invoice_items (54,873 rows)
Key facts: Customers span 38 countries. UK (Domestic_UK) is the largest segment. Revenue is quantity × unit_price.
Cancelled invoices (is_cancelled=1) are returns/refunds; confirmed revenue excludes these.

DATABASE 2 — MIMIC-IV Clinical Demo (100 ICU patients, Beth Israel Deaconess Medical Center):
Tables: patients | admissions (275 rows) | diagnoses_icd (4,506 rows) | labevents (107,727 rows) |
prescriptions (18,087 rows) | icustays (140 stays) | chartevents (668,862 measurements) | + 21 more tables
Key facts: De-identified data. hospital_expire_flag=1 means in-hospital death. seq_num=1 is primary diagnosis.
ICD codes use version 9 or 10. ICU length of stay (los) is in days.

CROSS-DATABASE LINK: patient_customer_bridge maps retail customer_id ↔ MIMIC subject_id (100 links).

Answer questions accurately based on what you were trained on. If asked for exact numbers, provide them.
If asked about schema, explain table structures and relationships clearly."""


def _make_chatml(instruction: str, output: str) -> dict:
    """Format a QA pair as a ChatML messages list."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_CONTEXT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
    }


def _make_alpaca(instruction: str, output: str, input_: str = "") -> dict:
    """Format a QA pair in Alpaca instruction-tuning format."""
    return {
        "instruction": instruction,
        "input": input_,
        "output": output,
        "system": SYSTEM_CONTEXT,
    }


def build_dataset() -> tuple[list[dict], list[dict]]:
    """
    Build the full training and validation datasets.
    Returns (train_examples, val_examples) as ChatML-formatted dicts.
    """
    import sys
    sys.path.insert(0, str(ROOT))

    from data_prep.serializer import (
        RETAIL_SCHEMA_NARRATION,
        MIMIC_SCHEMA_NARRATION,
        serialize_retail_rows,
        serialize_mimic_rows,
        generate_retail_facts,
        generate_mimic_facts,
    )
    from data_prep.qa_generator import generate_all_qa_pairs

    examples: list[dict] = []

    # ── 1. Schema understanding QA (from narration blocks) ────────────────────
    print("Building schema narration examples…")
    schema_questions = [
        ("Describe the schema of the Online Retail database.", RETAIL_SCHEMA_NARRATION),
        ("What tables are in the retail database and how do they relate?", RETAIL_SCHEMA_NARRATION),
        ("Explain the structure of the MIMIC-IV clinical database.", MIMIC_SCHEMA_NARRATION),
        ("What tables are in the MIMIC database and what does each store?", MIMIC_SCHEMA_NARRATION),
        (
            "Summarise both databases: what they contain and how they are linked.",
            RETAIL_SCHEMA_NARRATION + "\n\n" + MIMIC_SCHEMA_NARRATION
            + "\n\nThese two databases are linked via the patient_customer_bridge table.",
        ),
    ]
    for q, a in schema_questions:
        examples.append(_make_chatml(q, a))

    # ── 2. Fact-based QA from serialized rows ──────────────────────────────────
    print("Building row serialization examples…")
    retail_rows = serialize_retail_rows(n_customers=30, n_products=40, n_invoices=30)
    mimic_rows = serialize_mimic_rows(n_patients=100, n_admissions=50, n_diagnoses=80)

    # Turn batches of rows into QA pairs
    for i in range(0, len(retail_rows), 10):
        batch = retail_rows[i:i+10]
        examples.append(_make_chatml(
            f"Tell me about some retail customers and products from the database.",
            "\n".join(batch),
        ))

    for i in range(0, len(mimic_rows), 8):
        batch = mimic_rows[i:i+8]
        examples.append(_make_chatml(
            f"Describe some patients and admissions from the MIMIC clinical database.",
            "\n".join(batch),
        ))

    # ── 3. Aggregate facts ─────────────────────────────────────────────────────
    print("Building aggregate fact examples…")
    retail_facts = generate_retail_facts()
    mimic_facts = generate_mimic_facts()

    examples.append(_make_chatml(
        "Give me key statistics about the Online Retail database.",
        "\n".join(f"• {f}" for f in retail_facts),
    ))
    examples.append(_make_chatml(
        "What are the key clinical statistics from the MIMIC-IV database?",
        "\n".join(f"• {f}" for f in mimic_facts),
    ))

    # Individual fact QA
    for fact in retail_facts:
        q = _fact_to_question(fact, "retail")
        if q:
            examples.append(_make_chatml(q, fact))

    for fact in mimic_facts:
        q = _fact_to_question(fact, "mimic")
        if q:
            examples.append(_make_chatml(q, fact))

    # ── 4. Generated QA pairs ──────────────────────────────────────────────────
    print("Building generated QA pairs…")
    qa_pairs = generate_all_qa_pairs()
    for qa in qa_pairs:
        examples.append(_make_chatml(qa["instruction"], qa["output"]))

    # ── 5. Conversation-style multi-turn examples ──────────────────────────────
    print("Building conversation examples…")
    conversations = _build_conversations()
    examples.extend(conversations)

    # ── Shuffle and split ─────────────────────────────────────────────────────
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train = examples[:split_idx]
    val = examples[split_idx:]

    print(f"\nDataset summary:")
    print(f"  Total examples : {len(examples)}")
    print(f"  Training set   : {len(train)}")
    print(f"  Validation set : {len(val)}")

    return train, val


def _fact_to_question(fact: str, db: str) -> str | None:
    """Heuristically generate a question from a fact statement."""
    fact_lower = fact.lower()
    if "contain" in fact_lower and "customers" in fact_lower:
        return "How many customers are in the retail database?"
    if "total" in fact_lower and "revenue" in fact_lower:
        return "What is the total revenue in the Online Retail database?"
    if "cancellations" in fact_lower:
        return "What is the cancellation rate in the retail database?"
    if "countries" in fact_lower and "top" in fact_lower:
        return "Which countries generate the most retail revenue?"
    if "patients" in fact_lower and ("female" in fact_lower or "male" in fact_lower):
        return "What is the gender breakdown of patients in MIMIC?"
    if "mortality" in fact_lower:
        return "What is the in-hospital mortality rate in MIMIC?"
    if "icu" in fact_lower and "stay" in fact_lower and "average" in fact_lower:
        return "What is the average ICU length of stay in MIMIC?"
    if "laboratory" in fact_lower or "lab" in fact_lower:
        return "How many lab results are in the MIMIC database?"
    if "prescription" in fact_lower:
        return "How many prescriptions are in MIMIC?"
    if "bridge" in fact_lower or "linked" in fact_lower:
        return "How are the retail and MIMIC databases connected?"
    return None


def _build_conversations() -> list[dict]:
    """Build a few multi-turn style QA examples."""
    convs = []

    convs.append(_make_chatml(
        "I want to understand the retail database. Start with the basics: what data does it have and when does it cover?",
        (
            "The Online Retail database covers UK-based e-commerce transactions "
            "from December 2010 to December 2011 — approximately one full year. "
            "It contains four main tables: customers (4,372 registered shoppers from 38 countries), "
            "products (1,124 items with stock codes, descriptions, and prices), "
            "invoices (25,900 order headers, of which 3,836 are cancellations), "
            "and invoice_items (54,873 line items linking each product to each order). "
            "The database also includes a bridge table connecting retail customers to MIMIC clinical patients."
        ),
    ))

    convs.append(_make_chatml(
        "If I wanted to calculate total revenue, how should I do it and what should I be careful about?",
        (
            "To calculate total confirmed revenue from the Online Retail database, you need to: "
            "1) Join invoice_items to invoices using invoice_no. "
            "2) Filter out cancelled invoices using WHERE is_cancelled = 0 — this is critical "
            "because cancelled invoices represent returns/refunds and inflate the gross figure. "
            "3) Sum the line_total column (which is already computed as quantity × unit_price). "
            "The confirmed order revenue is £8,187,806.36. "
            "The gross total including cancellations is higher at approximately £9.7M, "
            "but this is misleading as returns are included."
        ),
    ))

    convs.append(_make_chatml(
        "What makes MIMIC-IV useful for research and what are its limitations?",
        (
            "MIMIC-IV is valuable for clinical research because it contains real, detailed "
            "electronic health records including diagnoses (ICD-coded), lab results with reference ranges, "
            "medication orders, ICU vital signs measured thousands of times per patient, "
            "and outcomes like in-hospital mortality. "
            "However, it has key limitations: "
            "1) The demo version only covers 100 patients — too small for statistically robust conclusions. "
            "2) All dates are shifted (de-identified), so real temporal trends cannot be studied. "
            "3) Patient identifiers and names are removed, preventing individual-level linking to external data. "
            "4) The 4.7% in-hospital mortality rate may not represent the full MIMIC population, "
            "as the demo is a curated subset."
        ),
    ))

    convs.append(_make_chatml(
        "What business questions could I answer by combining the retail and MIMIC databases?",
        (
            "By combining the Online Retail and MIMIC databases through the patient_customer_bridge, "
            "you could explore several cross-domain questions: "
            "1) Do patients with chronic conditions (e.g., heart failure diagnoses in MIMIC) "
            "show different purchasing patterns in retail? "
            "2) Do patients who had longer ICU stays (high LOS) tend to be high-value or low-value retail customers? "
            "3) Are there demographic patterns (age/gender from MIMIC) correlated with product category preferences? "
            "4) Do patients admitted urgently vs electively show different retail engagement? "
            "Note: With only 100 bridge links, statistical power is limited, "
            "but the methodology is sound and would scale with more data."
        ),
    ))

    return convs


def save_datasets(train: list[dict], val: list[dict]) -> None:
    """Save datasets to JSONL files and generate a report."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "training_dataset.jsonl"
    val_path = DATA_DIR / "validation_dataset.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    # Also save Alpaca format for compatibility
    alpaca_path = DATA_DIR / "training_dataset_alpaca.jsonl"
    with open(alpaca_path, "w") as f:
        for ex in train:
            msgs = ex["messages"]
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            f.write(json.dumps(_make_alpaca(user_msg, asst_msg)) + "\n")

    # Report
    report_path = DATA_DIR / "dataset_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  TRAINING DATASET REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training examples  : {len(train)}\n")
        f.write(f"Validation examples: {len(val)}\n")
        f.write(f"Total              : {len(train) + len(val)}\n\n")

        f.write("Files generated:\n")
        f.write(f"  {train_path}\n")
        f.write(f"  {val_path}\n")
        f.write(f"  {alpaca_path}\n\n")

        f.write("Format: ChatML (messages list with system/user/assistant roles)\n")
        f.write("Compatible with: Qwen2.5, Phi-3.5, LLaMA-3, Mistral (via TRL SFTTrainer)\n\n")

        # Sample
        f.write("Sample examples:\n")
        import random
        for ex in random.sample(train, min(3, len(train))):
            msgs = ex["messages"]
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            f.write(f"\nQ: {user_msg[:100]}...\n")
            f.write(f"A: {asst_msg[:200]}...\n")

    print(f"\nDatasets saved:")
    print(f"  Training  : {train_path}  ({train_path.stat().st_size//1024} KB)")
    print(f"  Validation: {val_path}  ({val_path.stat().st_size//1024} KB)")
    print(f"  Report    : {report_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    train, val = build_dataset()
    save_datasets(train, val)
    print("\n✔ Dataset generation complete.")
