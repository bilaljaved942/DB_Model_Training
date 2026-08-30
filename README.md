# Multi-Database Small Language Model (SLM) for Schema & Business Logic Reasoning

> **Candidate:** Bilal Javed  
> **Evaluation:** AI Engineer Leadership Evaluation Task — Adept Tech Solutions (Dr. Wasim)  
> **Dataset Sources:** [MIMIC-IV Clinical Database Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/) & [UCI Online Retail Database](https://archive.ics.uci.edu/dataset/352/online+retail)

---

## 📌 Executive Summary

This repository implements an end-to-end framework for **training a Small Language Model (SLM) on multi-relational enterprise databases**, allowing the model to act as a **domain-grounded replica** of the underlying data. 

Rather than relying on vector search (RAG)—which fundamentally lacks the ability to understand relational foreign keys, schema topology, and SQL aggregations—this solution transforms structured databases through a **3-tiered Relational Knowledge Distillation pipeline** and fine-tunes **`Qwen2.5-Coder-7B-Instruct`** using **QLoRA (4-bit quantization)**.

---

## 📊 Key Empirical Results (Google Colab T4 GPU)

| Metric | Base Model (Before Training) | QLoRA Fine-Tuned (After Training) | Impact |
|---|---|---|---|
| **Overall Score** | **7.7%** (0/13 passed) | **70.5%** (5/13 passed, 8/13 partial) | **+62.8% absolute gain** |
| **MIMIC Patient Count** | ❌ Hallucinated *"10,000 patients"* | ✅ Exact: **100 patients** | Ground-truth internalized |
| **In-Hospital Mortality Rate** | ❌ Vague / ungrounded | ✅ Exact: **5.5%** (15 / 275 admissions) | Computed metric retained |
| **Retail Revenue & Top Market** | ❌ Hallucinated *"United States"* | ✅ Exact: **UK with £685,023 confirmed** | Domain rules learned |
| **Customer Base (Retail)** | ❌ Evasive (*"query the database..."*) | ✅ Exact: **4,372 unique customers** | Factual recall |
| **Schema & Relational Breakdown** | ⚠️ Generic / incomplete | ✅ **Full schema topology with PK/FK links** | Structural mastery |
| **Training Efficiency** | — | **5 epochs in 290s (~4.8 min), Final Loss: 0.7715** | Fast & stable convergence |

---

## 🏗️ System Architecture & Methodology

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          1. RELATIONAL DATABASES                          │
│   • retail.db (541K rows, 4,372 customers, 1,124 products, 25.9K invoices) │
│   • mimic.db  (1.19M rows, 100 patients, 275 admissions, 140 ICU stays)   │
│   • patient_customer_bridge (100 cross-domain links: customer_id ↔ subject_id)│
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼─────────────────────────────────────┐
│               2. 3-TIERED KNOWLEDGE DISTILLATION PIPELINE                 │
│  Layer 1: Structural Schema Narratives (teaches table topology & PK/FK)   │
│  Layer 2: SQL-Derived Ground Truths (pre-computes exact totals & rates)   │
│  Layer 3: Serialized Instance Archetypes (teaches row-level syntax)       │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼─────────────────────────────────────┐
│                 3. QLoRA FINE-TUNING (Qwen2.5-Coder-7B)                   │
│  • Base Model: Qwen/Qwen2.5-Coder-7B-Instruct (4-bit NF4 Quantization)     │
│  • Trainable Parameters: 10,092,544 out of 4.36B (0.231% of weights)       │
│  • Memory Optimizations: gradient_checkpointing + paged_adamw_8bit         │
│  • VRAM Footprint: ~5.2 GB on Tesla T4 (leaving ~10 GB free headroom)     │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼─────────────────────────────────────┐
│                    4. EVALUATION & INFERENCE CHAT UI                      │
│  • Automated Before vs. After Benchmark (13 multi-dimensional tests)      │
│  • Streamlit Interactive Chat Dashboard (app.py)                          │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Database Entities & Business Logic

### 1. Online Retail Database (`retail.db`)
* **`customers` (4,372 rows)**: Customer demographics across 38 countries (`Domestic_UK`, `EU_Wholesale`, `International`).
* **`products` (1,124 rows)**: Product catalog with stock codes, descriptions, categories, and unit prices.
* **`invoices` (25,900 rows)**: Transaction headers.
  * **Core Business Logic**: `is_cancelled = 1` indicates refunds/returns (invoice numbers start with 'C'). True confirmed revenue (£1,276,568.60) **must exclude cancellations**.
* **`invoice_items` (54,873 rows)**: Transaction line items linking `invoice_no` to `stock_code`.

### 2. MIMIC-IV Clinical Demo Database (`mimic.db`)
* **`patients` (100 rows)**: Patient demographics (43 female, 57 male, mean anchor age: 61.8 years, 31 deceased).
* **`admissions` (275 rows)**: Hospital admissions.
  * **Core Clinical Logic**: `hospital_expire_flag = 1` indicates in-hospital mortality (15 / 275 = **5.5% mortality rate**).
* **`diagnoses_icd` (4,506 rows)**: ICD-9 & ICD-10 diagnosis codes. `seq_num = 1` represents the primary reason for admission.
* **`icustays` (140 rows)**: Intensive care unit stays across MICU, CVICU, SICU (Mean length of stay: **3.68 days**).
* **`labevents` (107,727 rows)**: Laboratory test results (**37.4% flagged abnormal**).
* **`chartevents` (668,862 rows)**: Real-time ICU monitor vital signs.

### 3. Cross-Database Bridge (`patient_customer_bridge`)
* 100 verified links connecting `customer_id ↔ subject_id`, enabling cross-domain analysis (e.g., correlating clinical diagnoses with retail shopping behaviors).

---

## 🤖 Why `Qwen2.5-Coder-7B-Instruct`?

1. **Code & SQL Specialization**: Pre-trained on **5.5 Trillion tokens** with heavy emphasis on SQL dialects, schema reasoning, and structured data manipulation.
2. **Native Hugging Face Architecture**: Uses standard `Qwen2ForCausalLM` without custom remote code, avoiding KV-cache compatibility issues.
3. **Hardware Fit for Free Tier GPU**: With 4-bit QLoRA, it occupies only **~5.2 GB VRAM**, training in under 5 minutes on a standard **Tesla T4 (15.6 GB VRAM)**.

---

## 📁 Repository Structure

```
DB_Training_Task/
├── DB_SLM_Training.ipynb     # Complete self-contained Google Colab notebook (executed & verified)
├── SUBMISSION_WRITEUP.md      # 1-2 page technical report for leadership evaluation
├── README.md                  # Project overview, architecture, and reproduction guide
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── data_prep/                 # Data Transformation Pipeline
│   ├── serializer.py          # Tabular-to-text conversion (schema narratives & row archetypes)
│   ├── qa_generator.py        # SQL-grounded QA generation (44 verified pairs)
│   └── dataset_builder.py     # Assembles ChatML JSONL training datasets
│
├── database/                  # Database Ingestion & Schema Extraction
│   ├── db_setup.py            # Loads raw Excel & CSVs into normalized SQLite databases
│   └── schema_graph.py        # Runtime schema introspection
│
├── training/                  # Training Scripts & Generators
│   ├── train.py               # Local QLoRA fine-tuning script
│   └── generate_colab.py      # Programmatic generator for the Colab notebook
│
├── eval/                      # Evaluation Suite
│   ├── evaluate.py            # Before vs. After benchmark scoring framework
│   └── eval_baseline.json     # Baseline scoring logs
│
└── db_slm_adapter/            # Exported LoRA Adapter Weights (Safetensors & Configs)
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── tokenizer.json
    └── training_summary.json
```

---

## 🚀 Step-by-Step Reproduction Guide

### Option A: Run in Google Colab (Recommended — GPU)
1. Open [Google Colab](https://colab.research.google.com).
2. Upload [`DB_SLM_Training.ipynb`](DB_SLM_Training.ipynb).
3. Set runtime: **Runtime** ➔ **Change runtime type** ➔ **T4 GPU**.
4. Click **Runtime** ➔ **Run all**. The entire data prep, baseline test, QLoRA training, before/after evaluation, and live querying in Step 12 will execute in ~5 minutes.

### Option B: Local Data Preparation Pipeline

```bash
# 1. Clone repository & create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate SQLite databases from raw data
python database/db_setup.py

# 4. Generate the instruction-tuning datasets (JSONL)
python data_prep/dataset_builder.py
```

---

## 📄 License & Compliance
* Built for technical interview evaluation at Adept Tech Solutions.
* MIMIC-IV demo data is de-identified and subject to PhysioNet Open Data guidelines.
