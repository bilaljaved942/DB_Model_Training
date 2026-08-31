# Training a Small Language Model as a Database Replica
**Candidate:** Bilal Javed | **Submission Date:** August 31, 2026  
**To:** amad.gakkhar@adept-techsolutions.com  
**Repository:** [https://github.com/bilaljaved942/DB_Model_Training](https://github.com/bilaljaved942/DB_Model_Training)  

---

## 1. Problem Analysis: Framing & Complexity

### What Is Actually Being Asked
The directive *"make the Small Language Model (SLM) a replica of the database"* asks whether an SLM can internalize the **relational schema topology, foreign key constraints, analytical distributions, and domain business rules** of multi-table databases directly into its parametric neural weights—enabling fast semantic reasoning without requiring a live SQL database connection at inference time.

### What Makes This Hard
1. **Relational vs. Sequential Mismatch:** Language models process unstructured, sequential natural language tokens left-to-right. Relational databases are multi-dimensional graphs governed by ACID properties, primary/foreign key relationships, and strict relational algebra.
2. **Hallucination on Private Aggregates:** General-purpose base models lack private domain context and hallucinate heavily when queried on proprietary database numbers (e.g., base `Qwen2.5-Coder-7B` hallucinated 100,157 patients and $6.3M revenue based on generic web training priors).
3. **Scale Mismatch & Token Explosion:** The MIMIC database alone contains over 1.19 million rows (e.g., 668K chart events, 107K lab results). Naively dumping raw table rows into an LLM context causes context bloat, high token costs, and loss of semantic coherence.
4. **Retention vs. Lossy Approximation:** Neural networks are lossy continuous function approximators. Storing granular row-level cell values directly in weights risks catastrophic forgetting, whereas encoding **structural schema narratives and deterministic SQL aggregate distributions** enables genuine domain reasoning.

---

## 2. Proposed Solution

### A. Data Preparation & Tabular Representation Strategy
Raw SQL tables cannot be effectively learned if treated as flat CSV text. I designed a **3-Layer Tabular Serialization Pipeline**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Schema Graph Narration (Graph Topology, Tables, PKs & FKs)     │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Deterministic SQL Analytical Ground Truths (Verified Facts)   │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Cross-Database Identity Bridging (Retail Customer ↔ Patient)   │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Schema Graph Narration (Structural Topology):**  
   Each table is serialized into structured prose detailing its operational purpose, column semantics, and join paths.  
   *Example:* `"The invoices table (25,900 rows) links to customers via customer_id. The is_cancelled column flags return/refund orders (1=cancelled, 0=confirmed)."`
2. **Deterministic SQL Analytical Facts (Ground-Truth Distillation):**  
   Pre-computed SQL queries on the actual SQLite databases extract exact statistical aggregates.  
   *Example:* Confirmed revenue (£1,276,568.60 with 3,836 cancelled orders excluded), in-hospital mortality rate (5.5% across 275 admissions), abnormal lab rate (37.4% across 107,727 lab events), and average ICU length of stay (3.68 days across 140 stays).
3. **Cross-Database Identity Bridging:**  
   Serializes the `patient_customer_bridge` table (100 links), connecting e-commerce `customer_id` directly to clinical `subject_id`.
4. **ChatML Instruction-Tuning Dataset:**  
   Assembled into multi-turn conversational pairs using the standard ChatML format (`<|im_start|>system...<|im_start|>user...<|im_start|>assistant...`).

### B. Model Choice & QLoRA Fine-Tuning Architecture
* **Base Model:** `Qwen/Qwen2.5-Coder-7B-Instruct` (7.6B parameters, 4.35B active 4-bit weights). Selected for superior structured code/SQL reasoning and native Hugging Face causal LM support.
* **4-Bit QLoRA (Quantized Low-Rank Adaptation):** Base model quantized to NF4 (NormalFloat4) with double quantization. LoRA adapter matrices ($r=16, \alpha=32$) were attached to all key attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
* **Trainable Footprint:** Only **10,092,544 trainable parameters (0.231% of base weights)**, preventing catastrophic forgetting while capturing domain representations.
* **Memory Optimizations:** Gradient Checkpointing + Paged AdamW 8-bit optimizer (`paged_adamw_8bit`).

### C. Evaluation Methodology: Factual Retention vs. Approximation
To verify genuine retention rather than shallow string approximation, I implemented a **4-tier evaluation suite**:
1. **Validation Cross-Entropy Loss & Perplexity:** Monitored on held-out validation tokens.
2. **Deterministic SQL Ground-Truth Exact Match (EM):** Generated answers were scored against ground-truth values extracted directly by executing SQL on SQLite.
3. **Out-of-Distribution (OOD) Semantic Reasoning:** Tested with un-prompted questions on business rules (e.g., *"Why must we exclude `is_cancelled=0` for revenue calculations?"*) and cross-table foreign key relationships.
4. **Side-by-Side Comparative Benchmarking:** Compared the base pre-trained model directly against the fine-tuned adapter.

---

## 3. Empirical Hardware Run & Results (Tesla T4 GPU)

The entire training and evaluation pipeline was executed and validated on a single **Tesla T4 GPU (15.6 GB VRAM)** on Google Colab:
* **Peak VRAM Footprint:** **5.2 GB** (leaving >10 GB headroom).
* **Training Runtime:** **290 seconds (~4.8 minutes)** for 5 epochs (40 optimization steps).
* **Final Training Loss:** **0.7715** (Perplexity: **2.16**).

### Before vs. After Empirical Benchmark Results

| Evaluation Metric | Untrained Base Model (`Qwen2.5-Coder-7B`) | QLoRA Fine-Tuned SLM (Our Model) | Ground-Truth SQL Verification |
| :--- | :--- | :--- | :--- |
| **MIMIC Patient Count** | ❌ 100,157 *(Hallucinated)* | ✅ **100 patients** | `COUNT(*) FROM patients` $\rightarrow$ **100** |
| **In-Hospital Mortality Rate** | ❌ 18.4% *(Hallucinated)* | ✅ **5.5%** | `AVG(hospital_expire_flag)` $\rightarrow$ **5.45% (15/275)** |
| **Hospital Admissions** | ❌ 52,011 *(Hallucinated)* | ✅ **275 admissions** | `COUNT(*) FROM admissions` $\rightarrow$ **275** |
| **ICU Stays** | ❌ 2,509 *(Hallucinated)* | ✅ **140 stays** | `COUNT(*) FROM icustays` $\rightarrow$ **140** |
| **Avg ICU Length of Stay** | ❌ Generic essay on CDC/EU averages | ✅ **3.68 days** | `AVG(los) FROM icustays` $\rightarrow$ **3.68 days** |
| **Unique Retail Customers** | ❌ Generated raw SQL query | ✅ **4,372 unique customers** | `COUNT(DISTINCT customer_id)` $\rightarrow$ **4,372** |
| **Total Invoices** | ❌ 5,419 *(Hallucinated)* | ✅ **25,900 invoices** | `COUNT(*) FROM invoices` $\rightarrow$ **25,900** |
| **Confirmed Retail Revenue** | ❌ $6,307,429 *(Hallucinated)* | ✅ **£1,276,568.60** | `SUM(line_total) WHERE is_cancelled=0` |
| **Abnormal Lab Result %** | ❌ Vague general discussion | ✅ **37.4%** | `AVG(flag=='abnormal')` $\rightarrow$ **37.4%** |
| **Recorded Date of Death** | ❌ Generic SQL snippet | ✅ **31 patients** | `COUNT(dod) WHERE dod IS NOT NULL` $\rightarrow$ **31** |
| **Top Revenue Country** | ❌ "United States ($5.4T)" | ✅ **United Kingdom (£685,023)** | Top market query |
| **Database Schema Topology** | ⚠️ Partial table list | ✅ **Complete breakdown with PK/FK** | Schema graph verification |
| **Overall Benchmark Pass Rate** | **7.7%** (1/13) | **100.0%** (12/12) | **+92.3% Absolute Gain** |

---

## 4. Trade-offs, Limitations & Production Architecture

1. **Parametric Memory vs. Real-Time Data Velocity:**
   * *Limitation:* LoRA adapter weights store a static snapshot of database knowledge. In environments with millisecond transaction updates, weights do not reflect live balance changes without retraining.
   * *Production Recommendation:* Adopt a **Hybrid Architecture**: the fine-tuned SLM handles natural language query intent, schema linking, and business reasoning, and generates deterministic SQL to execute against live read replicas for real-time lookups.
2. **Macro-Distribution vs. Micro-Cell Lookup:**
   * SLMs excel at understanding distributions (mortality rates, revenue totals, schema graph joins). Querying individual specific cell values (e.g., patient #10014729 lab result at 03:00 AM) is best handled deterministically rather than relying on neural memory.
3. **Data Privacy & Governance:**
   * Storing patient-identifiable data directly in neural weights presents regulatory considerations under HIPAA/GDPR. Distilling data into aggregate statistics and non-identifiable archetypes mitigates data leakage risks.

---

## 5. Artifacts & Deliverables in Repository

1. **`DB_SLM_Training.ipynb`**: Complete end-to-end Google Colab training notebook including data preparation, 4-bit QLoRA training on Tesla T4 GPU, loss monitoring, before-vs-after comparison, and adapter export.
2. **`DB_SLM_Inference.ipynb`**: Standalone 3-step Colab inference notebook that loads pre-trained adapter weights, enables interactive streaming inference, and runs the automated 12-query benchmark suite.
3. **`db_slm_adapter.zip`** (414 MB): Trained LoRA adapter weights (`adapter_model.safetensors`, `adapter_config.json`, tokenizer configs).
4. **`data_prep/` & `database/`**: Serialization engine and schema graph extraction modules.
5. **`README.md`**: Architectural documentation and reproduction guide.
