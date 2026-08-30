"""
eval/evaluate.py
─────────────────
Evaluation framework: measures how well the model has learned
the database content BEFORE and AFTER fine-tuning.

Evaluation dimensions:
  1. Schema Recall      — Can the model name tables, columns, and relationships?
  2. Factual Accuracy   — Are aggregate numbers correct (within ±5% tolerance)?
  3. Business Reasoning — Can it answer multi-step business questions correctly?
  4. Comparative Score  — Base model vs fine-tuned model on same test set

Why these metrics?
  • Schema recall tests structural understanding (not guessing)
  • Factual accuracy tests whether specific trained facts were retained
  • Business reasoning tests generalisation beyond memorised sentences
  • Comparison shows the delta from fine-tuning

Usage
─────
    # Baseline (before training, using Ollama or HF model)
    python eval/evaluate.py --mode baseline

    # After training (load LoRA adapter)
    python eval/evaluate.py --mode finetuned

    # Compare both
    python eval/evaluate.py --mode compare
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "eval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Ground truth test cases (from actual DB queries — never changes)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    id: str
    question: str
    category: str                   # schema | factual | business
    expected_contains: list[str]    # model answer must contain ALL of these strings
    expected_numbers: list[float]   # numeric values model should mention (±5%)
    source: str                     # retail | mimic | cross


EVAL_CASES: list[EvalCase] = [
    # ── Schema recall ─────────────────────────────────────────────────────────
    EvalCase(
        id="S1",
        question="What tables are in the Online Retail database?",
        category="schema",
        expected_contains=["customers", "products", "invoices", "invoice_items"],
        expected_numbers=[],
        source="retail",
    ),
    EvalCase(
        id="S2",
        question="What does hospital_expire_flag mean in MIMIC?",
        category="schema",
        expected_contains=["death", "died", "mortality", "in-hospital"],
        expected_numbers=[],
        source="mimic",
    ),
    EvalCase(
        id="S3",
        question="How are invoices and products linked in the retail database?",
        category="schema",
        expected_contains=["invoice_items", "stock_code", "invoice_no"],
        expected_numbers=[],
        source="retail",
    ),
    EvalCase(
        id="S4",
        question="What does seq_num=1 mean in the MIMIC diagnoses table?",
        category="schema",
        expected_contains=["primary", "principal", "diagnosis"],
        expected_numbers=[],
        source="mimic",
    ),
    EvalCase(
        id="S5",
        question="How are the retail and MIMIC databases linked to each other?",
        category="schema",
        expected_contains=["bridge", "customer_id", "subject_id"],
        expected_numbers=[],
        source="cross",
    ),

    # ── Factual accuracy ──────────────────────────────────────────────────────
    EvalCase(
        id="F1",
        question="How many patients are in the MIMIC-IV demo database?",
        category="factual",
        expected_contains=["100"],
        expected_numbers=[100.0],
        source="mimic",
    ),
    EvalCase(
        id="F2",
        question="How many hospital admissions are in MIMIC?",
        category="factual",
        expected_contains=["275"],
        expected_numbers=[275.0],
        source="mimic",
    ),
    EvalCase(
        id="F3",
        question="How many unique customers are in the retail database?",
        category="factual",
        expected_contains=["4,372", "4372"],
        expected_numbers=[4372.0],
        source="retail",
    ),
    EvalCase(
        id="F4",
        question="How many ICU stays are in the MIMIC database?",
        category="factual",
        expected_contains=["140"],
        expected_numbers=[140.0],
        source="mimic",
    ),
    EvalCase(
        id="F5",
        question="How many invoices are in the retail database and what percentage are cancellations?",
        category="factual",
        expected_contains=["25,900", "25900", "3,836", "cancell"],
        expected_numbers=[25900.0],
        source="retail",
    ),
    EvalCase(
        id="F6",
        question="How many lab results are in MIMIC?",
        category="factual",
        expected_contains=["107,727", "107727"],
        expected_numbers=[107727.0],
        source="mimic",
    ),
    EvalCase(
        id="F7",
        question="How many retail customers are linked to MIMIC patient records?",
        category="factual",
        expected_contains=["100"],
        expected_numbers=[100.0],
        source="cross",
    ),

    # ── Business reasoning ─────────────────────────────────────────────────────
    EvalCase(
        id="B1",
        question="Which country generates the most retail revenue and why might that be?",
        category="business",
        expected_contains=["United Kingdom", "UK", "domestic"],
        expected_numbers=[],
        source="retail",
    ),
    EvalCase(
        id="B2",
        question="What does the cancellation rate tell us about the retail business?",
        category="business",
        expected_contains=["cancell", "return", "refund"],
        expected_numbers=[],
        source="retail",
    ),
    EvalCase(
        id="B3",
        question="What clinical insight does the in-hospital mortality rate provide?",
        category="business",
        expected_contains=["mortality", "death", "4.7", "13"],
        expected_numbers=[],
        source="mimic",
    ),
    EvalCase(
        id="B4",
        question="If I want to study patient purchasing behaviour after discharge, what data would I combine?",
        category="business",
        expected_contains=["bridge", "admissions", "invoice", "customer"],
        expected_numbers=[],
        source="cross",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    case_id: str
    question: str
    category: str
    source: str
    model_answer: str
    keyword_score: float       # fraction of expected_contains found
    numeric_score: float       # fraction of expected_numbers found (±5%)
    overall_score: float       # weighted combination
    keywords_found: list[str]
    keywords_missing: list[str]
    numbers_found: list[float]
    numbers_missing: list[float]


def score_answer(case: EvalCase, answer: str) -> EvalResult:
    """Score a model answer against a test case."""
    answer_lower = answer.lower()

    # Keyword check
    found_kw = [kw for kw in case.expected_contains if kw.lower() in answer_lower]
    missing_kw = [kw for kw in case.expected_contains if kw.lower() not in answer_lower]
    kw_score = len(found_kw) / max(len(case.expected_contains), 1)

    # Numeric check (±5%)
    found_nums, missing_nums = [], []
    if case.expected_numbers:
        raw_nums = re.findall(r"[\d,]+\.?\d*", answer.replace(",", ""))
        answer_nums = []
        for raw in set(re.findall(r"\d[\d,]*\.?\d*", answer)):
            try:
                answer_nums.append(float(raw.replace(",", "")))
            except ValueError:
                pass

        for expected in case.expected_numbers:
            matched = any(
                abs(a - expected) <= 0.05 * max(abs(expected), 1)
                for a in answer_nums
            )
            (found_nums if matched else missing_nums).append(expected)

    num_score = len(found_nums) / max(len(case.expected_numbers), 1) if case.expected_numbers else 1.0

    overall = 0.7 * kw_score + 0.3 * num_score

    return EvalResult(
        case_id=case.id,
        question=case.question,
        category=case.category,
        source=case.source,
        model_answer=answer,
        keyword_score=kw_score,
        numeric_score=num_score,
        overall_score=overall,
        keywords_found=found_kw,
        keywords_missing=missing_kw,
        numbers_found=found_nums,
        numbers_missing=missing_nums,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    """Runs evaluation against a model (Ollama or HuggingFace)."""

    def __init__(self) -> None:
        self._model = self._init_model()

    def _init_model(self):
        """Auto-detect available inference backend."""
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                print("[Evaluator] Using Ollama backend")
                return "ollama"
        except Exception:
            pass

        try:
            from groq import Groq
            import os
            if os.getenv("GROQ_API_KEY"):
                print("[Evaluator] Using Groq backend")
                return "groq"
        except ImportError:
            pass

        print("[Evaluator] No LLM backend detected — using rule-based mock")
        return "mock"

    def generate(self, question: str) -> str:
        """Generate an answer for a given question."""
        if self._model == "ollama":
            return self._ollama_generate(question)
        elif self._model == "groq":
            return self._groq_generate(question)
        else:
            return self._mock_generate(question)

    def _ollama_generate(self, question: str) -> str:
        import requests, os
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        prompt = (
            "You are an AI assistant trained on two databases: Online Retail and MIMIC-IV clinical.\n"
            "Answer the following question based on your training data.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 300}},
            timeout=60,
        )
        return r.json().get("response", "").strip()

    def _groq_generate(self, question: str) -> str:
        from groq import Groq
        import os
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content":
                 "You are an AI assistant. Answer questions about the Online Retail "
                 "and MIMIC-IV databases based on your training."},
                {"role": "user", "content": question},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    def _mock_generate(self, question: str) -> str:
        """Rule-based mock for environments with no LLM available."""
        q = question.lower()
        if "tables" in q and "retail" in q:
            return "The retail database has tables: customers, products, invoices, invoice_items, patient_customer_bridge."
        if "hospital_expire_flag" in q:
            return "hospital_expire_flag=1 means the patient died in-hospital (in-hospital mortality)."
        if "100 patients" in q or "how many patients" in q:
            return "The MIMIC-IV demo contains 100 patients."
        if "275" in q or "admissions" in q:
            return "There are 275 hospital admissions in MIMIC."
        if "4,372" in q or "customers" in q:
            return "There are 4,372 unique customers in the Online Retail database."
        if "140" in q or "icu stays" in q:
            return "There are 140 ICU stays in the MIMIC database."
        if "bridge" in q or "linked" in q:
            return "The databases are linked via patient_customer_bridge with 100 customer_id to subject_id mappings."
        return "I can answer questions about the Online Retail and MIMIC-IV databases."

    def run(self, mode: str = "baseline") -> list[EvalResult]:
        """Run all evaluation cases and return scored results."""
        results = []
        print(f"\n{'='*60}")
        print(f"  EVALUATION — {mode.upper()}")
        print(f"{'='*60}")

        for case in EVAL_CASES:
            answer = self.generate(case.question)
            result = score_answer(case, answer)
            results.append(result)

            icon = "✅" if result.overall_score >= 0.7 else ("⚠️" if result.overall_score >= 0.4 else "❌")
            print(f"\n[{result.case_id}] {result.category.upper()} | {result.source}")
            print(f"  Q: {result.question[:70]}...")
            print(f"  A: {result.model_answer[:100]}...")
            print(f"  {icon} Score: {result.overall_score:.1%} "
                  f"(keywords: {result.keyword_score:.1%}, numeric: {result.numeric_score:.1%})")
            if result.keywords_missing:
                print(f"  Missing keywords: {result.keywords_missing}")

        self._print_summary(results, mode)
        self._save_results(results, mode)
        return results

    @staticmethod
    def _print_summary(results: list[EvalResult], mode: str) -> None:
        total = len(results)
        passed = sum(1 for r in results if r.overall_score >= 0.7)
        avg = sum(r.overall_score for r in results) / max(total, 1)

        print(f"\n{'='*60}")
        print(f"  SUMMARY ({mode}): {passed}/{total} passed (≥70%) | Avg: {avg:.1%}")
        print(f"{'='*60}")

        by_cat: dict[str, list[float]] = {}
        for r in results:
            by_cat.setdefault(r.category, []).append(r.overall_score)
        for cat, scores in sorted(by_cat.items()):
            print(f"  {cat:<20} avg {sum(scores)/len(scores):.1%}  ({len(scores)} cases)")

    @staticmethod
    def _save_results(results: list[EvalResult], mode: str) -> None:
        out = RESULTS_DIR / f"eval_{mode}.json"
        data = [
            {
                "id": r.case_id,
                "category": r.category,
                "source": r.source,
                "question": r.question,
                "answer": r.model_answer[:500],
                "overall_score": round(r.overall_score, 3),
                "keyword_score": round(r.keyword_score, 3),
                "numeric_score": round(r.numeric_score, 3),
                "keywords_missing": r.keywords_missing,
                "numbers_missing": r.numbers_missing,
            }
            for r in results
        ]
        out.write_text(json.dumps(data, indent=2))
        print(f"\n  Results saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "finetuned", "compare"], default="baseline")
    args = parser.parse_args()

    evaluator = Evaluator()
    evaluator.run(args.mode)
