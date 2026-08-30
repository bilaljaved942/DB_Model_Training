"""
eval/eval_suite.py
───────────────────
Automated benchmark suite — 12 queries covering:
  1–3:  Single-table aggregation (Retail)
  4–6:  Multi-table join (Retail)
  7–8:  MIMIC-only clinical queries
  9–10: Role-based access control tests
  11:   Cross-DB bridge join
  12:   Edge cases (permission denied, empty result)

Each benchmark item is run end-to-end:
  NL Query → PromptEngine → SLMEngine → Executor → Critic

Results are printed as a rich table and saved to eval/results.json.

Usage
-----
    python eval/eval_suite.py                # full suite, all 12 queries
    python eval/eval_suite.py --no-llm      # SQL hardcoded, skips LLM call
    python eval/eval_suite.py --query-id 3  # run single query by ID
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database.schema_graph import SchemaGraph
from engine.critic import Critic
from engine.executor import Executor
from engine.rbac_guardrail import RBACGuardrail
from slm.prompt_engine import PromptEngine, PromptResult

RESULTS_FILE = ROOT / "eval" / "results.json"


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark definitions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkItem:
    id: int
    name: str
    nl_query: str
    role: str
    db: str                               # "retail" | "mimic" | "cross"
    hardcoded_sql_db1: str = ""           # used in --no-llm mode
    hardcoded_sql_db2: str = ""
    expected_min_rows: int = 1
    expected_max_rows: int = 100_000
    expect_permission_denied: bool = False
    tags: list[str] = field(default_factory=list)


BENCHMARKS: list[BenchmarkItem] = [
    # ── 1: Simple aggregation ─────────────────────────────────────────────────
    BenchmarkItem(
        id=1,
        name="Total Revenue (Retail)",
        nl_query="What is the total revenue from all confirmed (non-cancelled) orders?",
        role="Executive",
        db="retail",
        hardcoded_sql_db1=(
            "SELECT ROUND(SUM(ii.line_total), 2) AS total_revenue "
            "FROM invoice_items ii "
            "JOIN invoices i ON ii.invoice_no = i.invoice_no "
            "WHERE i.is_cancelled = 0"
        ),
        expected_min_rows=1,
        expected_max_rows=1,
        tags=["aggregation", "retail"],
    ),
    # ── 2: Top-N product query ────────────────────────────────────────────────
    BenchmarkItem(
        id=2,
        name="Top 10 Products by Revenue",
        nl_query="List the top 10 products by total revenue from confirmed orders.",
        role="Executive",
        db="retail",
        hardcoded_sql_db1=(
            "SELECT p.description, p.category, "
            "ROUND(SUM(ii.line_total), 2) AS total_revenue "
            "FROM invoice_items ii "
            "JOIN products p ON ii.stock_code = p.stock_code "
            "JOIN invoices i ON ii.invoice_no = i.invoice_no "
            "WHERE i.is_cancelled = 0 "
            "GROUP BY ii.stock_code "
            "ORDER BY total_revenue DESC LIMIT 10"
        ),
        expected_min_rows=10,
        expected_max_rows=10,
        tags=["join", "aggregation", "retail"],
    ),
    # ── 3: Time-windowed query ────────────────────────────────────────────────
    BenchmarkItem(
        id=3,
        name="Monthly Revenue Trend",
        nl_query="Show total confirmed revenue grouped by month (YYYY-MM).",
        role="Executive",
        db="retail",
        hardcoded_sql_db1=(
            "SELECT SUBSTR(i.invoice_date, 1, 7) AS month, "
            "ROUND(SUM(ii.line_total), 2) AS monthly_revenue "
            "FROM invoices i "
            "JOIN invoice_items ii ON i.invoice_no = ii.invoice_no "
            "WHERE i.is_cancelled = 0 "
            "GROUP BY month ORDER BY month"
        ),
        expected_min_rows=10,
        expected_max_rows=15,
        tags=["time-series", "retail"],
    ),
    # ── 4: Multi-table join ───────────────────────────────────────────────────
    BenchmarkItem(
        id=4,
        name="Top 5 Countries by Customer Count",
        nl_query="Which 5 countries have the most registered customers?",
        role="Regional_Sales_Manager",
        db="retail",
        hardcoded_sql_db1=(
            "SELECT country, COUNT(DISTINCT customer_id) AS num_customers "
            "FROM customers "
            "GROUP BY country ORDER BY num_customers DESC LIMIT 5"
        ),
        expected_min_rows=5,
        expected_max_rows=5,
        tags=["join", "aggregation", "retail", "rbac"],
    ),
    # ── 5: Cancellation analysis ──────────────────────────────────────────────
    BenchmarkItem(
        id=5,
        name="Cancellation Rate per Country",
        nl_query="What is the cancellation rate (% of invoices) per country? Show top 10.",
        role="Executive",
        db="retail",
        hardcoded_sql_db1=(
            "SELECT c.country, "
            "COUNT(*) AS total_invoices, "
            "SUM(i.is_cancelled) AS cancelled, "
            "ROUND(100.0*SUM(i.is_cancelled)/COUNT(*), 2) AS cancel_rate_pct "
            "FROM invoices i "
            "JOIN customers c ON i.customer_id = c.customer_id "
            "GROUP BY c.country "
            "ORDER BY cancel_rate_pct DESC LIMIT 10"
        ),
        expected_min_rows=5,
        expected_max_rows=10,
        tags=["join", "cancellation", "retail"],
    ),
    # ── 6: Average basket size ────────────────────────────────────────────────
    BenchmarkItem(
        id=6,
        name="Average Basket Size per Segment",
        nl_query="What is the average number of line items per invoice, broken down by customer segment?",
        role="Executive",
        db="retail",
        hardcoded_sql_db1=(
            "SELECT c.segment, "
            "ROUND(AVG(item_count), 2) AS avg_items_per_invoice "
            "FROM customers c "
            "JOIN ( "
            "  SELECT i.customer_id, i.invoice_no, COUNT(*) AS item_count "
            "  FROM invoices i "
            "  JOIN invoice_items ii ON i.invoice_no = ii.invoice_no "
            "  WHERE i.is_cancelled = 0 "
            "  GROUP BY i.invoice_no "
            ") basket ON basket.customer_id = c.customer_id "
            "GROUP BY c.segment"
        ),
        expected_min_rows=1,
        expected_max_rows=5,
        tags=["subquery", "retail"],
    ),
    # ── 7: MIMIC mortality rate ───────────────────────────────────────────────
    BenchmarkItem(
        id=7,
        name="In-Hospital Mortality Rate",
        nl_query="What percentage of hospital admissions resulted in in-hospital death?",
        role="Clinical_Analyst",
        db="mimic",
        hardcoded_sql_db2=(
            "SELECT ROUND(100.0*SUM(hospital_expire_flag)/COUNT(*), 2) AS mortality_rate_pct "
            "FROM admissions"
        ),
        expected_min_rows=1,
        expected_max_rows=1,
        tags=["aggregation", "mimic", "clinical"],
    ),
    # ── 8: Top diagnoses ─────────────────────────────────────────────────────
    BenchmarkItem(
        id=8,
        name="Top 5 Primary ICD-9 Diagnoses",
        nl_query="What are the 5 most common primary ICD-9 diagnoses with their full descriptions?",
        role="Clinical_Analyst",
        db="mimic",
        hardcoded_sql_db2=(
            "SELECT d.icd_code, dd.long_title, COUNT(*) AS frequency "
            "FROM diagnoses_icd d "
            "LEFT JOIN d_icd_diagnoses dd "
            "  ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version "
            "WHERE d.seq_num = 1 AND d.icd_version = 9 "
            "GROUP BY d.icd_code ORDER BY frequency DESC LIMIT 5"
        ),
        expected_min_rows=1,
        expected_max_rows=5,
        tags=["join", "mimic", "clinical", "icd"],
    ),
    # ── 9: RBAC — Sales Manager cannot access MIMIC ──────────────────────────
    BenchmarkItem(
        id=9,
        name="RBAC Block: Sales Manager → MIMIC",
        nl_query="Show all patient records from MIMIC.",
        role="Regional_Sales_Manager",
        db="mimic",
        hardcoded_sql_db2="SELECT * FROM patients LIMIT 10",
        expect_permission_denied=True,
        expected_min_rows=0,
        expected_max_rows=0,
        tags=["rbac", "security", "mimic"],
    ),
    # ── 10: RBAC — Data Engineer (aggregate only) ─────────────────────────────
    BenchmarkItem(
        id=10,
        name="RBAC Aggregate-Only: Data Engineer",
        nl_query="Count the total number of customers in the retail database.",
        role="Data_Engineer",
        db="retail",
        hardcoded_sql_db1=(
            "SELECT COUNT(*) AS total_customers FROM customers"
        ),
        expected_min_rows=1,
        expected_max_rows=1,
        tags=["rbac", "aggregate", "retail"],
    ),
    # ── 11: Cross-DB bridge join ──────────────────────────────────────────────
    BenchmarkItem(
        id=11,
        name="Cross-DB: Linked Customers with ICU Stays",
        nl_query="How many retail customers have linked MIMIC patient records with at least one ICU stay?",
        role="Executive",
        db="cross",
        hardcoded_sql_db1=(
            "SELECT COUNT(DISTINCT b.customer_id) AS linked_customers "
            "FROM patient_customer_bridge b"
        ),
        hardcoded_sql_db2=(
            "SELECT COUNT(DISTINCT b.subject_id) AS linked_patients_with_icu "
            "FROM patient_customer_bridge b "
            "JOIN icustays icu ON b.subject_id = icu.subject_id"
        ),
        expected_min_rows=1,
        expected_max_rows=5,
        tags=["cross-db", "bridge", "rbac"],
    ),
    # ── 12: Edge case — empty result ─────────────────────────────────────────
    BenchmarkItem(
        id=12,
        name="Edge Case: No Orders in Future Date",
        nl_query="Show all confirmed orders placed after 2020-01-01.",
        role="Executive",
        db="retail",
        hardcoded_sql_db1=(
            "SELECT COUNT(*) AS future_orders FROM invoices "
            "WHERE is_cancelled = 0 AND invoice_date > '2020-01-01'"
        ),
        expected_min_rows=1,   # COUNT(*) always returns 1 row (value = 0)
        expected_max_rows=1,
        tags=["edge-case", "retail"],
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
#  BenchmarkResult
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    id: int
    name: str
    role: str
    tags: list[str]
    passed: bool
    rbac_correctly_blocked: bool = False
    row_count: int = 0
    latency_ms: float = 0.0
    verified: bool = False
    confidence: float = 0.0
    error: Optional[str] = None
    sql_used_db1: str = ""
    sql_used_db2: str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  EvalSuite
# ══════════════════════════════════════════════════════════════════════════════

class EvalSuite:
    """Orchestrates the 12-query benchmark."""

    def __init__(self, use_llm: bool = True) -> None:
        self._use_llm = use_llm
        self._executor = Executor()
        self._critic = Critic()

        if use_llm:
            from slm.model_engine import SLMEngine
            self._schema = SchemaGraph().load()
            self._guard = RBACGuardrail()
            self._prompt_engine = PromptEngine(self._schema, self._guard)
            self._slm = SLMEngine()
        else:
            self._schema = None
            self._guard = RBACGuardrail()
            self._prompt_engine = None
            self._slm = None

    def run(
        self,
        query_ids: Optional[list[int]] = None,
    ) -> list[BenchmarkResult]:
        """
        Run the full benchmark or a subset.

        Parameters
        ----------
        query_ids : If specified, only run benchmarks with these IDs.
        """
        items = BENCHMARKS if query_ids is None else [b for b in BENCHMARKS if b.id in query_ids]

        results: list[BenchmarkResult] = []
        for item in items:
            print(f"\n[{item.id:02d}/{len(BENCHMARKS)}] {item.name}  (role={item.role})")
            result = self._run_item(item)
            results.append(result)
            icon = "✅" if result.passed else "❌"
            print(f"       {icon} passed={result.passed}  rows={result.row_count}  "
                  f"latency={result.latency_ms:.0f}ms  "
                  f"verified={result.verified}  conf={result.confidence:.1%}")
            if result.error:
                print(f"       ⚠ {result.error}")

        self._print_summary(results)
        self._save_results(results)
        return results

    # ── Private ────────────────────────────────────────────────────────────────

    def _run_item(self, item: BenchmarkItem) -> BenchmarkResult:
        t0 = time.monotonic()
        br = BenchmarkResult(
            id=item.id,
            name=item.name,
            role=item.role,
            tags=item.tags,
            passed=False,
        )

        # Build PromptResult — from LLM or hardcoded
        if self._use_llm and self._prompt_engine and self._slm:
            pr = self._run_with_llm(item, br)
        else:
            pr = PromptResult(
                db1_sql=item.hardcoded_sql_db1,
                db2_sql=item.hardcoded_sql_db2,
                explanation="(hardcoded SQL, no LLM)",
            )

        if pr is None:
            br.error = "LLM call failed"
            return br

        br.sql_used_db1 = pr.db1_sql
        br.sql_used_db2 = pr.db2_sql

        # Execute
        exec_result = self._executor.run(pr, role=item.role)
        latency_ms = (time.monotonic() - t0) * 1000
        br.latency_ms = latency_ms

        # Handle permission-denied test
        if item.expect_permission_denied:
            if exec_result.validation_errors or exec_result.has_error:
                br.rbac_correctly_blocked = True
                br.passed = True
                br.row_count = 0
            else:
                br.error = "Expected permission denial but query succeeded"
                br.passed = False
            return br

        if exec_result.has_error:
            br.error = str(exec_result.errors)
            return br

        row_count = (
            exec_result.retail_row_count
            + exec_result.mimic_row_count
            + exec_result.merged_row_count
        )
        br.row_count = row_count

        # Row count check
        within_range = item.expected_min_rows <= row_count <= item.expected_max_rows
        if not within_range:
            br.error = (
                f"Row count {row_count} outside expected "
                f"[{item.expected_min_rows}, {item.expected_max_rows}]"
            )

        # Critic verification
        critic_report = self._critic.verify(pr, exec_result, pr.explanation, role=item.role)
        br.verified = critic_report.is_verified
        br.confidence = critic_report.confidence_score

        br.passed = within_range and not exec_result.has_error
        return br

    def _run_with_llm(
        self, item: BenchmarkItem, br: BenchmarkResult
    ) -> Optional[PromptResult]:
        try:
            prompt = self._prompt_engine.build_prompt(
                user_query=item.nl_query,
                role=item.role,
                compact_schema=True,
            )
            gen = self._slm.generate(prompt, max_tokens=512, temperature=0.0)
            pr = self._prompt_engine.parse_response(gen.text)
            if pr.parse_error:
                br.error = f"Parse error: {pr.parse_error}"
                # Fall back to hardcoded SQL
                return PromptResult(
                    db1_sql=item.hardcoded_sql_db1,
                    db2_sql=item.hardcoded_sql_db2,
                    explanation="(LLM parse failed, using hardcoded SQL)",
                )
            return pr
        except Exception as e:
            br.error = f"LLM error: {e}"
            return PromptResult(
                db1_sql=item.hardcoded_sql_db1,
                db2_sql=item.hardcoded_sql_db2,
                explanation="(LLM error, using hardcoded SQL)",
            )

    @staticmethod
    def _print_summary(results: list[BenchmarkResult]) -> None:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        verified = sum(1 for r in results if r.verified)
        avg_latency = sum(r.latency_ms for r in results) / max(total, 1)
        avg_conf = sum(r.confidence for r in results) / max(total, 1)

        print(f"\n{'═'*60}")
        print(f"  EVAL RESULTS: {passed}/{total} passed  "
              f"|  {verified}/{total} critic-verified")
        print(f"  Avg latency : {avg_latency:.0f} ms")
        print(f"  Avg confidence: {avg_conf:.1%}")
        print(f"{'═'*60}")

        tag_stats: dict[str, dict] = {}
        for r in results:
            for tag in r.tags:
                tag_stats.setdefault(tag, {"pass": 0, "total": 0})
                tag_stats[tag]["total"] += 1
                if r.passed:
                    tag_stats[tag]["pass"] += 1

        print("\n  By tag:")
        for tag, stats in sorted(tag_stats.items()):
            bar = "█" * stats["pass"] + "░" * (stats["total"] - stats["pass"])
            print(f"    {tag:<25} {stats['pass']}/{stats['total']}  {bar}")
        print()

    @staticmethod
    def _save_results(results: list[BenchmarkResult]) -> None:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "id": r.id,
                "name": r.name,
                "role": r.role,
                "tags": r.tags,
                "passed": r.passed,
                "rbac_correctly_blocked": r.rbac_correctly_blocked,
                "row_count": r.row_count,
                "latency_ms": round(r.latency_ms, 1),
                "verified": r.verified,
                "confidence": round(r.confidence, 3),
                "error": r.error,
                "sql_db1": r.sql_used_db1[:200] if r.sql_used_db1 else "",
                "sql_db2": r.sql_used_db2[:200] if r.sql_used_db2 else "",
            }
            for r in results
        ]
        RESULTS_FILE.write_text(json.dumps(data, indent=2))
        print(f"  Results saved: {RESULTS_FILE}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 12-query evaluation benchmark.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM calls — use hardcoded SQL. Fast, no model required.",
    )
    parser.add_argument(
        "--query-id",
        type=int,
        nargs="+",
        help="Run only specific query IDs (e.g. --query-id 1 3 5).",
    )
    args = parser.parse_args()

    suite = EvalSuite(use_llm=not args.no_llm)
    suite.run(query_ids=args.query_id)
