"""
Cost & Latency Tracker — per-call token accounting + budget circuit breaker.

Tracks LLM API calls, embedding operations, and cross-encoder scoring.
Circuit breaker halts pipeline if budget exceeded.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from contextlib import contextmanager

PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "local": {"input": 0.0, "output": 0.0},
}
class BudgetExceededError(Exception):
    pass
@dataclass
class APICall:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool
    error: Optional[str] = None
class CostTracker:
    def __init__(self, budget_limit: float = 5.00):
        self.budget_limit = budget_limit
        self.calls: List[APICall] = []
        self.start_time = time.time()

    def compute_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        p = PRICING.get(model, PRICING["local"])
        return round((in_tok / 1e6) * p["input"] + (out_tok / 1e6) * p["output"], 8)

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def check_budget(self, stage: str = "") -> None:
        if self.total_cost >= self.budget_limit:
            raise BudgetExceededError(
                f"Budget exceeded: ${self.total_cost:.4f} >= ${self.budget_limit:.2f} at {stage}")

    @contextmanager
    def track(self, stage: str, model: str, input_tokens: int = 0):
        self.check_budget(stage)
        call_data = {"output_tokens": 0, "success": True, "error": None}
        start = time.time()
        try:
            yield call_data
        except Exception as e:
            call_data["success"] = False
            call_data["error"] = str(e)[:200]
            raise
        finally:
            elapsed = (time.time() - start) * 1000
            cost = self.compute_cost(model, input_tokens, call_data["output_tokens"])
            self.calls.append(APICall(
                stage=stage, model=model, input_tokens=input_tokens,
                output_tokens=call_data["output_tokens"], cost_usd=cost,
                latency_ms=round(elapsed, 1), success=call_data["success"],
                error=call_data["error"]
            ))

    def record(self, stage: str, model: str, in_tok: int, out_tok: int, latency: float) -> None:
        cost = self.compute_cost(model, in_tok, out_tok)
        self.calls.append(APICall(stage, model, in_tok, out_tok, cost, latency, True))

    def get_summary(self, num_resumes: int = 1) -> dict:
        elapsed = (time.time() - self.start_time) * 1000
        total_in = sum(c.input_tokens for c in self.calls)
        total_out = sum(c.output_tokens for c in self.calls)
        n = max(num_resumes, 1)
        return {
            "total_calls": len(self.calls),
            "total_tokens": total_in + total_out,
            "total_cost_usd": round(self.total_cost, 6),
            "total_latency_ms": round(elapsed, 1),
            "budget_utilization_pct": round((self.total_cost / self.budget_limit) * 100, 2),
            "cost_per_resume": round(self.total_cost / n, 6),
            "cost_per_1000": round((self.total_cost / n) * 1000, 4),
        }

    def print_report(self, num_resumes: int = 1) -> None:
        s = self.get_summary(num_resumes)
        print(f"\n{'=' * 60}")
        print(f"  COST REPORT: {s['total_calls']} calls, {s['total_tokens']:,} tokens")
        print(f"  Total: ${s['total_cost_usd']:.6f} | Per resume: ${s['cost_per_resume']:.6f}")
        print(f"  Budget: {s['budget_utilization_pct']:.1f}% of ${self.budget_limit:.2f}")
        print(f"  Projection: ${s['cost_per_1000']:.4f} per 1,000 resumes")
        print(f"{'=' * 60}")
