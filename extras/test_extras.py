"""Tests for extras: compliance, MCP server, index store, cost tracker, feedback."""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extras.compliance import (
    compute_hash, compute_selection_rates, compute_impact_ratios,
    check_four_fifths_rule, generate_bias_audit_report, AuditRecord,
    create_audit_record, verify_reproducibility, compute_config_hash,
)
from extras.mcp_server import MCPServer, TOOL_REGISTRY, RESOURCE_REGISTRY
from src.retrieval.index_store import IndexStore, compute_corpus_hash
from src.contracts import MatchResult, ThreatReport, MatchRationale

try:
    from extras.cost_tracker import CostTracker, BudgetExceededError
    HAS_COST_TRACKER = True
except ImportError:
    HAS_COST_TRACKER = False

try:
    from extras.feedback import RecruiterFeedback, FeedbackStore, FeedbackAnalyzer, apply_weight_adjustments
    HAS_FEEDBACK = True
except ImportError:
    HAS_FEEDBACK = False

# ============================================================
# Cost Tracker
# ============================================================

@pytest.mark.skipif(not HAS_COST_TRACKER, reason="cost_tracker not installed")
class TestCostTracker:
    def test_basic(self):
        """Recording an LLM call accumulates non-zero total cost."""
        t = CostTracker(10.0)
        t.record("test", "gpt-4o-mini", 1000, 200, 50.0)
        assert t.total_cost > 0

    def test_local_free(self):
        """Local/open-source model calls are tracked at zero cost."""
        t = CostTracker()
        assert t.compute_cost("local", 10000, 0) == 0.0

    def test_circuit_breaker(self):
        """Exceeding the budget cap raises BudgetExceededError on next check — hard stop."""
        t = CostTracker(0.0001)
        t.record("test", "gpt-4o", 1_000_000, 500_000, 100.0)
        with pytest.raises(BudgetExceededError):
            t.check_budget("next")

    def test_context_manager(self):
        """Context manager pattern records the call with correct input token count."""
        t = CostTracker(10.0)
        with t.track("ext", "gpt-4o-mini", 500) as c:
            c["output_tokens"] = 150
        assert len(t.calls) == 1
        assert t.calls[0].input_tokens == 500

    def test_summary(self):
        """get_summary() returns cost_per_resume and cost_per_1000 — used in dashboard reporting."""
        t = CostTracker(10.0)
        t.record("ext", "gpt-4o-mini", 500, 100, 50.0)
        s = t.get_summary(num_resumes=5)
        assert s["cost_per_resume"] > 0
        assert s["cost_per_1000"] > 0


# ============================================================
# Feedback System
# ============================================================

@pytest.mark.skipif(not HAS_FEEDBACK, reason="feedback module not installed")
class TestFeedback:
    def test_store_and_load(self):
        """Recruiter feedback is persisted to disk and reloaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeedbackStore(tmpdir)
            fb = RecruiterFeedback(
                job_id="jd1", resume_id="r1", candidate_name="Alice",
                ai_score=0.8, ai_rank=1, recruiter_decision="ADVANCE",
                recruiter_relevance=0.9
            )
            store.record(fb)
            loaded = store.load_all()
            assert len(loaded) == 1
            assert loaded[0]["resume_id"] == "r1"

    def test_weight_adjustment_bounded(self):
        """Weight adjustments from feedback are capped at ±0.10 and re-normalised to sum to 1.0."""
        base = {"W_HARD_SKILLS": 0.4, "W_EXPERIENCE_CONTEXT": 0.35,
                "W_TRANSFERABILITY": 0.15, "W_CONSTRAINTS": 0.10}
        context = {"weight_adjustments": [
            {"target": "W_HARD_SKILLS", "delta": -0.20}
        ]}
        adjusted = apply_weight_adjustments(base, context)
        assert adjusted["W_HARD_SKILLS"] < base["W_HARD_SKILLS"]
        assert sum(adjusted.values()) == pytest.approx(1.0, abs=0.01)

    def test_no_adjustments_passthrough(self):
        """Empty adjustment list returns weights unchanged — safe default."""
        base = {"W_A": 0.5, "W_B": 0.5}
        adjusted = apply_weight_adjustments(base, {"weight_adjustments": []})
        assert adjusted == base


# ============================================================
# Compliance Module
# ============================================================

class TestHashing:
    def test_deterministic(self):
        assert compute_hash("hello") == compute_hash("hello")

    def test_different_inputs(self):
        assert compute_hash("hello") != compute_hash("world")

    def test_config_hash(self):
        h = compute_config_hash()
        assert isinstance(h, str)
        assert len(h) > 0
class TestSelectionRates:
    def test_all_selected(self):
        rates = compute_selection_rates({"A": [0.8, 0.9, 0.7]}, threshold=0.5)
        assert rates["A"] == 1.0

    def test_none_selected(self):
        rates = compute_selection_rates({"A": [0.1, 0.2, 0.3]}, threshold=0.5)
        assert rates["A"] == 0.0

    def test_partial(self):
        rates = compute_selection_rates({"A": [0.8, 0.3, 0.6]}, threshold=0.5)
        assert rates["A"] == pytest.approx(2/3, abs=0.01)
class TestImpactRatios:
    def test_equal_rates(self):
        ratios = compute_impact_ratios({"A": 0.8, "B": 0.8})
        assert ratios["A"] == 1.0
        assert ratios["B"] == 1.0

    def test_disparate(self):
        ratios = compute_impact_ratios({"A": 1.0, "B": 0.5})
        assert ratios["B"] == 0.5

    def test_empty(self):
        assert compute_impact_ratios({}) == {}
class TestFourFifthsRule:
    def test_compliant(self):
        violations = check_four_fifths_rule({"A": 1.0, "B": 0.85})
        assert len(violations) == 0

    def test_violation(self):
        violations = check_four_fifths_rule({"A": 1.0, "B": 0.6})
        assert len(violations) == 1
        assert violations[0]["group"] == "B"

    def test_critical_violation(self):
        violations = check_four_fifths_rule({"A": 1.0, "B": 0.3})
        assert violations[0]["severity"] == "CRITICAL"
class TestBiasAuditReport:
    def test_compliant_report(self):
        report = generate_bias_audit_report(
            sex_scores={"male": [0.8, 0.7, 0.9], "female": [0.7, 0.8, 0.85]},
            selection_threshold=0.5
        )
        assert report.compliant is True

    def test_non_compliant_report(self):
        report = generate_bias_audit_report(
            sex_scores={"male": [0.8, 0.9, 0.7], "female": [0.1, 0.2, 0.15]},
            selection_threshold=0.5
        )
        assert report.compliant is False
        assert len(report.violations) > 0

    def test_poc_mode_no_data(self):
        report = generate_bias_audit_report()
        assert "POC MODE" in report.recommendations[0]
class TestAuditRecord:
    def test_create_record(self):
        results = [
            MatchResult(resume_id="r1", name="Alice", final_score=0.8, confidence="HIGH",
                       rationale=MatchRationale("Good match", "STRONG_MATCH"), rank=1),
            MatchResult(resume_id="r2", name="Bob", final_score=0.3, confidence="LOW",
                       rationale=MatchRationale("Weak match", "WEAK_MATCH"), rank=2),
        ]
        record = create_audit_record(
            jd_id="test_jd", jd_text="Need Python developer",
            resumes={"r1": {"text": "Python dev"}, "r2": {"text": "HTML dev"}},
            results=results
        )
        assert record.num_resumes == 2
        assert record.hitl_required is True  # Bob has LOW confidence
        assert len(record.rankings) == 2
        assert record.jd_hash != ""

    def test_reproducibility_check(self):
        record = AuditRecord(
            audit_id="test", timestamp="now",
            model_id="gpt-4o-mini", config_hash="abc123"
        )
        result = verify_reproducibility(record, "abc123")
        assert result["reproducible"] is True

    def test_config_drift_detected(self):
        record = AuditRecord(
            audit_id="test", timestamp="now",
            model_id="gpt-4o-mini", config_hash="old_hash"
        )
        result = verify_reproducibility(record, "new_hash")
        assert result["reproducible"] is False
        assert "Config drift" in result["issues"][0]
# ============================================================
# MCP Server
# ============================================================

class TestMCPServer:
    def setup_method(self):
        self.server = MCPServer()

    def test_initialize(self):
        resp = self.server.handle({"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1})
        assert resp["result"]["serverInfo"]["name"] == "resume-matching-mcp"
        assert resp["result"]["protocolVersion"] == "2024-11-05"

    def test_list_tools(self):
        resp = self.server.handle({"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2})
        tools = resp["result"]["tools"]
        assert len(tools) >= 6  # at least our core tools
        names = [t["name"] for t in tools]
        assert "match_resumes" in names
        assert "run_bias_audit" in names
        assert "submit_feedback" in names

    def test_list_resources(self):
        resp = self.server.handle({"jsonrpc": "2.0", "method": "resources/list", "params": {}, "id": 3})
        resources = resp["result"]["resources"]
        assert len(resources) >= 3

    def test_read_config_resource(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "method": "resources/read",
            "params": {"uri": "matching://config"}, "id": 4
        })
        content = json.loads(resp["result"]["contents"][0]["text"])
        assert "weights" in content
        assert content["temperature"] == 0.0

    def test_call_bias_audit(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "run_bias_audit", "arguments": {}}, "id": 5
        })
        assert "content" in resp["result"]

    def test_call_unknown_tool(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}}, "id": 6
        })
        assert resp.get("error") is not None

    def test_unknown_method(self):
        resp = self.server.handle({"jsonrpc": "2.0", "method": "foo/bar", "params": {}, "id": 7})
        assert resp["error"]["code"] == -32601

    def test_ats_stub(self):
        resp = self.server.handle({
            "jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "update_candidate_status_in_ats",
                      "arguments": {"resume_id": "r1", "new_status": "SHORTLISTED"}}, "id": 8
        })
        assert "STUB" in resp["result"]["content"][0]["text"]

    def test_tool_schemas_valid(self):
        """Every tool must have a name, description, and inputSchema."""
        for name, tool in TOOL_REGISTRY.items():
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
# ============================================================
# Index Store
# ============================================================

class TestIndexStore:
    def test_corpus_hash_deterministic(self):
        docs = {"a": "hello", "b": "world"}
        assert compute_corpus_hash(docs) == compute_corpus_hash(docs)

    def test_corpus_hash_changes(self):
        docs1 = {"a": "hello"}
        docs2 = {"a": "hello", "b": "world"}
        assert compute_corpus_hash(docs1) != compute_corpus_hash(docs2)

    def test_build_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IndexStore(tmpdir)
            docs = {"r1": "Python developer with AWS", "r2": "Java developer with GCP"}
            store.build(docs)
            store.save()

            assert os.path.exists(os.path.join(tmpdir, "index_meta.json"))
            assert os.path.exists(os.path.join(tmpdir, "documents.json"))
            assert store.meta["num_documents"] == 2

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IndexStore(tmpdir)
            docs = {"r1": "Python developer", "r2": "Java developer"}
            store.build(docs)
            store.save()

            store2 = IndexStore(tmpdir)
            assert store2.load() is True
            assert len(store2.doc_ids) == 2
            assert store2.doc_ids == ["r1", "r2"]

    def test_is_valid_after_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = {"r1": "Python developer"}
            store = IndexStore(tmpdir)
            store.build(docs)
            store.save()
            assert store.is_valid(docs) is True

    def test_invalidation_on_corpus_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs1 = {"r1": "Python developer"}
            store = IndexStore(tmpdir)
            store.build(docs1)
            store.save()

            docs2 = {"r1": "Python developer", "r2": "New resume added"}
            assert store.is_valid(docs2) is False

    def test_invalidate_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = {"r1": "test"}
            store = IndexStore(tmpdir)
            store.build(docs)
            store.save()
            store.invalidate()
            assert not os.path.exists(store.meta_path)

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = IndexStore(tmpdir)
            store.build({"r1": "Python dev"})
            stats = store.get_stats()
            assert stats["num_documents"] == 1
            assert stats["embedding_model"] is not None
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
