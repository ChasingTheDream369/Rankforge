"""
In-memory job state for test-suite and ablation runs.
Simple module-level dict — suitable for demo; replace with DB/cache for prod.
"""
import importlib.util
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from django.conf import settings

# ── Shared state dicts ────────────────────────────────────────────────────────
_test_state  = {"status": "idle",  "results": None, "error": None}
_ablat_state = {
    "status": "idle",
    "results": None,
    "error": None,
    "progress": None,  # {"step": int, "total": int, "name": str} while running
    "started_at": None,  # time.time() when run started
}
_lock = threading.Lock()


def _set_ablation_progress(step: int, total: int, name: str) -> None:
    with _lock:
        if _ablat_state["status"] == "running":
            _ablat_state["progress"] = {"step": step, "total": total, "name": name}


# ── Test runner ───────────────────────────────────────────────────────────────

def _load_test_descriptions(test_file: str) -> dict:
    """Import the test module and extract docstrings keyed by 'ClassName::test_name'."""
    descriptions = {}
    try:
        spec = importlib.util.spec_from_file_location("_test_module", test_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import inspect
        for cls_name, cls_obj in inspect.getmembers(mod, inspect.isclass):
            for fn_name, fn_obj in inspect.getmembers(cls_obj, inspect.isfunction):
                if fn_name.startswith("test_"):
                    doc = inspect.getdoc(fn_obj) or ""
                    descriptions[f"{cls_name}::{fn_name}"] = doc
    except Exception:
        pass
    return descriptions


def _parse_pytest_output(stdout: str, descriptions: dict = None) -> dict:
    """Parse verbose pytest stdout into structured result dict."""
    descriptions = descriptions or {}
    tests = []
    current_class = None

    for line in stdout.splitlines():
        # Class header: "tests/test_all.py::TestFoo::test_bar PASSED"
        m = re.match(r'\S+::(\w+)::(\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)', line)
        if m:
            cls, name, status = m.group(1), m.group(2), m.group(3)
            desc = descriptions.get(f"{cls}::{name}", "")
            tests.append({"class": cls, "name": name, "status": status, "detail": "", "description": desc})
            current_class = cls

    # Attach failure details
    detail_lines = []
    in_fail = False
    for line in stdout.splitlines():
        if line.startswith("FAILED ") or "_ FAILED _" in line:
            in_fail = True
            detail_lines = [line]
        elif in_fail and (line.startswith("=") or line.startswith("_")):
            # attach to last matching test
            detail_text = "\n".join(detail_lines).strip()
            for t in reversed(tests):
                if t["status"] == "FAILED" and not t["detail"]:
                    t["detail"] = detail_text
                    break
            in_fail = False
            detail_lines = []
        elif in_fail:
            detail_lines.append(line)

    # Summary counts
    passed  = sum(1 for t in tests if t["status"] == "PASSED")
    failed  = sum(1 for t in tests if t["status"] in ("FAILED", "ERROR"))
    skipped = sum(1 for t in tests if t["status"] == "SKIPPED")

    # Group by class
    by_class = {}
    for t in tests:
        by_class.setdefault(t["class"], []).append(t)

    return {
        "tests":    tests,
        "by_class": by_class,
        "summary":  {"passed": passed, "failed": failed, "skipped": skipped, "total": len(tests)},
        "stdout":   stdout[-4000:],  # last 4k chars for raw log view
    }


def run_tests_bg():
    """Spawn background thread to run pytest and store results."""
    with _lock:
        if _test_state["status"] == "running":
            return
        _test_state.update({"status": "running", "results": None, "error": None})

    def _worker():
        try:
            python = sys.executable
            test_file = str(Path(settings.BASE_DIR) / "tests" / "test_all.py")
            descriptions = _load_test_descriptions(test_file)
            result = subprocess.run(
                [python, "-m", "pytest", test_file, "-v", "--tb=short", "--no-header"],
                capture_output=True, text=True,
                cwd=str(settings.BASE_DIR),
                timeout=120,
            )
            parsed = _parse_pytest_output(result.stdout + result.stderr, descriptions)
            with _lock:
                _test_state["status"]  = "done"
                _test_state["results"] = parsed
        except Exception as exc:
            with _lock:
                _test_state["status"] = "error"
                _test_state["error"]  = str(exc)

    threading.Thread(target=_worker, daemon=True).start()


def get_test_state() -> dict:
    with _lock:
        return dict(_test_state)


# ── Ablation runner ───────────────────────────────────────────────────────────

def run_ablation_bg():
    """Spawn background thread to run ablation study."""
    with _lock:
        if _ablat_state["status"] == "running":
            return
        _ablat_state.update(
            {
                "status": "running",
                "results": None,
                "error": None,
                "progress": None,
                "started_at": time.time(),
            }
        )

    def _worker():
        try:
            # Ensure src is importable
            root = str(settings.BASE_DIR)
            if root not in sys.path:
                sys.path.insert(0, root)

            # Import ablation functions (they live in resume_matcher root now)
            import importlib.util, os
            abl_path = Path(settings.BASE_DIR) / "ablation.py"
            if not abl_path.exists():
                # fall back to engine directory
                abl_path = Path(settings.BASE_DIR).parent / "resume-matcher-v2-clean 2" / "ablation.py"
            spec = importlib.util.spec_from_file_location("ablation", str(abl_path))
            abl  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(abl)

            jd_text, resumes, labels, jd_id = abl.load_data()

            if not resumes:
                raise RuntimeError("No resumes found in sample data.")

            approaches = [
                ("TF-IDF only",        abl.run_tfidf_only),
                ("TF-IDF + BM25",      abl.run_tfidf_bm25),
                ("BM25 + Dense + RRF", abl.run_hybrid_rrf),
                ("+ Cross-encoder",    abl.run_hybrid_crossencoder),
                ("Hybrid + Agentic (Ontology Grounding)", abl.run_full_system),
            ]

            rows = []
            n_ap = len(approaches)
            for i, (name, fn) in enumerate(approaches, start=1):
                _set_ablation_progress(i, n_ap, name)
                try:
                    ranked_ids, scores = fn(jd_text, resumes)
                    metrics = abl.evaluate_ranking(ranked_ids, labels) if labels else {}
                    rows.append({"approach": name, "metrics": metrics,
                                 "top5": ranked_ids[:5], "scores": scores, "error": None})
                except Exception as e:
                    rows.append({"approach": name, "metrics": {}, "top5": [],
                                 "scores": {}, "error": str(e)})

            with _lock:
                _ablat_state["status"] = "done"
                _ablat_state["results"] = {"rows": rows, "jd_id": jd_id, "has_labels": bool(labels)}
                _ablat_state["progress"] = None
                _ablat_state["started_at"] = None
        except Exception as exc:
            with _lock:
                _ablat_state["status"] = "error"
                _ablat_state["error"]  = str(exc)
                _ablat_state["progress"] = None
                _ablat_state["started_at"] = None

    threading.Thread(target=_worker, daemon=True).start()


def get_ablation_state() -> dict:
    with _lock:
        out = dict(_ablat_state)
    if out.get("status") == "running" and out.get("started_at"):
        out["elapsed_sec"] = int(time.time() - out["started_at"])
    return out
