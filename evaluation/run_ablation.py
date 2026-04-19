import json
import time
import asyncio
import copy
from pathlib import Path
from unittest.mock import patch

from app.pipeline.graph.workflow import cdss_app

BASE_DIR = Path(__file__).resolve().parent

def run_pipeline_with_mock(query):
    initial_state = {"query": query}
    # Run the graph synchronously using asyncio for the test
    loop = asyncio.get_event_loop()
    final_state = loop.run_until_complete(asyncio.to_thread(cdss_app.invoke, initial_state))
    return final_state

def score_case(case, final_state):
    answer = final_state.get("final_answer", "")
    trace = final_state.get("_trace", [])
    citations = final_state.get("citations", [])

    forbidden = case["expected"].get("forbidden_content", [])
    hallucinations = sum([1 for f in forbidden if f.lower() in answer.lower()])
    hallu_score = (hallucinations / max(1, len(forbidden))) * 100 if forbidden else 0

    expected_citations = case["expected"].get("expected_citations_contain", [])
    actual_sources = [c.source for c in citations]
    ci_hits = sum([1 for e in expected_citations if any(e.lower() in s.lower() for s in actual_sources)])
    cite_score = (ci_hits / max(1, len(expected_citations))) * 100 if expected_citations else 100

    return hallu_score, cite_score

def run_ablation():
    path = BASE_DIR / "golden_set.json"
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    # We will test 3 configs: Full Pipeline, No CRAG, No Guardrails
    configs = [
        {"name": "Full pipeline"},
        {"name": "No CRAG filter", "mock_target": "app.pipeline.graph.nodes.evaluate_chunks"},
        {"name": "No guardrails", "mock_target": "app.pipeline.generation.self_rag._enforce_gnomad_accuracy"}
    ]

    print("Starting Ablation Study...")
    
    summary = []
    summary.append("# Ablation Study Results")
    summary.append("| Configuration | Hallucination Rate (Avg) | Citation Accuracy (Avg) |")
    summary.append("|---------------|--------------------------|-------------------------|")

    for config in configs:
        print(f"\n--- Running Config: {config['name']} ---")
        total_hallu = 0
        total_cite = 0
        valid_cases = 0
        
        for case in cases[:5]: # Run a subset for speed during ablation
            print(f"  Testing {case['id']}...")
            try:
                if "mock_target" in config:
                    # Mocking different components to simulate ablation
                    if "evaluate_chunks" in config["mock_target"]:
                        # Mock CRAG to just pass all chunks
                        with patch(config["mock_target"]) as mock_eval:
                            mock_eval.side_effect = lambda q, chunks: {"correct": chunks, "ambiguous": [], "incorrect": []}
                            state = run_pipeline_with_mock(case["query"])
                    elif "enforce_" in config["mock_target"]:
                        # Mock the hallucination guardrails to do nothing (return original text)
                        with patch(config["mock_target"]) as mock_guard:
                            mock_guard.side_effect = lambda ans, chunks: ans
                            state = run_pipeline_with_mock(case["query"])
                else:
                    state = run_pipeline_with_mock(case["query"])

                h_score, c_score = score_case(case, state)
                total_hallu += h_score
                total_cite += c_score
                valid_cases += 1
            except Exception as e:
                print(f"  Failed: {e}")
        
        avg_h = total_hallu / max(1, valid_cases)
        avg_c = total_cite / max(1, valid_cases)
        
        summary.append(f"| {config['name']} | {avg_h:.1f}% | {avg_c:.1f}% |")

    out_path = results_dir / f"ablation_{int(time.time())}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    print(f"\nAblation complete. Report saved to {out_path}")

if __name__ == "__main__":
    run_ablation()
