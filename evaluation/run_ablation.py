import json
import time
import asyncio
import copy
from contextlib import ExitStack
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


# ── Guardrail targets for ablation ────────────────────────────────────────────
# Each tuple: (mock target path, pass-through side_effect)
# Different guardrails have different signatures, so side_effects are tailored.

GUARDRAIL_MOCKS = [
    # (answer, chunks) -> answer
    ("app.pipeline.generation.self_rag._enforce_gnomad_accuracy",
     lambda ans, chunks: ans),
    ("app.pipeline.generation.self_rag._enforce_no_fabricated_biology",
     lambda ans, chunks: ans),
    ("app.pipeline.generation.self_rag._enforce_clinvar_classification",
     lambda ans, chunks: ans),
    ("app.pipeline.generation.self_rag._enforce_no_fabricated_predictions",
     lambda ans, chunks: ans),
    ("app.pipeline.generation.self_rag._enforce_no_fabricated_acmg",
     lambda ans, chunks: ans),
    # (answer, query) -> answer
    ("app.pipeline.generation.self_rag._enforce_gene_screening_boundaries",
     lambda ans, query: ans),
    # (answer) -> answer
    ("app.pipeline.generation.self_rag._cleanup_orphaned_text",
     lambda ans: ans),
]


def run_ablation():
    path = BASE_DIR / "golden_set.json"
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 3 configs: Full Pipeline, No CRAG, No Guardrails (all 7 mocked)
    configs = [
        {"name": "Full pipeline"},
        {"name": "No CRAG filter", "mock_target": "app.pipeline.graph.nodes.evaluate_chunks"},
        {"name": "No guardrails", "mock_all_guardrails": True},
    ]

    print("Starting Ablation Study...")
    
    summary = []
    summary.append("# Ablation Study Results")
    summary.append(f"\n**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"**Cases:** {min(5, len(cases))} (subset for speed)\n")
    summary.append("| Configuration | Hallucination Rate (Avg) | Citation Accuracy (Avg) |")
    summary.append("|---------------|--------------------------|-------------------------|")

    # Store per-config averages for chart
    config_results = []

    for config in configs:
        print(f"\n--- Running Config: {config['name']} ---")
        total_hallu = 0
        total_cite = 0
        valid_cases = 0
        
        for case in cases[:5]: # Run a subset for speed during ablation
            print(f"  Testing {case['id']}...")
            try:
                if config.get("mock_all_guardrails"):
                    # Mock ALL guardrails using ExitStack
                    with ExitStack() as stack:
                        for target, side_effect in GUARDRAIL_MOCKS:
                            mock_obj = stack.enter_context(patch(target))
                            mock_obj.side_effect = side_effect
                        state = run_pipeline_with_mock(case["query"])

                elif "mock_target" in config:
                    if "evaluate_chunks" in config["mock_target"]:
                        with patch(config["mock_target"]) as mock_eval:
                            mock_eval.side_effect = lambda q, chunks: {"correct": chunks, "ambiguous": [], "incorrect": []}
                            state = run_pipeline_with_mock(case["query"])
                else:
                    state = run_pipeline_with_mock(case["query"])

                h_score, c_score = score_case(case, state)
                total_hallu += h_score
                total_cite += c_score
                valid_cases += 1
                icon = "✅" if h_score == 0 else "❌"
                print(f"    {icon} Hallu={h_score:.0f}% Cite={c_score:.0f}%")
            except Exception as e:
                print(f"    Failed: {e}")
        
        avg_h = total_hallu / max(1, valid_cases)
        avg_c = total_cite / max(1, valid_cases)
        
        summary.append(f"| {config['name']} | {avg_h:.1f}% | {avg_c:.1f}% |")
        config_results.append({"name": config["name"], "hallu": avg_h, "cite": avg_c})

    summary.append("")
    summary.append("## Methodology")
    summary.append("")
    summary.append("- **Full pipeline**: All components active (CRAG + 7 guardrails)")
    summary.append("- **No CRAG filter**: CRAG evaluator bypassed — all retrieved chunks pass through")
    summary.append("- **No guardrails**: All 7 post-generation guardrails disabled:")
    summary.append("  - `_enforce_gnomad_accuracy`")
    summary.append("  - `_enforce_no_fabricated_biology`")
    summary.append("  - `_enforce_clinvar_classification`")
    summary.append("  - `_enforce_no_fabricated_predictions`")
    summary.append("  - `_enforce_no_fabricated_acmg`")
    summary.append("  - `_enforce_gene_screening_boundaries`")
    summary.append("  - `_cleanup_orphaned_text`")

    ts = int(time.time())
    out_path = results_dir / f"ablation_{ts}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))

    # ── Charts ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        charts_dir = results_dir / f"ablation_{ts}_charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        config_names = [r["name"] for r in config_results]
        hallu_vals   = [r["hallu"] for r in config_results]
        cite_vals    = [r["cite"] for r in config_results]

        # — Ablation Grouped Bar Chart ————————————————————————————————————————
        x = np.arange(len(config_names))
        bar_width = 0.3

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - bar_width/2, hallu_vals, bar_width,
                        label="Hallucination Rate (%)", color="#e74c3c", alpha=0.85)
        bars2 = ax.bar(x + bar_width/2, cite_vals, bar_width,
                        label="Citation Accuracy (%)", color="#2ecc71", alpha=0.85)

        # Value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

        ax.set_xlabel("Pipeline Configuration", fontsize=12)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title("Ablation Study: Impact of Component Removal", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, fontsize=11)
        ax.set_ylim(0, 115)
        ax.legend(loc="upper right", fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(charts_dir / "ablation_comparison.png", dpi=150)
        plt.close(fig)

        # — Ablation Radar Chart ——————————————————————————————————————————————
        labels = config_names
        # Normalize: for radar, we want "goodness" — so invert hallucination (100 - hallu)
        hallu_free = [100.0 - h for h in hallu_vals]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for vals, name, color in [
            (hallu_free, "Hallucination-Free (%)", "#e74c3c"),
            (cite_vals,  "Citation Accuracy (%)", "#2ecc71"),
        ]:
            vals_plot = vals + [vals[0]]
            angs_plot = angles + [angles[0]]
            ax.fill(angs_plot, vals_plot, alpha=0.15, color=color)
            ax.plot(angs_plot, vals_plot, color=color, linewidth=2, marker="o",
                    markersize=7, label=name)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_title("Ablation: Config Quality Comparison", fontsize=13,
                      fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.tight_layout()
        fig.savefig(charts_dir / "ablation_radar.png", dpi=150)
        plt.close(fig)

        print(f"  [Charts] Saved ablation charts to {charts_dir}/")

    except ImportError:
        print("  [Info] Install matplotlib for ablation charts: pip install matplotlib")

    print(f"\nAblation complete. Report saved to {out_path}")

if __name__ == "__main__":
    run_ablation()
