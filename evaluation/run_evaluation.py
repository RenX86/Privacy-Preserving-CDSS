"""
Evaluation runner for the Privacy-Preserving CDSS golden test set.

Methodology:
  - Hallucination check runs against DRAFT (pre-guardrail) answer to expose what the
    LLM actually fabricated, not what the guardrail already cleaned up.
  - Guardrail effectiveness = how many forbidden terms were in draft but not in final.
  - Citation check runs against FINAL answer (post-enforcer).
  - Screening keyword check distinguishes "expected" (must be present) from "forbidden".
  - Pass requires ALL of: zero hallucination in draft AND 100% citation accuracy.

Usage (API server must be running):
    python evaluation/run_evaluation.py

Output:
    evaluation/results/run_<timestamp>.md          — markdown summary table
    evaluation/results/run_<timestamp>_charts/     — PNG charts for thesis
"""
import json
import time
import requests
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False
    print("[Warning] matplotlib not installed — charts will be skipped.")

BASE_DIR  = Path(__file__).resolve().parent
API_URL   = "http://127.0.0.1:5656/query/detailed"
TIMEOUT_S = 180


# ── helpers ───────────────────────────────────────────────────────────────────

def call_api(query: str) -> dict | None:
    try:
        res = requests.post(API_URL, json={"query": query}, timeout=TIMEOUT_S)
        res.raise_for_status()
        return res.json()
    except Exception as exc:
        print(f"    [API ERROR] {exc}")
        return None


def score_case(case: dict, data: dict) -> dict:
    """
    Score a single test case against API response.

    Hallucination check:
        Runs against `draft_answer` — the raw LLM output BEFORE guardrails strip anything.
        This is the academically honest measure: did the LLM hallucinate?

    Guardrail effectiveness:
        Counts how many forbidden terms appeared in draft but were stripped from final.
        This shows the guardrail layer is doing real work.

    Citation check:
        Runs against final `citations` list (post-citation-enforcer).
        An expected citation MUST appear as a source.

    Screening keyword check:
        Runs against the final `answer`. These are positive keywords that MUST appear.
    """
    # Use draft (pre-guardrail) for hallucination testing
    draft  = data.get("draft_answer", "")
    final  = data.get("answer", "")
    trace  = data.get("trace", [])
    citations = data.get("citations", [])

    # ── 1. Hallucination in DRAFT (what the LLM actually produced) ─────────────
    forbidden = case["expected"].get("forbidden_content", [])
    draft_hallu_hits  = [f for f in forbidden if f.lower() in draft.lower()]
    final_hallu_hits  = [f for f in forbidden if f.lower() in final.lower()]
    # Terms that were in draft but stripped by guardrails
    guardrail_caught  = [f for f in draft_hallu_hits if f not in final_hallu_hits]

    hallu_rate = (len(draft_hallu_hits) / max(1, len(forbidden))) * 100 if forbidden else 0.0

    # ── 2. Guardrail effectiveness ─────────────────────────────────────────────
    guardrail_eff = (len(guardrail_caught) / max(1, len(draft_hallu_hits))) * 100 if draft_hallu_hits else None

    # ── 3. Citation accuracy (from final citations list) ──────────────────────
    expected_src = case["expected"].get("expected_citations_contain", [])
    actual_src   = [c["source"] for c in citations]
    cite_hits    = [e for e in expected_src if any(e.lower() in s.lower() for s in actual_src)]
    # Only score if citations were actually expected — no free 100% for empty lists
    cite_acc = (len(cite_hits) / len(expected_src)) * 100 if expected_src else None

    # Check for UNEXPECTED citations (citation hallucination)
    known_sources = {"Clinvar", "gnomAD", "ClinGen", "Detection, Prevention, and Risk Reduction"}
    spurious_citations = [s for s in actual_src if not any(k.lower() in s.lower() for k in known_sources)]

    # ── 4. Screening keyword accuracy (positive keywords in final answer) ──────
    keywords = case["expected"].get("expected_screening_keywords", [])
    kw_hits  = [k for k in keywords if k.lower() in final.lower()]
    kw_acc   = (len(kw_hits) / len(keywords)) * 100 if keywords else None

    # ── 5. Node timing ─────────────────────────────────────────────────────────
    timings = {n["node"]: n["duration_ms"] for n in trace}

    # ── 6. Retrieval recall from Evaluator trace ───────────────────────────────
    evaluator = next((n for n in trace if n["node"] == "Evaluator"), None)
    verified_count = evaluator["data"].get("verified_chunks_count", 0) if evaluator else 0

    # ── Pass criteria ──────────────────────────────────────────────────────────
    # STRICT: LLM must not hallucinate in draft AND all required citations present
    # (Guardrails are a safety net but not a free pass on hallucination scoring)
    cite_ok  = (cite_acc == 100.0) if cite_acc is not None else True
    hallu_ok = (hallu_rate == 0.0)
    passed   = hallu_ok and cite_ok

    return {
        "id":               case["id"],
        "gene":             case["expected"].get("gene"),
        # Hallucination in raw LLM draft
        "hallu_rate":       hallu_rate,
        "draft_hallu_hits": draft_hallu_hits,
        "final_hallu_hits": final_hallu_hits,
        "guardrail_caught": guardrail_caught,
        "guardrail_eff":    guardrail_eff,
        # Citations
        "cite_acc":         cite_acc,
        "cite_hits":        cite_hits,
        "expected_src":     expected_src,
        "spurious_citations": spurious_citations,
        # Retrieval
        "verified":         verified_count,
        # Screening keywords
        "kw_acc":           kw_acc,
        "kw_hits":          kw_hits,
        # Timing
        "timings":          timings,
        "total_ms":         data.get("total_duration_ms", 0),
        # Overall
        "passed":           passed,
    }


# ── chart generation ──────────────────────────────────────────────────────────

def generate_charts(scores: list[dict], charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    ids = [s["id"] for s in scores]

    # — 1. Hallucination Rate (DRAFT, before guardrails) ——————————————————————
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#e74c3c" if s["hallu_rate"] > 0 else "#2ecc71" for s in scores]
    ax.bar(ids, [s["hallu_rate"] for s in scores], color=colors)
    ax.set_title("LLM Hallucination Rate in Draft (before guardrails) (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Test Case ID")
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_ylim(0, 110)
    green_p = mpatches.Patch(color="#2ecc71", label="Pass (0%)")
    red_p   = mpatches.Patch(color="#e74c3c", label="Fail (>0%)")
    ax.legend(handles=[green_p, red_p])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(charts_dir / "hallucination_rate.png", dpi=150)
    plt.close(fig)

    # — 2. Guardrail Effectiveness (% of draft hallucinations caught) —————————
    guard_scores = [s for s in scores if s["guardrail_eff"] is not None]
    if guard_scores:
        fig, ax = plt.subplots(figsize=(14, 5))
        g_ids  = [s["id"] for s in guard_scores]
        g_vals = [s["guardrail_eff"] for s in guard_scores]
        ax.bar(g_ids, g_vals, color="#8e44ad")
        ax.set_title("Guardrail Effectiveness (% of LLM hallucinations caught & stripped) (%)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Test Case ID")
        ax.set_ylabel("Guardrail Effectiveness (%)")
        ax.set_ylim(0, 110)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(charts_dir / "guardrail_effectiveness.png", dpi=150)
        plt.close(fig)

    # — 3. Citation Accuracy ——————————————————————————————————————————————————
    cite_scores = [s for s in scores if s["cite_acc"] is not None]
    if cite_scores:
        fig, ax = plt.subplots(figsize=(14, 5))
        c_ids  = [s["id"] for s in cite_scores]
        colors = ["#2ecc71" if s["cite_acc"] == 100 else "#e67e22" for s in cite_scores]
        ax.bar(c_ids, [s["cite_acc"] for s in cite_scores], color=colors)
        ax.set_title("Citation Accuracy per Test Case (% required sources present)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Test Case ID")
        ax.set_ylabel("Citation Accuracy (%)")
        ax.set_ylim(0, 110)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(charts_dir / "citation_accuracy.png", dpi=150)
        plt.close(fig)

    # — 4. Retrieval Recall ——————————————————————————————————————————————————
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(ids, [s["verified"] for s in scores], color="#3498db")
    ax.set_title("Verified Chunks Retrieved per Test Case (post-CRAG)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Test Case ID")
    ax.set_ylabel("Chunk Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(charts_dir / "retrieval_recall.png", dpi=150)
    plt.close(fig)

    # — 5. Node Timing Breakdown ——————————————————————————————————————————————
    node_order  = ["Decomposer", "DB_Retriever", "PDF_Retriever", "Evaluator", "Generator", "Enforcer"]
    node_colors = ["#9b59b6", "#3498db", "#1abc9c", "#f39c12", "#e74c3c", "#95a5a6"]
    bottoms = [0.0] * len(scores)
    fig, ax = plt.subplots(figsize=(14, 6))
    for node, color in zip(node_order, node_colors):
        vals = [s["timings"].get(node, 0) / 1000 for s in scores]
        ax.bar(ids, vals, bottom=bottoms, label=node, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title("Node Timing Breakdown per Test Case (seconds)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Test Case ID")
    ax.set_ylabel("Duration (s)")
    ax.legend(loc="upper right")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(charts_dir / "node_timing.png", dpi=150)
    plt.close(fig)

    # — 6. Pass / Fail pie ————————————————————————————————————————————————————
    passed = sum(1 for s in scores if s["passed"])
    failed = len(scores) - passed
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [passed, failed],
        labels=[f"Pass ({passed})", f"Fail ({failed})"],
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.0f%%", startangle=90
    )
    ax.set_title("Overall Pass / Fail", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(charts_dir / "pass_fail_summary.png", dpi=150)
    plt.close(fig)

    print(f"  [Charts] Saved 6 charts to {charts_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate():
    path = BASE_DIR / "golden_set.json"
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp  = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_md     = results_dir / f"run_{timestamp}.md"
    raw_out    = results_dir / f"raw_outputs_{timestamp}.json"   # consumed by run_ragas.py
    charts_dir = results_dir / f"run_{timestamp}_charts"

    print(f"\n{'='*70}")
    print(f"  CDSS Evaluation — {len(cases)} cases")
    print(f"  Hallucination check: DRAFT (raw LLM, pre-guardrail)")
    print(f"  Citation check:      FINAL (post-guardrail)")
    print(f"{'='*70}\n")

    scores = []
    _raw_api_responses = []    # parallel list — same index as cases
    for case in cases:
        desc = case.get("description", "")
        print(f"[{case['id']}] {desc[:70]}")
        data = call_api(case["query"])
        _raw_api_responses.append(data)
        if data is None:
            scores.append({
                "id": case["id"], "gene": case["expected"].get("gene"),
                "hallu_rate": -1, "draft_hallu_hits": [], "final_hallu_hits": [],
                "guardrail_caught": [], "guardrail_eff": None,
                "cite_acc": -1, "cite_hits": [], "expected_src": [],
                "spurious_citations": [],
                "verified": 0, "kw_acc": None, "kw_hits": [],
                "timings": {}, "total_ms": 0, "passed": False,
            })
            continue

        s = score_case(case, data)
        scores.append(s)

        icon = "✅" if s["passed"] else "❌"
        cite_str = f"{s['cite_acc']:.0f}%" if s["cite_acc"] is not None else "N/A"
        kw_str   = f"{s['kw_acc']:.0f}%"   if s["kw_acc"]   is not None else "N/A"
        print(f"  {icon}  Hallu(draft)={s['hallu_rate']:.0f}%  CitationAcc={cite_str}  Chunks={s['verified']}  KW={kw_str}  [{s['total_ms']}ms]")

        if s["draft_hallu_hits"]:
            print(f"     ⚠ LLM fabricated (draft): {s['draft_hallu_hits']}")
        if s["guardrail_caught"]:
            print(f"     🛡 Guardrails caught:       {s['guardrail_caught']}")
        if s["final_hallu_hits"]:
            print(f"     ❌ Still in final answer:   {s['final_hallu_hits']}")
        if s["spurious_citations"]:
            print(f"     ⚠ Spurious citations:       {s['spurious_citations']}")

    # — summary stats ──────────────────────────────────────────────────────────
    valid = [s for s in scores if s["hallu_rate"] >= 0]
    n     = max(1, len(valid))

    avg_hallu = sum(s["hallu_rate"] for s in valid) / n
    cite_vals = [s["cite_acc"] for s in valid if s["cite_acc"] is not None]
    avg_cite  = sum(cite_vals) / max(1, len(cite_vals))
    avg_v     = sum(s["verified"] for s in valid) / n
    kw_vals   = [s["kw_acc"] for s in valid if s["kw_acc"] is not None]
    avg_kw    = sum(kw_vals) / max(1, len(kw_vals))

    all_guardrail = [s for s in valid if s["guardrail_eff"] is not None]
    avg_geff  = sum(s["guardrail_eff"] for s in all_guardrail) / max(1, len(all_guardrail))

    # — Markdown table ─────────────────────────────────────────────────────────
    lines = [
        f"# CDSS Evaluation Results — {timestamp}", "",
        f"**Total cases:** {len(scores)}  |  **Passed:** {sum(s['passed'] for s in scores)}  |  **Failed:** {sum(not s['passed'] for s in scores)}", "",
        "> **Methodology:** Hallucination rate is measured on the RAW LLM draft (before guardrails).",
        "> Citation accuracy checks that all required sources appear in the final response.",
        "> Guardrail Effectiveness shows what % of LLM fabrications were caught and stripped.", "",
        "| ID | Gene | Hallucination (draft) | Guardrail Eff. | Citation Acc. | Chunks | KW Acc. | Pass |",
        "|----|------|----------------------|----------------|---------------|--------|---------|------|",
    ]
    for s in scores:
        h  = f"{s['hallu_rate']:.0f}%"      if s["hallu_rate"] >= 0     else "ERR"
        ge = f"{s['guardrail_eff']:.0f}%"   if s["guardrail_eff"] is not None else "—"
        c  = f"{s['cite_acc']:.0f}%"        if s["cite_acc"] is not None      else "—"
        kw = f"{s['kw_acc']:.0f}%"          if s["kw_acc"] is not None        else "—"
        ok = "✅" if s["passed"] else "❌"
        # annotate draft hits
        note = ""
        if s.get("draft_hallu_hits"):
            note = f" ⚠`{'`,`'.join(s['draft_hallu_hits'][:3])}`"
        lines.append(f"| {s['id']} | {s['gene']} | {h}{note} | {ge} | {c} | {s['verified']} | {kw} | {ok} |")

    lines += [
        "",
        f"| **AVG** | — | **{avg_hallu:.1f}%** | **{avg_geff:.1f}%** | **{avg_cite:.1f}%** | **{avg_v:.1f}** | **{avg_kw:.1f}%** | — |",
        "",
        "## Guardrail Summary",
        "",
        f"Cases where LLM hallucinated in draft: **{len(all_guardrail)}**",
        f"Average guardrail effectiveness: **{avg_geff:.1f}%**",
        "",
        "## Citation Hallucination",
        "",
    ]
    spurious = [(s["id"], s["spurious_citations"]) for s in valid if s["spurious_citations"]]
    if spurious:
        for sid, cits in spurious:
            lines.append(f"- **{sid}**: Spurious citations: {cits}")
    else:
        lines.append("- None detected — all citations matched known sources.")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Save raw outputs for Ragas (Phase 2) ─────────────────────────────────
    # Each entry contains everything run_ragas.py needs: query, contexts,
    # draft_answer, final_answer. Run: python evaluation/run_ragas.py <this file>
    raw_records = []
    for case, data_resp in zip(cases, _raw_api_responses):
        if data_resp is None:
            continue
        evaluator_node = next((n for n in data_resp.get("trace", []) if n["node"] == "Evaluator"), None)
        chunk_texts = []
        if evaluator_node:
            chunk_texts = [
                c["text"] for c in evaluator_node["data"].get("chunk_texts", [])
            ]
        raw_records.append({
            "id":           case["id"],
            "description":  case.get("description", ""),
            "query":        case["query"],
            "draft_answer": data_resp.get("draft_answer", ""),
            "final_answer": data_resp.get("answer", ""),
            "citations":    data_resp.get("citations", []),
            "contexts":     chunk_texts,         # verified chunk texts — used as Ragas context
        })
    with open(raw_out, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, indent=2, ensure_ascii=False)

    print(f"\n[Results]    → {out_md}")
    print(f"[Raw outputs] → {raw_out}  (feed to run_ragas.py for Ragas scoring)")
    print(f"\nSummary: Hallu={avg_hallu:.1f}%  GuardrailEff={avg_geff:.1f}%  CitationAcc={avg_cite:.1f}%  KW={avg_kw:.1f}%\n")

    if CHARTS_AVAILABLE:
        valid_scores = [s for s in scores if s["hallu_rate"] >= 0]
        if valid_scores:
            generate_charts(valid_scores, charts_dir)
    else:
        print("[Info] Install matplotlib to generate charts: pip install matplotlib")

    print("Done.\n")


if __name__ == "__main__":
    evaluate()
