"""
Phase 2 Ragas evaluation — runs on saved raw_outputs_*.json from run_evaluation.py.

Uses local Ollama via its OpenAI-compatible API (/v1) as judge.
No external API keys. No patient data leaves the machine.

Usage:
    python evaluation/run_ragas.py
    python evaluation/run_ragas.py evaluation/results/raw_outputs_<timestamp>.json

Dependencies (install once):
    pip install ragas openai datasets langchain-huggingface
"""

import importlib
import inspect
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OLLAMA_BASE_URL = "http://localhost:11434"          # matches LOCAL_LLM_URL in .env
OLLAMA_MODEL    = "ministral-3:14b"                 # matches LOCAL_LLM_MODEL in .env
EMBED_MODEL     = "BAAI/bge-base-en-v1.5"           # matches EMBEDDING_MODEL in .env


# ── Helpers ─────────────────────────────────────────────────────────────────

def _import_attr(*candidates: tuple):
    """Try (module_path, attr) pairs in order; return first found or None."""
    for mod_path, attr in candidates:
        try:
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, attr, None)
            if obj is not None:
                return obj
        except (ImportError, AttributeError):
            continue
    return None


def _make_instance(obj, **kwargs):
    """Instantiate obj if it's a class, otherwise return as-is."""
    if inspect.isclass(obj):
        return obj(**kwargs)
    return obj


# ── I/O ─────────────────────────────────────────────────────────────────────

def get_latest_raw_outputs() -> Path:
    results_dir = BASE_DIR / "results"
    files = sorted(results_dir.glob("raw_outputs_*.json"), reverse=True)
    if not files:
        raise FileNotFoundError(
            "No raw_outputs_*.json in evaluation/results/. "
            "Run python evaluation/run_evaluation.py first."
        )
    return files[0]


def load_raw_outputs(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    valid   = [r for r in records if r.get("contexts")]
    skipped = len(records) - len(valid)
    if skipped:
        print(f"  [Info] Skipped {skipped} record(s) with empty contexts "
              "(safe-failure / API error cases).")
    return valid


# ── Core ─────────────────────────────────────────────────────────────────────

def run_ragas(records: list[dict], output_path: Path) -> None:

    # ── 1. Core imports ───────────────────────────────────────────────────────
    try:
        from ragas import evaluate
        from datasets import Dataset
    except ImportError as e:
        print(f"\n[Error] {e}\nRun: pip install ragas datasets\n")
        return

    # ── 2. LLM — Ollama via OpenAI-compatible /v1 endpoint ───────────────────
    # Ragas deprecated LangchainLLMWrapper in favour of llm_factory + OpenAI client.
    # Ollama natively supports the OpenAI REST API at /v1.
    try:
        from openai import OpenAI
        from ragas.llms import llm_factory
    except ImportError as e:
        print(f"\n[Error] {e}\nRun: pip install openai ragas\n")
        return

    print(f"\n  Judge LLM:  {OLLAMA_MODEL}  (Ollama OpenAI-compatible endpoint)")
    oai_client = OpenAI(
        api_key="ollama",                              # Ollama ignores the key value
        base_url=f"{OLLAMA_BASE_URL}/v1",
    )
    try:
        llm = llm_factory(OLLAMA_MODEL, client=oai_client)
    except TypeError:
        # Older llm_factory signature may not accept client kwarg — fallback
        from ragas.llms import LangchainLLMWrapper
        from langchain_ollama import ChatOllama
        llm = LangchainLLMWrapper(
            ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        )
        print("  [Warn] llm_factory kwarg unsupported — using LangchainLLMWrapper fallback.")

    # ── 3. Embeddings ─────────────────────────────────────────────────────────
    # ragas-native HuggingFaceEmbeddings lacks embed_query — always use the
    # LangChain wrapper which provides the full embedding interface Ragas needs.
    print(f"  Embeddings: {EMBED_MODEL}  (local HuggingFace / sentence-transformers)")
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        LCEmbed = _import_attr(
            ("langchain_huggingface",     "HuggingFaceEmbeddings"),
            ("langchain_community.embeddings", "HuggingFaceEmbeddings"),
        )
        if LCEmbed is None:
            print("\n[Error] No HuggingFace embedding provider found.\n"
                  "Run: pip install langchain-huggingface\n")
            return
        ragas_embeddings = LangchainEmbeddingsWrapper(LCEmbed(model_name=EMBED_MODEL))
    except Exception as e:
        print(f"\n[Error] Embeddings setup failed: {e}\n")
        return

    # ── 4. Locate metric classes ──────────────────────────────────────────────
    # Ragas 0.2+: capital-letter classes instantiated with constructor args.
    # ResponseRelevancy is the current name; AnswerRelevancy is an older alias.

    Faithfulness_cls = _import_attr(
        ("ragas.metrics",             "Faithfulness"),     # real class — works
        ("ragas.metrics.collections", "Faithfulness"),     # proxy — may fail isinstance check
    )
    AnswerRel_cls = _import_attr(
        ("ragas.metrics",             "ResponseRelevancy"),   # current name
        ("ragas.metrics",             "AnswerRelevancy"),      # older alias
        ("ragas.metrics.collections", "ResponseRelevancy"),
        ("ragas.metrics.collections", "AnswerRelevancy"),
        ("ragas.metrics",             "answer_relevancy"),     # deprecated singleton
    )
    CtxPrec_cls = _import_attr(
        ("ragas.metrics",             "LLMContextPrecisionWithoutReference"),
        ("ragas.metrics",             "NonReferenceContextPrecision"),
        ("ragas.metrics.collections", "LLMContextPrecisionWithoutReference"),
        ("ragas.metrics.collections", "NonReferenceContextPrecision"),
    )

    if Faithfulness_cls is None or AnswerRel_cls is None:
        print("\n[Error] Could not locate Faithfulness / ResponseRelevancy classes.\n"
              "Try: pip install --upgrade ragas\n")
        return

    # ── 5. Instantiate metrics with LLM/embeddings in constructor ─────────────
    faith_obj = _make_instance(Faithfulness_cls, llm=llm)
    ar_obj    = _make_instance(AnswerRel_cls,    llm=llm, embeddings=ragas_embeddings)

    metrics = [faith_obj, ar_obj]
    print(f"  Metrics:    {type(faith_obj).__name__}, {type(ar_obj).__name__}", end="")

    if CtxPrec_cls is not None:
        ctx_obj = _make_instance(CtxPrec_cls, llm=llm)
        metrics.append(ctx_obj)
        print(f", {type(ctx_obj).__name__}", end="")
    else:
        print("\n  [Warn] Context precision metric not found — skipping.", end="")
        ctx_obj = None
    print()

    # Context budget for the local Ragas judge (ministral-3:14b).
    # 350 chars was cutting off key clinical facts (ages, procedures) mid-sentence
    # in NCCN chunks. 600 chars covers a complete NCCN evidence chunk.
    # 4 chunks × 600 chars = 2400 chars ≈ 600 context tokens — fits comfortably
    # within the judge's 2048-token output limit alongside the 1500-char answer cap.
    MAX_CHUNK_CHARS = 600    # was 350 — now covers full NCCN protocol chunks
    MAX_CHUNKS      = 4      # was 5 — slightly reduced to keep total prompt size sane

    def _is_bibliography_chunk(text: str) -> bool:
        """
        Detect NCCN PDF reference-section chunks that contain no clinical protocol content.
        These pass CRAG due to header metadata keywords but are useless for Ragas faithfulness.
        Heuristics:
          - Contains 'Available at: http' (bibliography URL pattern)
          - Starts with a numbered citation like '327. Author...' after the Header_2 block
        """
        if "Available at: http" in text:
            return True
        # Numbered citation at start of meaningful content (after header)
        import re
        # Strip header block if present
        body = re.sub(r"^\[Header_\d+:.*?\]\n##.*?\n", "", text, flags=re.DOTALL).strip()
        if re.match(r"^\d{2,3}\.\s+[A-Z][a-z]+", body):   # e.g. "327. Sherman ME..."
            return True
        return False

    def _trim_contexts(ctxs: list[str]) -> list[str]:
        # Remove bibliography chunks first, then truncate remaining
        clinical = [c for c in ctxs if not _is_bibliography_chunk(c)]
        # Fall back to all chunks if filter removes everything
        pool = clinical if clinical else ctxs
        return [c[:MAX_CHUNK_CHARS] for c in pool[:MAX_CHUNKS]]

    dataset = Dataset.from_dict({
        "user_input":         [r["query"]                      for r in records],
        # Cap answer at 1500 chars: shorter answers → fewer Faithfulness statements
        # → LLM JSON fits within Ollama's token limit → no N/A scores
        "response":           [r["final_answer"][:1500]        for r in records],
        "retrieved_contexts": [_trim_contexts(r["contexts"])  for r in records],
    })

    # max_workers=1 serialises calls — avoids hammering local Ollama with parallel requests
    try:
        from ragas import RunConfig
        run_cfg = RunConfig(max_workers=1, timeout=300)
    except (ImportError, TypeError):
        run_cfg = None

    print(f"\n  Running Ragas on {len(records)} records (contexts truncated to "
          f"{MAX_CHUNK_CHARS} chars × {MAX_CHUNKS} chunks per case)...")
    t0      = time.perf_counter()
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        **(dict(run_config=run_cfg) if run_cfg else {}),
    )
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    # ── 7. Build report ───────────────────────────────────────────────────────
    df = results.to_pandas()

    def _col(df, *names):
        """Return first matching column name (case-insensitive partial match)."""
        for n in names:
            for c in df.columns:
                if n.lower() in c.lower():
                    return c
        return None

    faith_col = _col(df, "faithfulness")
    ar_col    = _col(df, "response_relevancy", "answer_relevancy")
    cp_col    = _col(df, "context_precision") if ctx_obj else None

    def _fmt(col, i):
        if col is None:
            return "N/A"
        v = df[col].iloc[i]
        return "N/A" if str(v) == "nan" else f"{v:.3f}"

    def _avg(col):
        if col is None:
            return float("nan")
        return df[col].mean()

    avg_f  = _avg(faith_col)
    avg_ar = _avg(ar_col)
    avg_cp = _avg(cp_col)

    def _s(v, threshold, good, bad):
        return f"{v:.3f}  {'✅ ' + good if v >= threshold else '⚠ ' + bad}"

    lines = [
        f"# Ragas Evaluation — {time.strftime('%Y-%m-%d_%H-%M-%S')}",
        "",
        f"**Judge model:** `{OLLAMA_MODEL}` (local Ollama OpenAI-compatible API — no external calls)",
        f"**Embedding model:** `{EMBED_MODEL}` (local sentence-transformers)",
        f"**Cases evaluated:** {len(records)}",
        f"**Duration:** {elapsed:.1f}s",
        "",
        "## Metric Definitions",
        "",
        "| Metric | What it measures |",
        "|--------|-----------------|",
        "| **Faithfulness** | Claims in final answer supported by retrieved chunks (0–1) |",
        "| **ResponseRelevancy** | Answer is relevant and complete for the question (0–1) |",
        "| **Context Precision** | Retrieved chunks are relevant to the query — no ground truth needed (0–1) |",
        "",
        "## Per-Case Scores",
        "",
        "| ID | Faithfulness | Response Relevancy | Context Precision |",
        "|----|-------------|-------------------|-------------------|",
    ]

    for i, record in enumerate(records):
        lines.append(
            f"| {record['id']} | {_fmt(faith_col, i)} | {_fmt(ar_col, i)} | {_fmt(cp_col, i)} |"
        )

    lines += [
        "",
        f"| **AVG** | **{avg_f:.3f}** | **{avg_ar:.3f}** | **{avg_cp:.3f}** |",
        "",
        "## Interpretation",
        "",
        f"- Faithfulness **{_s(avg_f, 0.8, 'Answers grounded in retrieved context', 'Some claims may not be supported by chunks')}**",
        f"- Response Relevancy **{_s(avg_ar, 0.7, 'Answers are on-topic', 'Answers may be drifting from the query')}**",
        f"- Context Precision **{_s(avg_cp, 0.7, 'CRAG retrieval is accurate', 'Some retrieved chunks are not useful — consider tightening CRAG threshold')}**",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [Report] → {output_path}")


# ── Entry ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) >= 2:
        raw_path = Path(sys.argv[1])
        if not raw_path.exists():
            print(f"[Error] File not found: {raw_path}")
            sys.exit(1)
    else:
        raw_path = get_latest_raw_outputs()
        print(f"[Info] Using most recent raw outputs: {raw_path.name}")

    stem     = raw_path.stem.replace("raw_outputs_", "")
    out_path = raw_path.parent / f"ragas_{stem}.md"

    print(f"\n{'='*65}")
    print(f"  Ragas Phase 2 Evaluation")
    print(f"  Input:  {raw_path.name}")
    print(f"  Output: {out_path.name}")
    print(f"{'='*65}")

    records = load_raw_outputs(raw_path)
    if not records:
        print("[Error] No valid records with contexts. Cannot run Ragas.")
        sys.exit(1)

    run_ragas(records, out_path)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
