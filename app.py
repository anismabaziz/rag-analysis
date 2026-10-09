"""Streamlit demo for the retrieval comparison.

Three tabs. Live compare asks one question to two architectures side by side,
using the same registry and cached answers as the CLI. Results reads the
committed run files and renders the pooled table plus a per-question drilldown.
Qdrant shows store status and per-collection ingest.

Run it with: streamlit run app.py
Live queries need Qdrant up (docker compose up -d). Cached answers replay
with no API key; anything uncached needs GROQ_API_KEY.
"""

import asyncio
import time
from pathlib import Path

import streamlit as st

from config.configuration import frozen_configuration
from config.params import Params
from core.chunk import REFUSAL, build_context
from core.registry import all_architectures, architecture
from evaluation.cache import CACHE_DIR, generate_with_cache
from evaluation.set import Stratum
from evaluation.summary import (
    POOLED_SCOPE,
    load_committed_runs,
    load_runs,
    summarize,
)
from vector.store import collection_points, get_qdrant_client

st.set_page_config(page_title="RAG analysis", layout="wide")
st.title("RAG analysis")
st.caption("Same corpus, same generator. Only retrieval changes.")


def render_qdrant_tab():
    """Qdrant status and per-collection ingest, so a missing collection is one click to fix."""
    st.caption(f"Vector store behind live retrieval at {Params.QDRANT_URL}.")

    try:
        client = get_qdrant_client()
        client.get_collections()
    except Exception:
        st.error("Qdrant is down")
        st.caption("Start it yourself with docker compose up -d qdrant.")
        return

    st.success("Qdrant is up")

    architectures = all_architectures()
    statuses = [
        (registered, collection_points(client, registered.collection))
        for registered in architectures
    ]
    ready = sum(1 for _, count in statuses if count)
    total_points = sum(count for _, count in statuses if count is not None)

    summary_left, summary_mid, summary_right = st.columns(3)
    summary_left.metric("Collections ready", f"{ready}/{len(statuses)}")
    summary_mid.metric("Total points", f"{total_points:,}")
    summary_right.metric("Architectures", str(len(statuses)))
    if st.button("Refresh status"):
        st.rerun()

    st.divider()
    st.subheader("Collections")

    for registered, count in statuses:
        with st.container(border=True):
            info, state, action = st.columns([3, 1, 1])
            with info:
                st.markdown(f"**{registered.name}**")
                st.caption(registered.description)
                st.caption(
                    f"`{registered.collection}` · vectors: {', '.join(registered.vectors)}"
                )
            with state:
                if count is None:
                    st.metric("Points", "missing")
                else:
                    st.metric("Points", f"{count:,}")
            with action:
                st.write("")
                if not count:
                    st.warning("Not ingested" if count is None else "Empty")
                    if st.button("Ingest", key=f"ingest-{registered.name}"):
                        with st.spinner(
                            f"Ingesting {registered.name}... reads the whole corpus, takes a while"
                        ):
                            try:
                                architecture(registered.name).ingest()
                            except Exception as error:
                                st.error(f"{registered.name} failed: {error}")
                            else:
                                st.rerun()
                else:
                    st.success("Ready")


@st.cache_resource(show_spinner=False)
def get_pipeline(name: str):
    return architecture(name).build()


@st.cache_data(show_spinner=False)
def list_names() -> list[str]:
    return [item.name for item in all_architectures()]


@st.cache_data(show_spinner=False)
def load_summary():
    try:
        runs = load_committed_runs()
    except (FileNotFoundError, ValueError):
        runs = load_runs()
    return summarize(runs), runs


def run_live(name: str, question: str, cache_dir: Path):
    pipeline = get_pipeline(name)
    registered = architecture(name)
    configuration = frozen_configuration(registered.resolved_chunker())
    started = time.perf_counter()
    try:
        chunks = asyncio.run(pipeline.retrieve(question))
    except Exception as error:
        if "doesn't exist" in str(error):
            raise RuntimeError(
                f"collection '{registered.collection}' is not in Qdrant yet. "
                f"Ingest it from the Qdrant tab, or run "
                f"`uv run python scripts/init_qdrant.py {name}`."
            ) from error
        raise
    retrieval_s = time.perf_counter() - started
    if not chunks:
        return {
            "answer": REFUSAL,
            "chunks": [],
            "retrieval_s": retrieval_s,
            "cached": False,
        }
    context = build_context(chunks)
    started = time.perf_counter()
    answer = asyncio.run(
        generate_with_cache(pipeline, configuration, question, context, cache_dir)
    )
    generation_s = time.perf_counter() - started
    return {
        "answer": answer,
        "chunks": chunks,
        "retrieval_s": retrieval_s,
        "generation_s": generation_s,
    }


def render_result(name: str, outcome: dict):
    st.subheader(name)
    st.caption(architecture(name).description)
    st.write(outcome["answer"])
    meta = f"retrieval {outcome['retrieval_s']:.2f}s"
    if outcome.get("generation_s") is not None:
        meta += f" / generation {outcome['generation_s']:.2f}s"
    st.caption(meta)
    with st.expander(f"What {name} retrieved ({len(outcome['chunks'])} chunks)"):
        for index, chunk in enumerate(outcome["chunks"], start=1):
            with st.container(border=True):
                head, score = st.columns([3, 1])
                with head:
                    st.markdown(f"**Chunk {index}** · {chunk.citation()}")
                    provenance = chunk.provenance
                    bits = []
                    if provenance.source:
                        bits.append(f"source: {provenance.source}")
                    if provenance.section:
                        bits.append(f"section: {provenance.section}")
                    if provenance.page is not None:
                        bits.append(f"page: {provenance.page}")
                    if provenance.node_id:
                        bits.append(f"node: {provenance.node_id}")
                    if bits:
                        st.caption("  ·  ".join(bits))
                with score:
                    st.metric("similarity", f"{chunk.score:.4f}")
                if 0.0 <= chunk.score <= 1.0:
                    st.progress(chunk.score)
                text = chunk.text if len(chunk.text) <= 800 else chunk.text[:800] + "…"
                st.write(text)


live_tab, results_tab, qdrant_tab = st.tabs(["Live compare", "Results", "Qdrant"])

with live_tab:
    names = list_names()
    left, right = st.columns(2)
    with left:
        arch_a = st.selectbox(
            "Architecture A",
            names,
            index=names.index("hybrid") if "hybrid" in names else 0,
        )
        st.caption(architecture(arch_a).description)
    with right:
        default_b = names.index("naive") if "naive" in names else min(1, len(names) - 1)
        arch_b = st.selectbox("Architecture B", names, index=default_b)
        st.caption(architecture(arch_b).description)
    question = st.text_input(
        "Question",
        "how are the positional encodings scaled?",
    )
    go = st.button("Compare", type="primary")
    if go and question.strip():
        col_a, col_b = st.columns(2)
        for column, arch in ((col_a, arch_a), (col_b, arch_b)):
            with column:
                with st.spinner(f"Asking {arch}…"):
                    try:
                        render_result(arch, run_live(arch, question.strip(), CACHE_DIR))
                    except Exception as error:
                        st.error(f"{arch} failed: {error}")
                        st.caption(
                            "Live queries need Qdrant up: docker compose up -d. Uncached answers need GROQ_API_KEY."
                        )

with results_tab:
    try:
        summary, runs = load_summary()
    except (FileNotFoundError, ValueError) as error:
        st.info(
            f"No run files yet: {error}. Run `uv run rag-analysis run hybrid --domain papers` first."
        )
    else:
        st.subheader("Which retriever finds the answer?")
        st.caption(
            "One run per architecture over the same 32 papers questions, generation frozen, "
            "so gaps come from retrieval alone. The claim under test: hybrid leads on "
            "identifier-heavy questions (rare names, numbers, symbols) and trails dense-only "
            "on paraphrased ones. Read recall@5 and mrr per slice, not the pooled average."
        )
        st.caption(
            f"{len(summary.files)} runs at {', '.join(summary.commits)} "
            f"/ {summary.generation_model} / depth {summary.retrieval_depth}"
        )
        st.caption("chunk-fixed, chunk-semantic, and chunk-hierarchical are naive dense retrieval. Only the chunking differs.")
        deepest = summary.depths[-1]
        st.caption(
            f"Recall is read at depth {deepest}, the committed depth every architecture "
            "retrieves at, so the rows compare rankings rather than context lengths."
        )
        by_scope = {(row.architecture, row.scope): row for row in summary.rows}
        identifiers = f"{POOLED_SCOPE}:{Stratum.IDENTIFIER_HEAVY.value}"
        paraphrase = f"{POOLED_SCOPE}:{Stratum.PARAPHRASE.value}"
        rows = [
            {
                "architecture": name,
                f"recall@{deepest}": round(
                    by_scope[(name, POOLED_SCOPE)].recall_at.get(deepest, 0.0), 4
                ),
                f"identifiers recall@{deepest}": round(
                    by_scope[(name, identifiers)].recall_at.get(deepest, 0.0), 4
                ),
                f"paraphrase recall@{deepest}": round(
                    by_scope[(name, paraphrase)].recall_at.get(deepest, 0.0), 4
                ),
                "mrr": round(by_scope[(name, POOLED_SCOPE)].reciprocal_rank, 4),
            }
            for name in sorted({row.architecture for row in summary.rows})
        ]
        st.dataframe(rows, use_container_width=True)
        scopes = sorted({run.domain for run in runs})
        scope = st.selectbox("Domain drilldown", scopes)
        arch_names = sorted({run.architecture for run in runs if run.domain == scope})
        pick = st.selectbox("Run", arch_names)
        target = next(
            run for run in runs if run.domain == scope and run.architecture == pick
        )
        qrows = [
            {
                "question": result.question,
                "stratum": str(result.stratum),
                "recall": result.recall,
                "rr": result.reciprocal_rank,
                "cite": result.citation_accuracy,
                "retrieval_s": round(result.retrieval_latency_s, 3),
            }
            for result in target.results
        ]
        st.dataframe(qrows, use_container_width=True)
        selected = st.selectbox(
            "Question detail", [result.question for result in target.results]
        )
        detail = next(
            result for result in target.results if result.question == selected
        )
        st.write(detail.answer)
        with st.expander(f"Retrieved chunks ({len(detail.retrieved)})"):
            for item in detail.retrieved:
                st.markdown(f"**{item.document}** {item.section} (rank {item.rank})")
                st.text((item.text[:500] + "…") if len(item.text) > 500 else item.text)

with qdrant_tab:
    render_qdrant_tab()
