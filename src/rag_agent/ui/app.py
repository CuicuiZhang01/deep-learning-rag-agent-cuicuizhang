"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )
    if st.sidebar.button("Ingest Documents", disabled=not uploaded_files):
        try:
            with TemporaryDirectory() as temp_dir:
                file_paths = []
                for index, uploaded_file in enumerate(uploaded_files):
                    upload_dir = Path(temp_dir) / str(index)
                    upload_dir.mkdir()
                    file_path = upload_dir / Path(uploaded_file.name).name
                    file_path.write_bytes(uploaded_file.getvalue())
                    file_paths.append(file_path)
                chunks = chunker.chunk_files(file_paths)
                result = store.ingest(chunks)
            st.session_state["last_ingestion_result"] = result
            summary = (
                f"{result.ingested} chunks added, "
                f"{result.skipped} duplicates skipped"
            )
            if result.ingested:
                st.sidebar.success(summary)
            else:
                st.sidebar.warning(summary)
            if not chunks:
                st.sidebar.warning("No readable chunks found in the uploaded files.")
            for error in result.errors:
                st.sidebar.error(error)
        except Exception as exc:
            st.sidebar.error(f"Document ingestion failed: {exc}")

    try:
        documents = store.list_documents()
        st.session_state["ingested_documents"] = documents
    except Exception as exc:
        st.sidebar.error(f"Could not load documents: {exc}")
        return

    if not documents:
        st.sidebar.info("Upload .pdf or .md files to populate the corpus.")
    for document in documents:
        source = document["source"]
        st.sidebar.write(
            f"{source} | {document['topic']} | "
            f"{document['chunk_count']} chunks"
        )
        if st.sidebar.button("🗑 Remove", key=f"remove_document_{source}"):
            try:
                store.delete_document(source)
                st.session_state["ingested_documents"] = store.list_documents()
                if st.session_state.get("selected_document") == source:
                    st.session_state["selected_document"] = None
            except Exception as exc:
                st.sidebar.error(f"Could not remove {source}: {exc}")
            else:
                st.rerun()


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    try:
        stats = store.get_collection_stats()
    except Exception as exc:
        st.sidebar.error(f"Could not load corpus statistics: {exc}")
        return

    st.sidebar.metric("Total Chunks", stats["total_chunks"])
    st.sidebar.write("Topics:", ", ".join(stats["topics"]))
    if stats["bonus_topics_present"]:
        st.sidebar.success("✅ Bonus topics present")
    else:
        st.sidebar.warning("⚠️ No bonus topics yet")


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    documents = st.session_state.get("ingested_documents", [])
    if not documents:
        st.session_state["selected_document"] = None
        st.info("Ingest documents using the sidebar to view content here.")
        return

    sources = [document["source"] for document in documents]
    if st.session_state.get("selected_document") not in sources:
        st.session_state["selected_document"] = sources[0]
    selected_source = st.selectbox(
        "Select document", options=sources, key="selected_document"
    )
    try:
        chunks = store.get_document_chunks(selected_source)
    except Exception as exc:
        st.error(f"Could not load {selected_source}: {exc}")
        return

    last_response = next(
        (
            message
            for message in reversed(st.session_state.get("chat_history", []))
            if message["role"] == "assistant"
        ),
        {},
    )
    scores = {
        chunk.chunk_id: chunk.score
        for chunk in last_response.get("retrieved_chunks", [])
    }
    with st.container(height=400):
        if not chunks:
            st.info("No chunks found for this document.")
        for chunk in chunks:
            metadata = chunk.metadata
            st.caption(
                f"{metadata.topic} | {metadata.difficulty} | {metadata.type}"
            )
            st.markdown(chunk.chunk_text)
            if chunk.chunk_id in scores:
                st.caption(f"Similarity: {scores[chunk.chunk_id]:.2f}")
            st.divider()

    st.caption(f"{len(chunks)} chunks")
    if chunks:
        topics = sorted({chunk.metadata.topic for chunk in chunks})
        difficulties = sorted({chunk.metadata.difficulty for chunk in chunks})
        types = sorted({chunk.metadata.type for chunk in chunks})
        st.caption(
            f"Topics: {', '.join(topics)} | "
            f"Difficulties: {', '.join(difficulties)} | "
            f"Types: {', '.join(types)}"
        )


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------


def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent.
    Displays source citations with every response.
    Shows a clear "no relevant context" indicator when the
    hallucination guard fires.

    Parameters
    ----------
    graph : CompiledStateGraph
        The compiled LangGraph agent from get_compiled_graph().
    """
    st.subheader("💬 Interview Prep Chat")

    # Filters
    col_topic, col_diff = st.columns(2)
    with col_topic:
        st.selectbox(
            "Topic",
            options=[
                None, "ANN", "CNN", "RNN", "LSTM", "Seq2Seq",
                "Autoencoder", "SOM", "BoltzmannMachine", "GAN",
            ],
            format_func=lambda topic: "All topics" if topic is None else topic,
            key="topic_filter",
        )
    with col_diff:
        st.selectbox(
            "Difficulty",
            options=[None, "beginner", "intermediate", "advanced"],
            format_func=lambda difficulty: (
                "All difficulties" if difficulty is None else difficulty.capitalize()
            ),
            key="difficulty_filter",
        )

    # Chat history display
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("⚠️ No relevant content found in corpus.")

    # Chat input
    query = st.chat_input("Ask about a deep learning topic...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = graph.invoke(
                        {
                            "messages": [HumanMessage(content=query)],
                            "topic_filter": st.session_state.topic_filter,
                            "difficulty_filter": st.session_state.difficulty_filter,
                        },
                        config={
                            "configurable": {
                                "thread_id": st.session_state.thread_id,
                            }
                        },
                    )
                response: AgentResponse = result["final_response"]
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "no_context_found": response.no_context_found,
                    }
                )
                st.markdown(response.answer)
                if response.sources:
                    with st.expander("📎 Sources"):
                        for source in response.sources:
                            st.caption(source)
                if response.no_context_found:
                    st.warning("⚠️ No relevant content found in corpus.")


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
