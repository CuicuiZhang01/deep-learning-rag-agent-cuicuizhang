"""
nodes.py
========
LangGraph node functions for the RAG interview preparation agent.

Each function in this module is a node in the agent state graph.
Nodes receive the current AgentState, perform their operation,
and return a dict of state fields to update.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from rag_agent.agent.prompts import (
    QUESTION_GENERATION_PROMPT,
    SYSTEM_PROMPT,
)
from rag_agent.agent.state import AgentResponse, AgentState, RetrievedChunk
from rag_agent.config import LLMFactory, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Node: Query Rewriter
# ---------------------------------------------------------------------------


def query_rewrite_node(state: AgentState) -> dict:
    """
    Rewrite the user's query to maximise retrieval effectiveness.

    Natural language questions are often poorly suited for vector
    similarity search. This node rephrases the query into a form
    that produces better embedding matches against the corpus.

    Example
    -------
    Input:  "I'm confused about how LSTMs remember things long-term"
    Output: "LSTM long-term memory cell state forget gate mechanism"

    Interview talking point: query rewriting is a production RAG pattern
    that significantly improves retrieval recall. It acknowledges that
    users do not phrase queries the way documents are written.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: messages (for context).

    Returns
    -------
    dict
        Updates: original_query, rewritten_query.
    """
    latest_message = next(
        (
            message
            for message in reversed(state.get("messages", []))
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    original_query = latest_message.text if latest_message is not None else ""
    rewritten = original_query

    if original_query.strip():
        try:
            llm = LLMFactory(get_settings()).create()
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Rewrite the question as a concise vector-search query. "
                            "Preserve its meaning and key technical terms. "
                            "Return only the query; do not answer the question."
                        )
                    ),
                    HumanMessage(content=original_query),
                ]
            )
            rewritten = response.text.strip() or original_query
        except Exception:
            rewritten = original_query

    return {"original_query": original_query, "rewritten_query": rewritten}


# ---------------------------------------------------------------------------
# Node: Retriever
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_vector_store() -> VectorStoreManager:
    """Reuse the vector store and embedding model across retrieval calls."""
    return VectorStoreManager()


def retrieval_node(state: AgentState) -> dict:
    """
    Retrieve relevant chunks from ChromaDB based on the rewritten query.

    Sets the no_context_found flag if no chunks meet the similarity
    threshold. This flag is checked by generation_node to trigger
    the hallucination guard.

    Interview talking point: separating retrieval into its own node
    makes it independently testable and replaceable — you could swap
    ChromaDB for Pinecone or Weaviate by changing only this node.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: rewritten_query, topic_filter, difficulty_filter.

    Returns
    -------
    dict
        Updates: retrieved_chunks, no_context_found.
    """
    manager = _get_vector_store()
    chunks = manager.query(
        query_text=state["rewritten_query"],
        topic_filter=state.get("topic_filter"),
        difficulty_filter=state.get("difficulty_filter"),
    )
    return {"retrieved_chunks": chunks, "no_context_found": not chunks}


# ---------------------------------------------------------------------------
# Node: Generator
# ---------------------------------------------------------------------------


def generation_node(state: AgentState) -> dict:
    """
    Generate the final response using retrieved chunks as context.

    Implements the hallucination guard: if no_context_found is True,
    returns a clear "no relevant context" message rather than allowing
    the LLM to answer from parametric memory.

    Implements token-aware conversation memory trimming: when the
    message history approaches max_context_tokens, the oldest
    non-system messages are removed.

    Interview talking point: the hallucination guard is the most
    commonly asked about production RAG pattern. Interviewers want
    to know how you prevent the model from confidently making up
    information when the retrieval step finds nothing relevant.

    Parameters
    ----------
    state : AgentState
        Current graph state.
        Reads: retrieved_chunks, no_context_found, messages,
               original_query, topic_filter.

    Returns
    -------
    dict
        Updates: final_response, messages (with new AIMessage appended).
    """
    # ---- Hallucination Guard -----------------------------------------------
    chunks = state.get("retrieved_chunks", [])
    if state.get("no_context_found", False) or not chunks:
        no_context_message = (
            "I was unable to find relevant information in the corpus for your query. "
            "This may mean the topic is not yet covered in the study material, or "
            "your query may need to be rephrased. Please try a more specific "
            "deep learning topic such as 'LSTM forget gate' or 'CNN pooling layers'."
        )
        response = AgentResponse(
            answer=no_context_message,
            sources=[],
            confidence=0.0,
            no_context_found=True,
            rewritten_query=state.get("rewritten_query", ""),
        )
        return {
            "final_response": response,
            "messages": [AIMessage(content=no_context_message)],
        }

    # ---- Build Context from Retrieved Chunks --------------------------------
    settings = get_settings()
    llm = LLMFactory(settings).create()
    context = "\n\n".join(
        f"[SOURCE: {chunk.metadata.topic} | {chunk.metadata.source}]\n"
        f"{chunk.chunk_text}\n"
        for chunk in chunks
    )
    sources = list(dict.fromkeys(chunk.to_citation() for chunk in chunks))
    confidence = sum(chunk.score for chunk in chunks) / len(chunks)
    prompt = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"Retrieved source material:\n\n{context}"),
    ]
    question = HumanMessage(content=state["original_query"])
    history = list(state.get("messages", []))
    # The current question is already in the accumulated graph history.
    if history and isinstance(history[-1], HumanMessage):
        history = history[:-1]
    history_budget = max(
        0,
        settings.max_context_tokens
        - count_tokens_approximately([*prompt, question]),
    )
    trimmed_history = trim_messages(
        history,
        max_tokens=history_budget,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        include_system=False,
    )
    result = llm.invoke([*prompt, *trimmed_history, question])
    new_ai_message = AIMessage(content=result.content)
    response = AgentResponse(
        answer=new_ai_message.text,
        sources=sources,
        confidence=confidence,
        no_context_found=False,
        rewritten_query=state.get("rewritten_query", ""),
    )
    return {"final_response": response, "messages": [new_ai_message]}


# ---------------------------------------------------------------------------
# Routing Function
# ---------------------------------------------------------------------------


def should_retry_retrieval(state: AgentState) -> str:
    """
    Conditional edge function: select generation or the no-context guard.

    Called after retrieval_node. This simple version performs no retries.
    Both branches enter generation_node, which returns a fixed response
    without calling the LLM when no context was found.

    Interview talking point: conditional edges in LangGraph enable
    agentic behaviour — the graph makes decisions about its own
    execution path rather than following a fixed sequence.

    Parameters
    ----------
    state : AgentState
        Current graph state. Reads: no_context_found, retrieved_chunks.

    Returns
    -------
    str
        "generate" — proceed to generation_node.
        "end"      — return the no-context response through the guard.

    Notes
    -----
    Retry logic should be limited to one attempt to prevent infinite loops.
    Track retry count in AgentState if implementing retry behaviour.
    """
    if state.get("no_context_found", False) or not state.get("retrieved_chunks"):
        return "end"
    return "generate"
