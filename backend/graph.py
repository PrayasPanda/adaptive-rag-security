import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from router import route_question
from retriever import retrieve_chunks
from grader import grade_chunks, rewrite_query
from generator import generate_answer, extract_sources
from hallucination import check_hallucination
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


# Define the state that flows through the graph
class RAGState(TypedDict):
    question: str
    doc_name: str
    strategy: str
    chunks: list
    relevant_chunks: list
    answer: str
    is_grounded: bool
    sources: list
    iterations: int
    retrieval_retries: int
    generation_retries: int


# ── NODE FUNCTIONS ──────────────────────────────────────

def router_node(state: RAGState) -> RAGState:
    """Node 1: Decide retrieval strategy"""
    print(f"[Router] Question: {state['question']}")
    strategy = route_question(state["question"])
    print(f"[Router] Strategy: {strategy}")
    return {**state, "strategy": strategy}


def retriever_node(state: RAGState) -> RAGState:
    """Node 2: Retrieve chunks from FAISS"""
    print(f"[Retriever] Strategy: {state['strategy']}")
    chunks = retrieve_chunks(
        state["question"],
        state["doc_name"],
        state["strategy"]
    )
    print(f"[Retriever] Retrieved {len(chunks)} chunks")
    return {**state, "chunks": chunks}


def grader_node(state: RAGState) -> RAGState:
    """Node 3: Grade chunks for relevance"""
    print(f"[Grader] Grading {len(state['chunks'])} chunks...")
    relevant = grade_chunks(state["question"], state["chunks"])
    print(f"[Grader] {len(relevant)} relevant chunks found")

    retries = state.get("retrieval_retries", 0)

    if not relevant and retries < MAX_RETRIES:
        # Rewrite query and try again
        new_question = rewrite_query(state["question"])
        print(f"[Grader] No relevant chunks. Rewriting query: {new_question}")
        return {
            **state,
            "relevant_chunks": [],
            "question": new_question,
            "retrieval_retries": retries + 1
        }

    return {
        **state,
        "relevant_chunks": relevant,
        "retrieval_retries": retries
    }


def generator_node(state: RAGState) -> RAGState:
    """Node 4: Generate answer from relevant chunks"""
    chunks = state["relevant_chunks"] or state["chunks"]
    print(f"[Generator] Generating answer from {len(chunks)} chunks...")

    answer = generate_answer(state["question"], chunks)
    sources = extract_sources(chunks)
    iterations = state.get("iterations", 0) + 1

    print(f"[Generator] Answer generated (iteration {iterations})")
    return {
        **state,
        "answer": answer,
        "sources": sources,
        "iterations": iterations
    }


def hallucination_node(state: RAGState) -> RAGState:
    """Node 5: Check if answer is grounded"""
    chunks = state["relevant_chunks"] or state["chunks"]
    print("[Hallucination Checker] Verifying answer...")

    is_grounded = check_hallucination(state["answer"], chunks)
    gen_retries = state.get("generation_retries", 0)

    print(f"[Hallucination Checker] Grounded: {is_grounded}")

    if not is_grounded and gen_retries < MAX_RETRIES:
        return {
            **state,
            "is_grounded": False,
            "generation_retries": gen_retries + 1
        }

    return {**state, "is_grounded": is_grounded}


# ── CONDITIONAL EDGE FUNCTIONS ──────────────────────────

def should_retrieve_again(state: RAGState) -> str:
    """After grading: go to generator or retrieve again"""
    if state["relevant_chunks"]:
        return "generator"
    if state.get("retrieval_retries", 0) >= MAX_RETRIES:
        return "generator"
    return "retriever"


def should_generate_again(state: RAGState) -> str:
    """After hallucination check: end or regenerate"""
    if state["is_grounded"]:
        return "end"
    if state.get("generation_retries", 0) >= MAX_RETRIES:
        return "end"
    return "generator"


# ── BUILD THE GRAPH ─────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("generator", generator_node)
    graph.add_node("hallucination_checker", hallucination_node)

    # Set entry point
    graph.set_entry_point("router")

    # Define edges
    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "grader")

    # Conditional: after grading
    graph.add_conditional_edges(
        "grader",
        should_retrieve_again,
        {
            "generator": "generator",
            "retriever": "retriever"
        }
    )

    # After generating always check hallucination
    graph.add_edge("generator", "hallucination_checker")

    # Conditional: after hallucination check
    graph.add_conditional_edges(
        "hallucination_checker",
        should_generate_again,
        {
            "end": END,
            "generator": "generator"
        }
    )

    return graph.compile()


# Compile once at module level
rag_graph = build_rag_graph()


def run_pipeline(question: str, doc_name: str) -> dict:
    """
    Run the full adaptive RAG pipeline.
    Returns answer, strategy, iterations, grounded status, sources.
    """
    initial_state = RAGState(
        question=question,
        doc_name=doc_name,
        strategy="SIMPLE",
        chunks=[],
        relevant_chunks=[],
        answer="",
        is_grounded=False,
        sources=[],
        iterations=0,
        retrieval_retries=0,
        generation_retries=0
    )

    final_state = rag_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "strategy": final_state["strategy"],
        "iterations": final_state["iterations"],
        "is_grounded": final_state["is_grounded"],
        "sources": final_state["sources"]
    }
