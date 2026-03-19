from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from app.pipeline.graph.state import CDSSGraphState

from app.pipeline.graph.nodes import (decompose_node, retrieve_node, retrieve_pdf_node, evaluate_node, generate_node, critic_node, citation_node)

workflow = StateGraph(CDSSGraphState)

workflow.add_node("Decomposer", decompose_node)
workflow.add_node("DB_Retriever", retrieve_node)
workflow.add_node("PDF_Retriever", retrieve_pdf_node)
workflow.add_node("Evaluator", evaluate_node)
workflow.add_node("Generator", generate_node)
workflow.add_node("Critic", critic_node)
workflow.add_node("Citation_Enforcer", citation_node)

# ── True parallel fan-out using Send ──────────────────────────────────────────
# add_edge("Decomposer", "DB_Retriever") + add_edge("Decomposer", "PDF_Retriever")
# does NOT produce parallel execution — it runs sequentially and the second
# node's state update can overwrite the first's.
# Send() dispatches both nodes concurrently and merges results via the
# Annotated[list, operator.add] reducers in CDSSGraphState.
def fan_out_retrievers(state: CDSSGraphState):
    return [
        Send("DB_Retriever",  state),
        Send("PDF_Retriever", state),
    ]

workflow.add_edge(START, "Decomposer")
workflow.add_conditional_edges("Decomposer", fan_out_retrievers, ["DB_Retriever", "PDF_Retriever"])
workflow.add_edge("DB_Retriever", "Evaluator")
workflow.add_edge("PDF_Retriever", "Evaluator")
workflow.add_edge("Evaluator", "Generator")
workflow.add_edge("Generator", "Critic")
workflow.add_edge("Critic", "Citation_Enforcer")
workflow.add_edge("Citation_Enforcer", END)

cdss_app = workflow.compile()