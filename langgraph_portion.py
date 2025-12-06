# Shriramajayam

from langgraph.graph import END, StateGraph
# 💡 CRITICAL CHANGE: Import the module itself, not just the functions
import agent_functions as af
import sys
# Import itemgetter for the conditional edge fix
from operator import itemgetter

# Reference the state definition and functions using the alias 'af'
workflow = StateGraph(af.AgentState)

# 💡 Use af. prefix for all node functions
workflow.add_node("router", af.router_node)
workflow.add_node("respond_with_gk", af.respond_with_gk)
workflow.add_node("retrieve_info_from_outside", af.retrieve_info_from_outside)
workflow.add_node("retrieve_info_from_conversation", af.retrieve_info_from_conversation)
workflow.add_node("synthesize_rag_answer", af.synthesize_rag_answer) # NEW NODE

workflow.set_entry_point("router")

# Conditional Edges from Router
workflow.add_conditional_edges(
    "router",
    # FIX: Read the 'next_node' key for the immediate routing
    lambda x: x["next_node"],
    {
        "retrieve_info_from_outside": "retrieve_info_from_outside",
        "retrieve_info_from_conversation": "retrieve_info_from_conversation",
        "respond_with_gk": "respond_with_gk",
    }
)

# --- NEW RAG CHAINING LOGIC ---

# 1. After retrieving external info, check the router's decision to see if it needs the conversation step
workflow.add_conditional_edges(
    "retrieve_info_from_outside",
    # FIX: Read the 'router_decision' key from the state (which persists)
    lambda x: "COMPLEX_RAG_CONTINUE" if x.get("router_decision") == "COMPLEX_RAG" else "SIMPLE_RAG_END",
    {
        # If it was a COMPLEX_RAG query, continue to conversation retrieval
        "COMPLEX_RAG_CONTINUE": "retrieve_info_from_conversation",
        # If it was an EXTERNAL_RAG query, skip conversation and synthesize
        "SIMPLE_RAG_END": "synthesize_rag_answer",
    }
)

# 2. After retrieving conversation info (comes from either initial router call or COMPLEX_RAG chain), always synthesize
workflow.add_edge("retrieve_info_from_conversation", "synthesize_rag_answer")

# 3. All RAG paths finish at the Synthesis Node
workflow.add_edge("synthesize_rag_answer", END)

# 4. GK path still finishes immediately
workflow.add_edge("respond_with_gk", END)

# Export the compiled graph as 'app' to be consistent with app.py's import
app = workflow.compile()

try:
    png_data = app.get_graph().draw_mermaid_png()

    with open("chatbot.png", "wb") as f:
        f.write(png_data)
except Exception as e:
    print(f"Graph drawing failed. Please ensure all required dependencies are installed. Error: {e}", file=sys.stderr)