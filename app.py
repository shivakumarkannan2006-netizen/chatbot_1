# Shriramajayam

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from typing import List, Dict
# Import 'app' which is the compiled graph in langgraph_portion.py
from langgraph_portion import app as graph_app
# Assuming HumanMessage, AIMessage, BaseMessage are imported correctly via agent_functions
from agent_functions import AgentState, HumanMessage, AIMessage, BaseMessage

# --- Flask Initialization ---
app = Flask(__name__)
# Enable CORS for all origins
CORS(app)


# --- Helper Function: Convert JSON history to LangChain BaseMessage objects ---
def json_to_langchain_messages(history_json: List[Dict[str, str]]) -> List[BaseMessage]:
    """Converts a list of JSON message objects (role, content) to LangChain BaseMessage objects."""

    messages = []
    for msg in history_json:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


# --- NEW ROUTE: Serve the HTML template ---
@app.route("/")
def index():
    # Assumes your HTML file is named index_pink_blue.html and is inside a folder named 'templates'
    return render_template("index_teal.html")


# --- Flask Route: Handle RAG Queries ---
@app.route("/query-rag", methods=["POST"])
def query_rag():
    data = request.json
    # The frontend is sending the full history array, which already contains the last prompt
    history_json = data.get("history", [])

    # We don't strictly need the separate 'prompt' field if the history is current,
    # but we can grab the last message content for a quick check.
    user_prompt = data.get("prompt")
    if not user_prompt:  # Check if the prompt field is missing from JSON
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # 1. Convert the JSON history (which includes the new prompt) into LangChain objects
        langchain_messages = json_to_langchain_messages(history_json)

        # 2. Define the initial state for the LangGraph
        initial_state: AgentState = {"messages": langchain_messages}

        # 3. Run the LangGraph
        final_state = graph_app.invoke(initial_state)

        # 4. Extract the final AI response from the state
        final_ai_message = final_state["messages"][-1]

        if isinstance(final_ai_message, AIMessage):
            response_text = final_ai_message.content
        else:
            # This line will only run if the graph finished, but the final message wasn't an AIMessage.
            response_text = "The agent finished running but did not produce a final AI message."

        # 5. Return the response to the frontend
        return jsonify({
            "answer": response_text
        })

    except Exception as e:
        # Log the full error to the console for debugging
        print("\n--- LangGraph Execution Traceback ---\n")
        # Print the traceback so you can see the exact cause
        import traceback
        traceback.print_exc()
        print("\n-------------------------------------\n")

        # The error details now include the traceback output to aid debugging
        return jsonify({
            "error": "An internal error occurred during RAG processing. Check server console for full traceback.",
            "details": str(e)
        }), 500


# --- Run the Flask App ---
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)