# Shriramajayam
from typing import TypedDict, List, Literal, Dict
from langchain_community.llms import Ollama
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.documents.base import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Initialization ---
llm = Ollama(
    model="llama3:8b",
    temperature=0.1
)

# 2. Use a dedicated, fast embedding model
embd_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

DESSERTINO_WEB_PATHS = [
    "https://dessertinoglobal.com/menu/",
    "https://dessertinoglobal.com/",
    "https://dessertinoglobal.com/about-us/",
    "https://franchise.dessertinoglobal.com/",
    "https://dessertinoglobal.com/stores/",
    "https://dessertinoglobal.com/contact/"
]

# This vectorstore is initialized once when the file is loaded
vectorstore_external_docs = Chroma.from_documents(
    documents=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20).split_documents(
        WebBaseLoader(DESSERTINO_WEB_PATHS).load()
    ),
    embedding=embd_model,
)


# --- State Definition ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    # NEW: Keys to store retrieved context
    external_context: List[str]
    conversation_context: List[str]
    router_decision: str  # Added for persistence in conditional routing


# --- Helper Function ---
def messages_to_document(messages: List[BaseMessage]) -> Document:
    history_string = ""
    for message in messages:
        role = message.type.capitalize()
        history_string += f"**{role}:** {message.content}\n\n"

    return Document(
        page_content=history_string,
        metadata={"source": "conversation_history"}
    )


# --- Agent Nodes ---

def respond_with_gk(state: AgentState) -> AgentState:
    """Handles general knowledge and non-RAG responses."""
    last_human_msg = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]
    response_content = llm.invoke(last_human_msg.content).strip()

    # Returns the dictionary required for state update
    new_messages = state["messages"] + [AIMessage(content=response_content)]
    return {"messages": new_messages}


def retrieve_info_from_conversation(state: AgentState) -> AgentState:
    """Handles retrieval from the current conversation history and stores context."""
    last_human_msg = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]

    # Create a vector store on the fly for the current conversation history
    conversation_history_doc = messages_to_document(state["messages"])
    vectorstore = Chroma.from_documents(
        documents=[conversation_history_doc],
        embedding=embd_model
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(k=2),
        llm=llm
    )

    # Retrieve documents
    docs = retriever.invoke(last_human_msg.content)

    # Store context in state
    return {"conversation_context": [doc.page_content for doc in docs]}


def retrieve_info_from_outside(state: AgentState) -> AgentState:
    """Handles retrieval from the external web document vector store and stores context."""
    last_human_msg = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]

    # Uses the pre-initialized external vector store
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore_external_docs.as_retriever(k=3),
        llm=llm
    )

    # Retrieve documents
    docs = retriever.invoke(last_human_msg.content)

    # Store context in state
    return {"external_context": [doc.page_content for doc in docs]}


def synthesize_rag_answer(state: AgentState) -> AgentState:
    """Combines retrieved contexts from both sources and generates the final answer."""
    last_human_msg = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]

    # Combine all contexts
    all_context = ""
    if state.get("external_context"):
        all_context += "\n--- External Document Context ---\n" + "\n".join(state["external_context"])
    if state.get("conversation_context"):
        all_context += "\n--- Conversation History Context ---\n" + "\n".join(state["conversation_context"])

    # If no context was retrieved (shouldn't happen if routed correctly, but safe check)
    if not all_context:
        # Fallback to General Knowledge if no context was retrieved
        return respond_with_gk(state)

    SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
        """You are a helpful assistant for Dessertino. Use ALL of the provided context to fully and accurately answer the user's question. 
        If information from both sources is relevant, combine them logically.

        TOTAL CONTEXT:
        --------------------
        {context}
        --------------------

        USER QUESTION: {question}"""
    )

    # Use LangChain to invoke the final synthesis step
    rag_chain = (
            SYNTHESIS_PROMPT
            | llm
            | StrOutputParser()
    )

    response = rag_chain.invoke({"context": all_context, "question": last_human_msg.content})

    # Returns the dictionary required for state update
    new_messages = state["messages"] + [AIMessage(content=response)]
    return {"messages": new_messages}


def router_node(state: AgentState) -> Dict:
    """Routes the request to the appropriate node based on content."""
    # Get the last message sent by the user
    last_message = state["messages"][-1]

    router_prompt = ChatPromptTemplate.from_template(
        """You are a routing LLM. Your task is to decide if the user's request is best answered by :
        retrieving information from **both** external documents and conversation (COMPLEX_RAG), retrieving only from external documents (EXTERNAL_RAG), retrieving only from conversation (CONVERSATION_RAG), or using general knowledge (RESPOND_WITH_GK).

        Use COMPLEX_RAG if the query relies on facts from the website AND specific details previously discussed.

        User Request: "{last_message_content}"

        Respond with **only** one of the following four keywords:
        - **COMPLEX_RAG**
        - **EXTERNAL_RAG**
        - **CONVERSATION_RAG**
        - **RESPOND_WITH_GK** KEYWORD:"""
    ).format_prompt(last_message_content=last_message.content)

    # Invoke the LLM to get the routing decision
    llm_response = llm.invoke(router_prompt.to_string()).strip().upper()

    # Determine the next node based on the LLM's response
    if "COMPLEX_RAG" in llm_response:
        path = "retrieve_info_from_outside"
        decision = "COMPLEX_RAG"
    elif "EXTERNAL_RAG" in llm_response:
        path = "retrieve_info_from_outside"
        decision = "EXTERNAL_RAG"
    elif "CONVERSATION_RAG" in llm_response:
        path = "retrieve_info_from_conversation"
        decision = "CONVERSATION_RAG"
    else:
        path = "respond_with_gk"
        decision = "RESPOND_WITH_GK"

    # FIX: Return the starting path under 'next_node' AND the flow decision under 'router_decision'
    # The 'router_decision' key persists in the state for later branching.
    return {"next_node": path, "router_decision": decision}