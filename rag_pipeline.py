import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import pandas as pd
import autogen
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
load_dotenv()
 

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
config_list = [{
    "model":    "gemini-2.5-flash",              # ← fix 1
    "api_key":  os.environ.get("GOOGLE_API_KEY"), # ← fix 2
    "api_type": "google"
}]

llm_config = {
    "config_list": config_list,
    "temperature": 0.2,
    "cache_seed":  None   # disable caching for fresh responses
}
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)
faiss_store = FAISS.load_local("faiss_index", hf_embeddings, allow_dangerous_deserialization=True)
def rag_search(query: str, k: int = 3) -> str:
    """Search FAISS knowledge base and return formatted results."""
    docs = faiss_store.similarity_search(query, k=k)
    results = []
    for i, doc in enumerate(docs):
        results.append(
            f"[{i+1}] Intent  : {doc.metadata.get('intent', 'N/A')}\n"
            f"     Category: {doc.metadata.get('category', 'N/A')}\n"
            f"     Q: {doc.page_content}\n"
            f"     A: {doc.metadata.get('response', 'N/A')}"
        )
    return "\n\n".join(results)

 
def validate_retrieval(query: str, context: str) -> str:
    prompt = f"""
    Customer Query: {query}
    Retrieved Context: {context}

    Is this context relevant and sufficient to answer the query?
    Reply ONLY with one of:
    - valid: <one line reason>
    - invalid: <one line reason>
    """
    response = gemini_llm.invoke(prompt)
    return response.content.strip()

 
tools = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the customer support knowledge base for relevant answers",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "k":     {"type": "integer", "description": "Number of results", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_retrieval",
            "description": "Validate if retrieved context is relevant to the query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":   {"type": "string"},
                    "context": {"type": "string"}
                },
                "required": ["query", "context"]
            }
        }
    }
]

llm_config_with_tools = {**llm_config, "functions": tools}

 
query_agent = autogen.AssistantAgent(
    name="QueryAnalyst",
    system_message="""You are a Query Understanding Analyst.
    Analyze customer queries and extract:
    - intent     (e.g. cancel_order, track_order, get_refund)
    - category   (e.g. ORDER, SHIPPING, BILLING, ACCOUNT)
    - sentiment  (positive / neutral / negative)
    - escalate   (true / false)
    - confidence (high / medium / low)
    - rephrased_query (clean version)

    Use rag_search to find similar queries first.
    Always output a structured summary before passing to next agent.
    """,
    llm_config=llm_config_with_tools
)

 
retrieval_agent = autogen.AssistantAgent(
    name="KnowledgeRetriever",
    system_message="""You are a Knowledge Retrieval Specialist.
    Your job:
    1. Use rag_search with the rephrased query from QueryAnalyst
    2. Use validate_retrieval to confirm relevance
    3. If invalid, retry rag_search with a different query
    4. Summarize the best answer found

    Always validate before passing answer to next agent.
    Output: validated answer + source intents used.
    """,
    llm_config=llm_config_with_tools
)

 
escalation_agent = autogen.AssistantAgent(
    name="EscalationManager",
    system_message="""You are an Escalation Manager.
    Apply these rules to decide routing:

    ESCALATE if any of:
    - sentiment is negative
    - intent is get_refund, complaint, payment_issue
    - confidence is low
    - escalate flag is true

    Output:
    - route: human_agent OR automated_response
    - priority: high (2+ reasons) | medium (1 reason) | low (0 reasons)
    - reasons: list of escalation reasons
    """,
    llm_config=llm_config
)

 
response_agent = autogen.AssistantAgent(
    name="SupportResponder",
    system_message="""You are a Senior Customer Support Responder.

    If route = human_agent:
      Write a warm empathetic handoff message (max 3 sentences).
      Tell customer a specialist will help them shortly.

    If route = automated_response:
      Polish the retrieved answer into a professional response.
      Be warm, concise, and resolve the customer's concern directly.
      Never mention AI, knowledge base, or context.

    End with TERMINATE when response is ready.
    """,
    llm_config=llm_config
)

 
user_proxy = autogen.UserProxyAgent(
    name="CustomerProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    function_map={
        "rag_search":         rag_search,
        "validate_retrieval": validate_retrieval
    },
    code_execution_config=False
)

 
def run_pipeline(customer_query: str) -> dict:

    groupchat = autogen.GroupChat(
        agents=[user_proxy, query_agent, retrieval_agent, escalation_agent, response_agent],
        messages=[],
        max_round=20,
        speaker_selection_method="round_robin"
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )

    user_proxy.initiate_chat(
        manager,
        message=f"Customer query: {customer_query}"
    )

    agent_name_map = {
        "QueryAnalyst":       "Query Understanding",
        "KnowledgeRetriever": "RAG Retrieval",
        "EscalationManager":  "Escalation",
        "SupportResponder":   "Response Generation"
    }

    agent_outputs  = []
    final_response = ""

    for msg in groupchat.messages:
        name    = msg.get("name", "")
        content = msg.get("content", "") or ""

        if name in agent_name_map:
            agent_outputs.append({
                "agent":  agent_name_map[name],
                "output": content
            })

        if name == "SupportResponder":
            final_response = content.replace("TERMINATE", "").strip()

    # ── Fallback: use last non-empty message if SupportResponder didn't reply
    if not final_response:
        for msg in reversed(groupchat.messages):
            content = msg.get("content", "") or ""
            name    = msg.get("name", "")
            if content and name != "CustomerProxy":
                final_response = content.replace("TERMINATE", "").strip()
                break

    return {
        "final_response": final_response,
        "agent_outputs":  agent_outputs
    }