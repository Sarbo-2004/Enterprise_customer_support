import os
import autogen
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

load_dotenv()

# =========================================================
# SETUP
# =========================================================

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

faiss_store = FAISS.load_local(
    "faiss_index",
    hf_embeddings,
    allow_dangerous_deserialization=True
)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

config_list = [{
    "model": "gemini-flash-lite-latest",
    "api_key": os.environ["GOOGLE_API_KEY"],
    "api_type": "google"
}]

llm_config = {
    "config_list": config_list,
    "temperature": 0.2,
    "cache_seed": None
}

# =========================================================
# TOOLS
# =========================================================

def rag_search(query: str, k: int = 3) -> str:
    docs = faiss_store.similarity_search(query, k=k)
    results = []
    for i, doc in enumerate(docs):
        results.append(
            f"[{i+1}] {doc.page_content}\n{doc.metadata.get('response','')}"
        )
    return "\n\n".join(results)


def validate_retrieval(query: str, context: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Query: {query}
    Context: {context}

    Reply ONLY:
    valid OR invalid
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# =========================================================
# AGENTS
# =========================================================

query_agent = autogen.AssistantAgent(
    name="QueryAnalyst",
    system_message="Analyze query and rephrase it.",
    llm_config=llm_config
)

retrieval_agent = autogen.AssistantAgent(
    name="KnowledgeRetriever",
    system_message="Use rag_search and summarize answer.",
    llm_config=llm_config
)

escalation_agent = autogen.AssistantAgent(
    name="EscalationManager",
    system_message="Decide if escalation needed."
)

response_agent = autogen.AssistantAgent(
    name="SupportResponder",
    system_message="""
    You are FINAL agent.

    Generate final answer for customer.
    Do not continue conversation.
    End with TERMINATE.
    """,
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="CustomerProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config=False,
    function_map={
        "rag_search": rag_search,
        "validate_retrieval": validate_retrieval
    }
)

# =========================================================
# PIPELINE
# =========================================================

def run_pipeline(customer_query: str) -> str:

    groupchat = autogen.GroupChat(
        agents=[
            user_proxy,
            query_agent,
            retrieval_agent,
            escalation_agent,
            response_agent
        ],
        messages=[],
        max_round=6,
        speaker_selection_method="auto",   # ✅ FIX
        allow_repeat_speaker=False
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )

    user_proxy.initiate_chat(
        manager,
        message=f"Customer query: {customer_query}"
    )

    # ✅ extract response safely
    responses = [
        msg.get("content")
        for msg in groupchat.messages
        if msg.get("name") == "SupportResponder"
        and msg.get("content")
    ]

    if responses:
        return responses[-1].replace("TERMINATE", "").strip()

    return "No response generated"
