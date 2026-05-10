# %%
from datasets import load_dataset

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# %%
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from sentence_transformers import SentenceTransformer

# %%
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# %%
df = pd.read_csv("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv", encoding="utf-8")

# %%
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

# Build proper Documents with metadata (no splitter needed)
docs = [
    Document(
        page_content=row["instruction"],
        metadata={
            "intent":    row["intent"],
            "category":  row["category"],
            "response":  row["response"],
            "flags":     row["flags"]
        }
    )
    for _, row in df.iterrows()
]

# Build FAISS directly — metadata stays attached
faiss_store = FAISS.from_documents(docs, hf_embeddings)
faiss_store.save_local("faiss_index")

# %%
faiss_store = FAISS.load_local("faiss_index", hf_embeddings, allow_dangerous_deserialization=True)

# %%
retriever = faiss_store.as_retriever(search_kwargs={"k": 3})

# %%
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# %%
import autogen

# %%
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# %%
config_list = [{
    "model":    "gemini-flash-lite-latest",
    "api_key":  os.environ["GOOGLE_API_KEY"],
    "api_type": "google"
}]

llm_config = {
    "config_list": config_list,
    "temperature": 0.2,
    "cache_seed":  None   # disable caching for fresh responses
}

# %%
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

# %%
def validate_retrieval(query: str, context: str) -> str:
    """Use Gemini to semantically validate retrieval relevance."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Customer Query: {query}
    Retrieved Context: {context}

    Is this context relevant and sufficient to answer the query?
    Reply ONLY with one of:
    - valid: <one line reason>
    - invalid: <one line reason>
    """

    response = model.generate_content(prompt)
    return response.text.strip()

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
def run_pipeline(customer_query: str) -> str:

    groupchat = autogen.GroupChat(
        agents=[user_proxy, query_agent, retrieval_agent, escalation_agent, response_agent],
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin"  # sequential like CrewAI
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )

    user_proxy.initiate_chat(
        manager,
        message=f"Customer query: {customer_query}"
    )

    # Extract final response from SupportResponder
    for msg in reversed(groupchat.messages):
        if msg.get("name") == "SupportResponder":
            return msg.get("content", "")

    return "No response generated"

# %%
if __name__ == "__main__":
    test_queries = [
        "I want to cancel my order immediately",
        "I was charged twice and no one is helping me",
        "Where is my package? It has been 2 weeks",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Customer: {query}")
        print('='*60)
        response = run_pipeline(query)
        print(f"\nFinal Response:\n{response}")

# %%
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download("punkt")

# %%
import google.generativeai as genai

# %%
gemini    = genai.GenerativeModel("gemini-flash-lite-latest")
sem_model = SentenceTransformer("all-MiniLM-L6-v2")
scorer    = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
smoother  = SmoothingFunction().method1

# %%
test_set = [
    {
        "query":        "I want to cancel my order",
        "ground_truth": "I've understood you have a question regarding canceling order, I'll do my best to help you with this."
    },
    {
        "query":        "Where is my package?",
        "ground_truth": "I understand you're inquiring about the status of your delivery, I'll do my best to assist you."
    },
    {
        "query":        "I was charged twice for the same item",
        "ground_truth": "I understand the frustration that arises when you notice an unexpected charge."
    },
    {
        "query":        "How do I reset my account password?",
        "ground_truth": "I'll assist you with the account password reset process right away."
    },
    {
        "query":        "I want a refund for my order",
        "ground_truth": "I've understood that you're seeking a refund, I'll do my best to help you with this."
    },
]

# %%
def bleu_score(reference: str, hypothesis: str) -> float:
    ref   = [reference.lower().split()]
    hyp   = hypothesis.lower().split()
    return round(sentence_bleu(ref, hyp, smoothing_function=smoother), 4)


# %%
def rouge_scores(reference: str, hypothesis: str) -> dict:
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }

# %%
def relevance_score(query: str, response: str) -> float:
    """Semantic similarity between query and response using embeddings."""
    q_emb = sem_model.encode(query,    convert_to_tensor=True)
    r_emb = sem_model.encode(response, convert_to_tensor=True)
    return round(float(util.cos_sim(q_emb, r_emb)), 4)

# %%
def baseline_rag(query: str) -> str:
    """Simple RAG: retrieve → generate. No agents."""
    docs    = faiss_store.similarity_search(query, k=3)
    context = "\n".join([doc.metadata.get("response", "") for doc in docs])

    prompt = f"""
    Answer this customer query using only the context below.
    Query  : {query}
    Context: {context}
    Answer :
    """
    response = gemini.generate_content(prompt)
    return response.text.strip()

# %%
def agentic_rag(query: str) -> str:
    """Full agentic pipeline response."""
    # from pipeline import run_pipeline   # ← your AutoGen pipeline
    return run_pipeline(query)

# %%
def evaluate(query: str, ground_truth: str, generated: str) -> dict:
    return {
        "query":        query,
        "generated":    generated,
        "ground_truth": ground_truth,
        "bleu":         bleu_score(ground_truth, generated),
        **rouge_scores(ground_truth, generated),
        "relevance":    relevance_score(query, generated),
    }

# %%
def run_evaluation():
    baseline_results = []
    agentic_results  = []

    for item in test_set:
        query  = item["query"]
        gt     = item["ground_truth"]

        print(f"\nEvaluating: {query}")

        # Baseline
        baseline_response = baseline_rag(query)
        baseline_results.append(evaluate(query, gt, baseline_response))

        # Agentic
        agentic_response = agentic_rag(query)
        agentic_results.append(evaluate(query, gt, agentic_response))

# %%
baseline_results = []
agentic_results  = []

for item in test_set:
    query = item["query"]
    gt    = item["ground_truth"]

    print(f"\nEvaluating: {query}")

    # Baseline
    baseline_response = baseline_rag(query)
    baseline_results.append(evaluate(query, gt, baseline_response))

    # Agentic (skip for now if pipeline not ready)
    agentic_response = agentic_rag(query)
    agentic_results.append(evaluate(query, gt, agentic_response))

# Now create DataFrames
df_baseline = pd.DataFrame(baseline_results)
df_agentic  = pd.DataFrame(agentic_results)

print(df_baseline)
print(df_agentic)

# %%
metrics = ["bleu", "rouge1", "rouge2", "rougeL", "relevance"]

print("\n" + "="*60)
print("EVALUATION RESULTS — BASELINE vs AGENTIC RAG")
print("="*60)
comparison = {}
for metric in metrics:
    b = df_baseline[metric].mean()
    a = df_agentic[metric].mean()
    comparison[metric] = {
        "baseline": round(b, 4),
        "agentic":  round(a, 4),
        "improvement": f"{round((a - b) / (b + 1e-9) * 100, 2)}%"
    }
    print(f"\n{metric.upper()}")
    print(f"  Baseline : {b:.4f}")
    print(f"  Agentic  : {a:.4f}")
    print(f"  Improvement: {comparison[metric]['improvement']}")

# %%



