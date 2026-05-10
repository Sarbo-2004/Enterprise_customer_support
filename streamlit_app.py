"""
Streamlit Frontend
==================
Chat UI with full pipeline breakdown
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Customer Support AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Customer Support Automation")
st.caption("Powered by AutoGen + Gemini + FAISS")

# ── Session State ─────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Layout ────────────────────────────────────────────────────────────────────

chat_col, info_col = st.columns([1.2, 1])

# ── Chat Column ───────────────────────────────────────────────────────────────

with chat_col:
    st.subheader("💬 Chat")

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(msg["query"])
        with st.chat_message("assistant"):
            st.write(msg["final_response"])

    # Input
    query = st.chat_input("Type your query here...")

    if query:
        with st.spinner("Running through agents..."):
            try:
                res    = requests.post(f"{API_URL}/query", json={"query": query}, timeout=180)
                result = res.json()
                st.session_state.chat_history.append(result)
                st.session_state.last_result = result
                st.rerun()
            except Exception as e:
                st.error(f"API Error: {str(e)}")

# ── Pipeline Breakdown Column ─────────────────────────────────────────────────

with info_col:
    st.subheader("🔍 Pipeline Breakdown")

    if "last_result" in st.session_state:
        result = st.session_state.last_result

        # Agent flow visualization
        agents = [
            ("🔎", "Query Understanding",  "Detects intent, category, sentiment"),
            ("📚", "RAG Retrieval",         "Searches FAISS knowledge base"),
            ("⚖️",  "Escalation Manager",   "Routes to human or automated"),
            ("✍️",  "Response Generation",  "Generates final response"),
        ]

        for icon, name, desc in agents:
            with st.expander(f"{icon} {name}"):
                st.caption(desc)
                # Show matching agent output if available
                for agent_out in result.get("agent_outputs", []):
                    if agent_out["agent"] == name:
                        st.write(agent_out["output"])

        st.divider()
        st.markdown("**Final Response:**")
        st.info(result["final_response"])

    else:
        st.info("Send a query to see the pipeline breakdown.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📊 Session Stats")
    total = len(st.session_state.chat_history)
    st.metric("Total Queries", total)

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        if "last_result" in st.session_state:
            del st.session_state.last_result
        st.rerun()

    st.divider()
    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API: {health['status']}")
    except:
        st.error("API not reachable")