"""
Streamlit Frontend
==================
Chat UI with full pipeline breakdown
"""

import streamlit as st
import requests

API_URL = "http://57.162.107.9:8000"

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

    # Display history safely
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(msg.get("query", ""))
        with st.chat_message("assistant"):
            st.write(msg.get("final_response", "No response generated"))

    # Input
    query = st.chat_input("Type your query here...")

    if query:
        with st.spinner("Running through agents..."):
            try:
                res    = requests.post(f"{API_URL}/query", json={"query": query}, timeout=180)
                result = res.json()

                # Debug — remove after confirming keys are correct
                # st.write("Raw API response:", result)

                # Safely attach query
                result["query"]          = result.get("query", query)
                result["final_response"] = result.get("final_response", "No response generated")
                result["agent_outputs"]  = result.get("agent_outputs", [])

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

        agents = [
            ("🔎", "Query Understanding", "Detects intent, category, sentiment"),
            ("📚", "RAG Retrieval",        "Searches FAISS knowledge base"),
            ("⚖️", "Escalation",           "Routes to human or automated"),
            ("✍️", "Response Generation",  "Generates final response"),
        ]

        for icon, name, desc in agents:
            with st.expander(f"{icon} {name}"):
                st.caption(desc)
                matched = False
                for agent_out in result.get("agent_outputs", []):
                    if agent_out.get("agent") == name:
                        st.write(agent_out.get("output", ""))
                        matched = True
                if not matched:
                    st.info("No output captured for this agent.")

        st.divider()
        st.markdown("**Final Response:**")
        st.info(result.get("final_response", "No response generated"))

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
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API: {health.get('status', 'unknown')}")
    except:
        st.error("API not reachable")