import streamlit as st
import requests

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Customer Support AI",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🤖 Customer Support Automation")
st.caption("Agentic Customer Support Assistant (AutoGen + RAG + FAISS)")

# =========================================================
# CLEAR CHAT
# =========================================================

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# API CONFIG
# =========================================================

API_URL = "http://57.162.107.9:8000/query"

# =========================================================
# API CALL FUNCTION (SIMPLE & STABLE)
# =========================================================

def call_fastapi(query):
    try:
        payload = {
            "query": query,
            "session_id": "default"
        }

        response = requests.post(
            API_URL,
            json=payload,
            timeout=120   # ✅ simple timeout (no complex headers)
        )

        response.raise_for_status()

        data = response.json()

        # ✅ Extract response cleanly
        answer = data.get("final_response", "No response")

        return answer

    except Exception as e:
        return f"❌ API Error: {str(e)}"

# =========================================================
# CHAT INPUT
# =========================================================

prompt = st.chat_input("Ask anything about orders, refunds, issues...")

# =========================================================
# USER MESSAGE
# =========================================================

if prompt:

    # Show user message
    st.chat_message("user").markdown(prompt)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # =====================================================
    # FASTAPI CALL
    # =====================================================

    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):

            result = call_fastapi(prompt)

            # ✅ Display response
            st.markdown(result)

            # Save response
            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })
