import streamlit as st
import requests

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Financial AI Chatbot",
    page_icon="💬",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("Financial Market AI Chatbot")
st.caption("Agentic Financial Market Intelligence Assistant")

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
# MODE SELECTION (✅ NEW FEATURE)
# =========================================================

mode = st.radio(
    "Response Type",
    ["detailed", "simple"],
    index=0,              # default = detailed
    horizontal=True
)

# =========================================================
# BASIC CHATBOT RESPONSES
# =========================================================

def basic_chat_response(prompt):
    prompt = prompt.lower()

    greetings = [
        "hi","hello","hey","hii",
        "good morning","good evening",
        "holla","namaste"
    ]

    if prompt in greetings:
        return """
Hello 👋

I'm your Financial AI Assistant.

You can ask me about:

- Apple financial performance
- Investment opportunities
- Market risks
- Revenue growth
- Financial forecasting
- Portfolio insights
"""

    elif "who are you" in prompt:
        return """
I'm an AI-powered Financial Market Intelligence Assistant built using:

- LangGraph  
- Groq LLM  
- FAISS Vector Search  
- Multi-Agent Financial Reasoning  
"""

    elif "thank you" in prompt or "thanks" in prompt:
        return "You're welcome 😊"

    return None

# =========================================================
# FASTAPI CALL FUNCTION
# =========================================================

API_URL = "http://57.162.107.9:8080/analyze"

def call_fastapi(query, mode):
    try:
        payload = {
            "query": query,
            "mode": mode
        }

        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()

        # ✅ FIX: convert escaped newline → real newline
        raw_answer = data.get("answer", "No response")
        formatted_answer = raw_answer.replace("\\n", "\n")

        return {
            "answer": formatted_answer,
            "latency": data.get("latency", "N/A")
        }

    except Exception as e:
        return {
            "answer": f"❌ API Error: {str(e)}",
            "latency": "N/A"
        }

# =========================================================
# CHAT INPUT
# =========================================================

prompt = st.chat_input(
    "Ask about financial markets, Apple stock, investments, risks..."
)

# =========================================================
# USER MESSAGE
# =========================================================

if prompt:

    # Show user message
    st.chat_message("user").markdown(prompt)

    # Save message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # =====================================================
    # BASIC RESPONSE
    # =====================================================

    basic_response = basic_chat_response(prompt)

    if basic_response:
        with st.chat_message("assistant"):
            st.markdown(basic_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": basic_response
        })

    # =====================================================
    # FASTAPI CALL
    # =====================================================

    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing financial data..."):

                # ✅ Use selected mode (user controlled)
                result = call_fastapi(prompt, mode)

                # ✅ Proper formatted display
                st.markdown(result["answer"])
                st.caption(f"Response Time: {result['latency']} sec")

                # Save response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"]
                })