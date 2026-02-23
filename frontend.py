import streamlit as st
import requests

# Set page configuration
st.set_page_config(page_title="Security AI Chatbot", page_icon="🛡️")

st.title("🛡️ Security RAG Chatbot")

# Sidebar to choose the model
model_choice = st.sidebar.radio(
    "Select Knowledge Base:",
    ("OWASP", "MITRE ATT&CK")
)

st.sidebar.markdown("---")
st.sidebar.write("This bot retrieves answers using RAG (Retrieval-Augmented Generation).")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about a vulnerability or threat..."):
    # 1. Display user message in chat message container
    st.chat_message("user").write(prompt)
    
    # 2. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Determine which API endpoint to call
    api_url = "http://127.0.0.1:8000" # Ensure this matches your FastAPI port
    if model_choice == "OWASP":
        endpoint = f"{api_url}/askowasp"
    else:
        endpoint = f"{api_url}/askmitre"

    # 4. Call the FastAPI Backend
    try:
        with st.spinner(f"Consulting {model_choice} knowledge base..."):
            response = requests.post(endpoint, json={"query": prompt})
            
        if response.status_code == 200:
            data = response.json()
            bot_answer = data.get("answer", "No answer received.")
        else:
            bot_answer = f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.ConnectionError:
        bot_answer = "Error: Could not connect to the backend. Is FastAPI running?"

    # 5. Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(bot_answer)

    # 6. Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})