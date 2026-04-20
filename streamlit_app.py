import streamlit as st
import ollama
import os

# Retrieve OLLAMA_HOST and MODEL_ID from environment variables
ollama_host = os.environ.get("OLLAMA_HOST")
model_id = os.environ.get("MODEL_ID")

# Ensure the ollama_host ends with /api for correct client communication
if ollama_host and not ollama_host.endswith('/api'):
    ollama_host += '/api'

# Display the resolved Ollama Host in the Streamlit app for debugging
st.write(f"Ollama Host: {ollama_host}")

if not ollama_host:
    st.error("OLLAMA_HOST environment variable not set.")
    st.stop()
if not model_id:
    st.error("MODEL_ID environment variable not set.")
    st.stop()

# Initialize Ollama client
client = ollama.Client(host=ollama_host)

# Use st.session_state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title(f"Ollama Chat with {model_id}")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "content" in message:
            st.markdown(message["content"])
        if "images" in message and message["images"]:
            for img_path in message["images"]:
                if os.path.exists(img_path):
                    st.image(img_path, caption="Uploaded Image", width=200)


# Handle user input
prompt = st.chat_input("Ask me a question...")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

current_user_message = {"role": "user", "content": prompt or ""}
image_paths = []

if uploaded_file is not None:
    file_path = os.path.join("/tmp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    image_paths.append(file_path)
    current_user_message["images"] = image_paths

    st.sidebar.image(uploaded_file, caption="Recently uploaded image", width=150)


if prompt or uploaded_file is not None:
    st.session_state.messages.append(current_user_message)
    with st.chat_message("user"):
        if current_user_message["content"]:
            st.markdown(current_user_message["content"])
        if image_paths:
            for img_path in image_paths:
                st.image(img_path, caption="User provided image", width=200)

    with st.spinner("Generating response..."):
        ollama_messages_for_api = []
        for msg in st.session_state.messages:
            if "images" in msg and msg["images"]:
                ollama_messages_for_api.append({"role": msg["role"], "content": msg["content"], "images": msg["images"]})
            else:
                ollama_messages_for_api.append({"role": msg["role"], "content": msg["content"]})

        # Temporarily disable streaming for debugging the 405 error
        response_data = client.chat(model=model_id, messages=ollama_messages_for_api, stream=False)
        full_response_content = response_data['message']['content'] # Access content directly

        with st.chat_message("assistant"):
            st.markdown(full_response_content) # Display full response at once

        st.session_state.messages.append({"role": "assistant", "content": full_response_content})
