import streamlit as st
import ollama
import os
import fitz  # pymupdf

# Retrieve OLLAMA_HOST and MODEL_ID from environment variables
ollama_host = os.environ.get("OLLAMA_HOST")
model_id = os.environ.get("MODEL_ID")

if not ollama_host:
    st.error("OLLAMA_HOST environment variable not set.")
    st.stop()
if not model_id:
    st.error("MODEL_ID environment variable not set.")
    st.stop()

# Initialize Ollama client
client = ollama.Client(host=ollama_host)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}
if "token_stats" not in st.session_state:
    st.session_state.token_stats = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_id
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""


@st.cache_data(ttl=60, show_spinner=False)
def fetch_available_models(host):
    """Fetch models currently available on the Ollama server. Cached for 60s."""
    try:
        c = ollama.Client(host=host)
        result = c.list()
        return sorted([m.model for m in result.models])
    except Exception:
        return []


def extract_pdf_text(uploaded_file):
    """Extract all text from every page using PyMuPDF. Returns (text, page_count, extracted_pages)."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    pages_text = []
    extracted_pages = 0
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text and text.strip():
            pages_text.append(f"[Page {page_num} of {total_pages}]\n{text.strip()}")
            extracted_pages += 1
    doc.close()
    return "\n\n".join(pages_text), total_pages, extracted_pages


def estimate_tokens(char_count: int) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, char_count // 4)


def build_context_char_count() -> int:
    """Total characters across all loaded files + chat history."""
    total = 0
    for msg in st.session_state.messages:
        total += len(msg.get("content", ""))
    for data in st.session_state.processed_files.values():
        if data["type"] == "pdf":
            total += len(data["text"])
    return total


def required_num_ctx(extra_chars: int = 0) -> int:
    """Calculate num_ctx for Ollama so the full content fits, capped at 128K."""
    total_chars = build_context_char_count() + extra_chars
    needed = estimate_tokens(total_chars) + 2048
    ctx = 4096
    while ctx < needed:
        ctx *= 2
    return min(ctx, 131072)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Model selector
    st.header("Model")
    available_models = fetch_available_models(ollama_host)
    if available_models:
        default_idx = (
            available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models else 0
        )
        chosen = st.selectbox("Active model", available_models, index=default_idx,
                              help="Switch instantly — no restart needed")
        if chosen != st.session_state.selected_model:
            st.session_state.selected_model = chosen
            st.rerun()
    else:
        st.warning("Could not reach Ollama to list models.")
        st.session_state.selected_model = model_id

    st.markdown("---")
    # System prompt
    st.header("System Prompt")
    new_prompt = st.text_area(
        "Instructions for the model",
        value=st.session_state.system_prompt,
        height=120,
        placeholder="e.g. You are a concise legal assistant. Always cite page numbers.",
        help="Sent as a hidden system message before every conversation.",
    )
    if new_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = new_prompt
    st.caption("✅ System prompt active" if st.session_state.system_prompt else "No system prompt set")

    st.markdown("---")
    # File uploader
    st.header("Uploaded Files")
    uploaded_files = st.file_uploader(
        "Upload images or PDFs",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True,
    )

    current_names = set()
    for f in uploaded_files:
        current_names.add(f.name)
        if f.name not in st.session_state.processed_files:
            if f.type == "application/pdf":
                with st.spinner(f"Reading {f.name}..."):
                    text, total_pages, extracted_pages = extract_pdf_text(f)
                if text:
                    st.session_state.processed_files[f.name] = {
                        "type": "pdf", "text": text,
                        "total_pages": total_pages, "extracted_pages": extracted_pages,
                    }
                    if extracted_pages < total_pages:
                        st.warning(f"**{f.name}**: {extracted_pages}/{total_pages} pages extractable.")
                else:
                    st.warning(f"Could not extract text from **{f.name}**.")
            else:
                file_path = os.path.join("/tmp", f.name)
                with open(file_path, "wb") as out:
                    out.write(f.getbuffer())
                st.session_state.processed_files[f.name] = {"type": "image", "path": file_path}

    removed = [k for k in st.session_state.processed_files if k not in current_names]
    for k in removed:
        del st.session_state.processed_files[k]

    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("**Files ready as context:**")
        for name, data in st.session_state.processed_files.items():
            if data["type"] == "pdf":
                ep, tp = data["extracted_pages"], data["total_pages"]
                status = "✅" if ep == tp else "⚠️"
                st.markdown(f"{status} 📄 **{name}**  \nPages: {ep}/{tp} — {len(data['text']):,} chars")
            else:
                st.image(data["path"], caption=name, width=140)
    else:
        st.info("No files loaded yet. Upload files above and then ask a question in the chat.")

    # Token usage
    st.markdown("---")
    st.markdown("### Token Usage")
    estimated = estimate_tokens(build_context_char_count())
    ctx_size = required_num_ctx()
    st.metric("Context estimate", f"~{estimated:,} tokens", help="4 chars ≈ 1 token")
    st.caption(f"Context window will be set to **{ctx_size:,}** tokens for the next request.")
    if st.session_state.token_stats:
        ts = st.session_state.token_stats
        col1, col2 = st.columns(2)
        col1.metric("Prompt tokens", f"{ts['prompt']:,}")
        col2.metric("Reply tokens", f"{ts['response']:,}")
        st.caption(f"Total last request: **{ts['prompt'] + ts['response']:,}** tokens")
    else:
        st.caption("Actual counts will appear after your first message.")

    st.markdown("---")
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.session_state.token_stats = None
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title(f"Ollama Chat — {st.session_state.selected_model}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message and message["images"]:
            for img_path in message["images"]:
                if os.path.exists(img_path):
                    st.image(img_path, caption="Uploaded Image", width=200)

prompt = st.chat_input("Ask a question about your files or anything else...")

if prompt:
    pdf_contexts = []
    image_paths = []

    for name, data in st.session_state.processed_files.items():
        if data["type"] == "pdf":
            pdf_contexts.append(f"[PDF: {name}]\n{data['text']}")
        elif data["type"] == "image":
            image_paths.append(data["path"])

    combined_content = (
        "\n\n---\n\n".join(pdf_contexts) + "\n\n---\n\n" + prompt
        if pdf_contexts else prompt
    )

    user_message = {"role": "user", "content": combined_content}
    if image_paths:
        user_message["images"] = image_paths

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)
        for img_path in image_paths:
            st.image(img_path, caption="Uploaded image", width=200)

    # Build API messages — prepend system prompt if set
    api_messages = []
    if st.session_state.system_prompt.strip():
        api_messages.append({"role": "system", "content": st.session_state.system_prompt.strip()})
    for msg in st.session_state.messages:
        entry = {"role": msg["role"], "content": msg["content"]}
        if "images" in msg and msg["images"]:
            entry["images"] = msg["images"]
        api_messages.append(entry)

    num_ctx = required_num_ctx(extra_chars=len(combined_content))
    _token_capture = {}

    def stream_response():
        for chunk in client.chat(
            model=st.session_state.selected_model,
            messages=api_messages,
            stream=True,
            options={"num_ctx": num_ctx},
        ):
            yield chunk["message"]["content"]
            if chunk.get("done"):
                _token_capture["prompt"] = chunk.get("prompt_eval_count", 0)
                _token_capture["response"] = chunk.get("eval_count", 0)

    with st.chat_message("assistant"):
        reply = st.write_stream(stream_response())

    if _token_capture:
        st.session_state.token_stats = _token_capture

    st.session_state.messages.append({"role": "assistant", "content": reply})
