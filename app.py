
import os
import streamlit as st

from src.pdf_processor import extract_pdf_pages, chunk_pages
from src.vector_store import load_embedding_model, create_vector_index, search_documents
from src.rag_engine import generate_answer


st.set_page_config(
    page_title="CampusGPT",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 CampusGPT")
st.caption("AI-powered college document assistant")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.warning("GEMINI_API_KEY environment variable is missing.")
    st.stop()

# Load embedding model once
@st.cache_resource
def get_embedding_model():
    return load_embedding_model()

embedding_model = get_embedding_model()

# Session state
if "documents" not in st.session_state:
    st.session_state.documents = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("College Documents")

    college_name = st.text_input(
        "College Name",
        placeholder="Brainware University"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary"):
        if not college_name.strip():
            st.error("Enter a college name.")
        elif not uploaded_files:
            st.error("Upload at least one PDF.")
        else:
            all_chunks = []

            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    pages = extract_pdf_pages(
                        file.getvalue(),
                        file.name
                    )

                    chunks = chunk_pages(pages)

                    for chunk in chunks:
                        chunk["college"] = college_name.strip()

                    all_chunks.extend(chunks)

                if not all_chunks:
                    st.error("No readable text found in PDFs.")
                else:
                    index = create_vector_index(
                        all_chunks,
                        embedding_model
                    )

                    st.session_state.documents[college_name.strip()] = {
                        "chunks": all_chunks,
                        "index": index
                    }

                    st.success(
                        f"Processed {len(all_chunks)} chunks!"
                    )

    st.divider()

    selected_college = st.selectbox(
        "Select College",
        options=list(st.session_state.documents.keys()),
        index=0 if st.session_state.documents else None,
        placeholder="Choose a processed college"
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat
st.subheader("Ask CampusGPT")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(
                        f"{source['filename']} — "
                        f"Page {source['page']}"
                    )

question = st.chat_input(
    "Ask about syllabus, exams, attendance or notices..."
)

if question:
    if not selected_college:
        st.warning("Upload and process college documents first.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            document = st.session_state.documents[selected_college]

            retrieved = search_documents(
                question,
                document["index"],
                document["chunks"],
                embedding_model,
                top_k=4
            )

            if not retrieved:
                answer = "I could not find relevant information."
            else:
                answer = generate_answer(
                    api_key,
                    question,
                    retrieved
                )

            st.markdown(answer)

            with st.expander("View Sources"):
                for source in retrieved:
                    st.write(
                        f"📄 {source['filename']} — "
                        f"Page {source['page']}"
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved
    })
