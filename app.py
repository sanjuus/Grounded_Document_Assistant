
import hashlib

import streamlit as st

from rag_pipeline import (
    load_embedding_model,
    process_documents,
    hybrid_retrieve,
    generate_answer
)


# 1. Page Configuration
st.set_page_config(
    page_title="Grounded Document Assistant",
    page_icon="📚",
    layout="wide"
)


# 2. Session State
if "processed" not in st.session_state:
    st.session_state.processed = False

if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None


def clear_processed_documents():
    st.session_state.processed = False

    for key in [
        "chunks",
        "index",
        "bm25_index",
        "document_names"
    ]:
        st.session_state.pop(key, None)


# 3. Embedding Model
@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


# 4. Page Content
st.title("📚 Grounded Document Assistant")

st.write(
    "Upload PDF documents and ask questions. "
    "Get answers grounded in your documents with source citations."
)


# 5. PDF Upload
uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# Detect changes to uploaded documents
if uploaded_files:
    current_signature = tuple(
        (
            pdf.name,
            hashlib.sha256(pdf.getvalue()).hexdigest()
        )
        for pdf in uploaded_files
    )

    if current_signature != st.session_state.upload_signature:
        clear_processed_documents()
        st.session_state.upload_signature = current_signature

else:
    if st.session_state.upload_signature is not None:
        clear_processed_documents()
        st.session_state.upload_signature = None


# 6. Process Documents
if uploaded_files:
    st.write(f"Selected PDF files: {len(uploaded_files)}")

    if st.button("Process Documents", type="primary"):
        clear_processed_documents()

        with st.spinner("Processing your documents..."):
            try:
                model = get_embedding_model()

                # Reset file pointers before reading
                for pdf in uploaded_files:
                    pdf.seek(0)

                chunks, faiss_index, bm25_index = process_documents(
                    uploaded_files,
                    model
                )

                st.session_state.chunks = chunks
                st.session_state.index = faiss_index
                st.session_state.bm25_index = bm25_index

                st.session_state.document_names = [
                    pdf.name for pdf in uploaded_files
                ]

                st.session_state.processed = True

                st.success(
                    f"Successfully processed {len(uploaded_files)} PDF(s) "
                )

            except Exception as error:
                clear_processed_documents()
                st.error(f"Document processing failed: {error}")


# 7. Ask Questions
if st.session_state.processed:

    st.divider()
    st.subheader("Ask Your Documents")

    st.caption(
        "Processed documents: "
        + ", ".join(st.session_state.document_names)
    )

    with st.form("question_form"):
        question = st.text_input(
            "Enter your question",
            placeholder="What is the main topic of these documents?"
        )

        submitted = st.form_submit_button("Get Answer")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner("Wait, cooking your answer... 👨‍🍳🔥"):
                try:
                    model = get_embedding_model()

                    retrieved_chunks = hybrid_retrieve(
                        question,
                        model,
                        st.session_state.index,
                        st.session_state.bm25_index,
                        st.session_state.chunks,
                        top_k=12
                    )

                    answer = generate_answer(
                        question,
                        retrieved_chunks
                    )

                    st.subheader("Your Answer 💡")
                    st.write(answer)


                except Exception as error:
                    st.error(
                        f"Unable to generate an answer: {error}"
                    )
