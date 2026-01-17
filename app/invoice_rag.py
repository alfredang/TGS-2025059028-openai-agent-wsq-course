import streamlit as st

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Invoice RAG Chat", 
    page_icon="📄", 
    layout="wide")

import os
from pypdf import PdfReader
from pathlib import Path
from openai import OpenAI
import chromadb
import tempfile

# Get API key from Streamlit secrets (for cloud) or environment variable (for local)
api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is not set. Please add it to your Streamlit secrets or environment variables.")
    st.stop()

# Initialize OpenAI client
openai_client = OpenAI(api_key=api_key)

# Initialize ChromaDB - use persistent client to maintain data across reruns
@st.cache_resource
def get_chroma_collection():
    chroma = chromadb.Client()
    # Get or create collection
    collection = chroma.get_or_create_collection(name="invoice_rag_kb")
    return chroma, collection

chroma, collection = get_chroma_collection()

def load_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(f"[Page {i+1}]\n{text}")
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def index_pdf(pdf_file, filename: str) -> int:
    """Index a single PDF file into ChromaDB. Returns number of chunks indexed."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    try:
        # Extract and chunk text
        text = load_pdf_text(tmp_path)
        chunks = chunk_text(text)

        if not chunks:
            return 0

        all_chunks = []
        metadatas = []
        ids = []

        # Create unique stem from filename
        stem = Path(filename).stem

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                all_chunks.append(chunk)
                metadatas.append({
                    "source": filename,
                    "chunk": i
                })
                ids.append(f"{stem}_{i}")

        if not all_chunks:
            return 0

        # Generate embeddings
        embeddings = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=all_chunks
        ).data

        # Add to collection
        collection.add(
            documents=all_chunks,
            metadatas=metadatas,
            embeddings=[e.embedding for e in embeddings],
            ids=ids
        )

        return len(all_chunks)

    finally:
        # Clean up temp file
        os.unlink(tmp_path)

def retrieve_context(question: str) -> str:
    """Retrieve relevant context from ChromaDB for a question."""
    # Check if collection has any documents
    if collection.count() == 0:
        return "No documents have been indexed yet. Please upload some invoices first."

    q_embedding = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=question
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=4
    )

    if not results["documents"][0]:
        return "No relevant context found."

    return "\n\n".join(results["documents"][0])


def clear_collection():
    """Clear all documents from the collection."""
    global chroma, collection
    chroma.delete_collection(name="invoice_rag_kb")
    collection = chroma.get_or_create_collection(name="invoice_rag_kb")
    # Update the cached resource
    st.cache_resource.clear()


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Invoices")
    st.write("Upload invoice PDFs to query")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s)")

        if st.button("📤 Upload & Index", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_chunks = 0
            indexed_count = 0

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing: {file.name}")
                try:
                    chunks = index_pdf(file, file.name)
                    total_chunks += chunks
                    indexed_count += 1
                    st.session_state.indexed_files.append(file.name)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("")
            st.success(f"✅ Indexed {indexed_count} file(s) with {total_chunks} chunks")

    st.divider()

    # Show indexed files
    st.subheader("Indexed Files")
    doc_count = collection.count()
    st.write(f"Total chunks in database: {doc_count}")

    if st.session_state.indexed_files:
        for f in st.session_state.indexed_files:
            st.write(f"• {f}")

    st.divider()

    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear DB"):
            clear_collection()
            st.session_state.indexed_files = []
            st.success("Database cleared!")
            st.rerun()
    with col2:
        if st.button("🔄 New Chat"):
            st.session_state.messages = []
            st.rerun()

# Main chat interface
st.title("📄Invoice RAG Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your invoices..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response using RAG
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve context
            context = retrieve_context(prompt)

            # Create augmented prompt
            augmented_prompt = f"""Based on the following invoice context, answer the user's question.
If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {prompt}

Answer:"""

            # Get response from OpenAI
            response = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about invoices. Use the provided context to answer accurately. If the information is not in the context, say so."},
                    {"role": "user", "content": augmented_prompt}
                ]
            )

            assistant_response = response.choices[0].message.content
            st.write(assistant_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
