import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import requests
from pypdf import PdfReader
import io

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG with PDF - Attention is All You Need",
    page_icon="📄",
    layout="wide"
)

# Paper URL
PAPER_URL = "https://arxiv.org/pdf/1706.03762"
PAPER_TITLE = "Attention is All You Need (Transformer Paper)"

def download_pdf(url):
    """Download PDF from URL."""
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    reader = PdfReader(pdf_file)
    text_pages = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        text_pages.append({
            'page': page_num + 1,
            'text': text
        })

    return text_pages

def chunk_text(text, chunk_size=500, overlap=50):
    """ Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence or word boundary
        if end < text_length:
            # Look for sentence end
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_space = chunk.rfind(' ')

            # Use the best breaking point
            break_point = max(last_period, last_newline, last_space)
            if break_point > chunk_size * 0.5:  # Only if it's not too early
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks

@st.cache_data
def load_and_process_pdf(url):
    """Download and process PDF into chunks. Cached - runs only once."""
    # Download PDF
    pdf_file = download_pdf(url)

    # Extract text from pages
    pages = extract_text_from_pdf(pdf_file)

    # Chunk the text
    all_chunks = []
    for page_data in pages:
        page_num = page_data['page']
        text = page_data['text']

        # Chunk this page's text
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'page': page_num,
                'chunk_id': f"page_{page_num}_chunk_{i}",
                'content': chunk
            })

    return all_chunks

@st.cache_resource
def initialize_embedding_model():
    """Initialize and cache the embedding model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_resource
def build_faiss_index(_embedding_model, _chunks):
    """Build and cache the FAISS index from chunks. Cached - runs only once."""
    # Prepare chunks for embedding
    chunk_texts = [chunk["content"] for chunk in _chunks]

    # Generate embeddings for all chunks
    embeddings = _embedding_model.encode(chunk_texts, show_progress_bar=False)

    # Create FAISS index with L2 distance
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    return index, embeddings

def initialize_llm():
    """Initialize the language model."""
    api_key = os.getenv('GOOGLE_API_KEY') or st.secrets.get("GOOGLE_API_KEY", None)
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

def get_relevant_chunks(query, embedding_model, index, chunks, k=3):
    """
    Retrieves the most relevant chunks from FAISS index based on the query.
    """
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)

    relevant_chunks = []
    for idx, distance in zip(indices[0], distances[0]):
        chunk = chunks[idx].copy()
        chunk['distance'] = float(distance)
        relevant_chunks.append(chunk)

    return relevant_chunks

def generate_prompt(query, embedding_model, index, chunks):
    """
    Generates a structured prompt for the LLM based on the user's query and retrieved context.
    """
    relevant_chunks = get_relevant_chunks(query, embedding_model, index, chunks, k=3)

    context_parts = []
    for i, chunk in enumerate(relevant_chunks, 1):
        source = f"""[Source {i} - Page {chunk['page']}] {chunk['content']}"""
        context_parts.append(source)

    context = "\n".join(context_parts)

    if not context.strip():
        return None, relevant_chunks

    system_prompt = f"""You are a helpful AI assistant that answers questions based on the "Attention is All You Need" paper (the Transformer architecture paper).

                    **Instructions:**
                    1. Answer the user's question using ONLY the information in the Context section below.
                    2. Be concise and informative in your responses.
                    3. If the answer is not in the context, respond with: "I don't have enough information to answer that question based on the provided context."
                    4. When using information from the context, cite the source and page number (e.g., [Source 1 - Page 3]).
                    5. Do not make up or assume information not present in the context.
                    6. Maintain a helpful and professional tone.

                    **Question:** {query}

                    **Context:**
                    {context}

                    **Answer:**
                    """

    return system_prompt, relevant_chunks

def main():
    st.title("📄 RAG with PDF: Attention is All You Need")
    st.markdown("**Educational RAG Pipeline with PDF Ingestion & Chunking**")
    st.markdown(f"Ask questions about the famous Transformer paper: *{PAPER_TITLE}*")

    # Sidebar
    with st.sidebar:
        st.header("📖 Document Info")
        st.markdown(f"**Paper:** Attention is All You Need")
        st.markdown(f"**Source:** [arXiv:1706.03762]({PAPER_URL})")

        st.divider()
        st.header("🔧 Pipeline Settings")
        st.markdown("**Chunk Size:** 500 characters")
        st.markdown("**Overlap:** 50 characters")
        st.markdown("**Top K Results:** 3 chunks")

    # Load and process PDF
    try:
        chunks = load_and_process_pdf(PAPER_URL)
        embedding_model = initialize_embedding_model()
        index, embeddings = build_faiss_index(embedding_model, chunks)
        llm = initialize_llm()

        st.success(f"✓ System ready! Vector database contains {len(chunks)} chunks from the paper")

    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.stop()

    # Main query interface
    st.divider()

    # Query input
    query = st.text_input(
        "🔍 Ask your question about the paper:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., How does multi-head attention work?"
    )

    if query:
        st.divider()

        try:
            with st.spinner("🔎 Searching knowledge base and generating answer..."):
                prompt, relevant_chunks = generate_prompt(query, embedding_model, index, chunks)

                if prompt is None:
                    st.warning("I couldn't find any relevant information to answer your question.")
                    st.stop()

                response = llm.invoke(prompt)

            # Display final answer
            st.subheader("💬 Answer:")
            st.markdown(response.content)

            # Display sources
            st.divider()
            st.subheader("📖 Sources:")

            for i, chunk in enumerate(relevant_chunks, 1):
                relevance_score = 1 / (1 + chunk['distance'])
                with st.expander(f"Source {i}: Page {chunk['page']} (Relevance: {relevance_score:.2f})"):
                    st.markdown(f"**Page:** {chunk['page']}")
                    st.markdown(f"**Content:**")
                    st.text(chunk['content'])

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
