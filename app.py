import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Question Answering System",
    page_icon="🤖",
    layout="wide"
)

# Sample documents
DOCUMENTS = [
    {
        "title": "Introduction to AI",
        "content": "Artificial Intelligence is the simulation of human intelligence by machines. It includes learning, reasoning, and self-correction.",
        "date": "2023-01-15"
    },
    {
        "title": "Machine Learning Basics",
        "content": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
        "date": "2023-02-10"
    },
    {
        "title": "Deep Learning Overview",
        "content": "Deep Learning uses neural networks with multiple layers to analyze various factors of data. It's particularly effective for image and speech recognition.",
        "date": "2023-03-05"
    },
    {
        "title": "Natural Language Processing",
        "content": "Natural Language Processing (NLP) helps computers understand, interpret and generate human language in a valuable way.",
        "date": "2023-04-12"
    },
    {
        "title": "Computer Vision Fundamentals",
        "content": "Computer Vision enables computers to derive meaningful information from digital images, videos and other visual inputs.",
        "date": "2023-05-20"
    },
    {
        "title": "Python for Data Science",
        "content": "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science and AI.",
        "date": "2023-06-18"
    },
    {
        "title": "TensorFlow Framework",
        "content": "TensorFlow is an open-source machine learning framework developed by Google for building and training neural networks.",
        "date": "2023-07-25"
    }
]

@st.cache_resource
def initialize_embedding_model():
    """Initialize and cache the embedding model."""
    with st.status("Loading embedding model...", expanded=True) as status:
        st.write("Loading SentenceTransformer model: all-MiniLM-L6-v2")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.write("✓ Embedding model loaded successfully!")
        status.update(label="Embedding model ready!", state="complete")
    return model

@st.cache_resource
def build_faiss_index(_embedding_model):
    """Build and cache the FAISS index."""
    with st.status("Building FAISS vector database...", expanded=True) as status:
        st.write(f"Step 1: Extracting text from {len(DOCUMENTS)} documents")
        doc_texts = [doc["content"] for doc in DOCUMENTS]

        st.write("Step 2: Generating embeddings for all documents")
        embeddings = _embedding_model.encode(doc_texts)
        st.write(f"✓ Generated embeddings with dimension: {embeddings.shape[1]}")

        st.write("Step 3: Creating FAISS index with L2 (Euclidean) distance")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        st.write(f"✓ FAISS index created with {index.ntotal} vectors")

        status.update(label="Vector database ready!", state="complete")

    return index, embeddings

def initialize_llm():
    """Initialize the language model."""
    # Check for API key in environment or Streamlit secrets
    api_key = os.getenv('GOOGLE_API_KEY') or st.secrets.get("GOOGLE_API_KEY", None)
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

def get_relevant_articles(query, embedding_model, index, k=3):
    """
    Retrieves the most relevant articles from FAISS index based on the query.

    Args:
        query (str): The user's question
        embedding_model: The sentence transformer model
        index: FAISS index
        k (int): Number of articles to retrieve

    Returns:
        list: List of relevant documents with metadata
    """
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)

    relevant_articles = []
    for idx, distance in zip(indices[0], distances[0]):
        relevant_articles.append({
            'title': DOCUMENTS[idx]['title'],
            'content': DOCUMENTS[idx]['content'],
            'date': DOCUMENTS[idx]['date'],
            'distance': float(distance)
        })

    return relevant_articles

def generate_prompt(query, embedding_model, index):
    """
    Generates a structured prompt for the LLM based on the user's query and retrieved context.

    Args:
        query (str): The user's question
        embedding_model: The sentence transformer model
        index: FAISS index

    Returns:
        tuple: (prompt string, relevant articles)
    """
    relevant_articles = get_relevant_articles(query, embedding_model, index, k=3)

    context_parts = []
    for i, article in enumerate(relevant_articles, 1):
        source = f"""[Source {i}]
Title: {article.get('title', 'No Title')}
Date: {article.get('date', 'No Date')}
Content: {article.get('content', '')}
"""
        context_parts.append(source)

    context = "\n".join(context_parts)

    if not context.strip():
        return None, relevant_articles

    system_prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.

**Instructions:**
1. Answer the user's question using ONLY the information in the Context section below.
2. Be concise and informative in your responses.
3. If the answer is not in the context, respond with: "I don't have enough information to answer that question."
4. When using information from the context, cite the source number (e.g., [Source 1]).
5. Do not make up or assume information not present in the context.
6. Maintain a helpful and professional tone.

**Question:** {query}

**Context:**
{context}

**Answer:**"""

    return system_prompt, relevant_articles

def main():
    st.title("🤖 RAG Question Answering System")
    st.markdown("Ask questions about AI, Machine Learning, and related topics!")

    # Sidebar for API key configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value=os.getenv('GOOGLE_API_KEY', ''),
            help="Enter your Google API key or set it in environment variables"
        )

        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key

        st.divider()
        st.header("📚 Knowledge Base")
        st.markdown(f"**Documents:** {len(DOCUMENTS)}")

        with st.expander("View Documents"):
            for doc in DOCUMENTS:
                st.markdown(f"**{doc['title']}**")
                st.caption(f"Date: {doc['date']}")
                st.text(doc['content'])
                st.divider()

    # Initialize models
    try:
        embedding_model = initialize_embedding_model()
        index, embeddings = build_faiss_index(embedding_model)
        llm = initialize_llm()

        st.success(f"✓ System initialized with {index.ntotal} documents in the vector database")

    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.stop()

    # Main query interface
    st.divider()

    # Example queries
    st.subheader("💡 Try these example questions:")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("What is Deep Learning?"):
            st.session_state.query = "What is Deep Learning and how does it work?"

    with col2:
        if st.button("Explain Machine Learning"):
            st.session_state.query = "What is Machine Learning?"

    with col3:
        if st.button("Tell me about NLP"):
            st.session_state.query = "What is Natural Language Processing?"

    # Query input
    query = st.text_input(
        "🔍 Ask your question:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., What is the difference between AI and Machine Learning?"
    )

    if query:
        # Add a toggle to show/hide processing steps
        show_process = st.checkbox("Show processing steps (Educational Mode)", value=True)

        st.divider()

        try:
            if show_process:
                # Show detailed processing steps
                st.subheader("🔍 RAG Pipeline Process")

                # Step 1: Query embedding
                with st.expander("**Step 1: Convert Query to Embedding**", expanded=True):
                    st.write(f"**Query:** {query}")
                    st.write("Converting your question into a vector representation...")
                    query_embedding = embedding_model.encode([query]).astype('float32')
                    st.write(f"✓ Query embedded into {query_embedding.shape[1]}-dimensional vector")
                    st.code(f"Embedding shape: {query_embedding.shape}")

                # Step 2: Vector search
                with st.expander("**Step 2: Search FAISS Vector Database**", expanded=True):
                    st.write("Searching for the 3 most similar documents using L2 distance...")
                    distances, indices = index.search(query_embedding, 3)
                    st.write(f"✓ Found {len(indices[0])} relevant documents")

                    # Show search results
                    for i, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
                        st.write(f"**Match {i}:** {DOCUMENTS[idx]['title']} (Distance: {distance:.4f})")

                # Step 3: Retrieve documents
                with st.expander("**Step 3: Retrieve Full Document Content**", expanded=True):
                    relevant_articles = get_relevant_articles(query, embedding_model, index, k=3)
                    st.write(f"Retrieved {len(relevant_articles)} documents:")
                    for i, article in enumerate(relevant_articles, 1):
                        st.write(f"{i}. **{article['title']}** ({article['date']})")

                # Step 4: Build prompt
                with st.expander("**Step 4: Build RAG Prompt**", expanded=True):
                    prompt, _ = generate_prompt(query, embedding_model, index)
                    st.write("Constructing prompt with context for the LLM...")
                    st.code(prompt, language="text")

                # Step 5: LLM generation
                with st.expander("**Step 5: Generate Answer with LLM**", expanded=True):
                    st.write("Sending prompt to Google Gemini Pro...")
                    with st.spinner("Waiting for LLM response..."):
                        response = llm.invoke(prompt)
                    st.write("✓ Response received!")

                st.divider()
            else:
                # Quick mode without detailed steps
                with st.spinner("🔎 Searching knowledge base and generating answer..."):
                    prompt, relevant_articles = generate_prompt(query, embedding_model, index)

                    if prompt is None:
                        st.warning("I couldn't find any relevant information to answer your question.")
                        st.stop()

                    response = llm.invoke(prompt)

            # Display final answer
            st.subheader("💬 Final Answer:")
            st.markdown(response.content)

            # Display sources
            st.divider()
            st.subheader("📖 Source Documents:")

            relevant_articles = get_relevant_articles(query, embedding_model, index, k=3)
            for i, article in enumerate(relevant_articles, 1):
                with st.expander(f"Source {i}: {article['title']} (Relevance Score: {1/(1+article['distance']):.2f})"):
                    st.markdown(f"**Date:** {article['date']}")
                    st.markdown(f"**Content:** {article['content']}")
                    st.caption(f"L2 Distance: {article['distance']:.4f}")

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
