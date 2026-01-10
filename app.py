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
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def build_faiss_index(_embedding_model):
    """Build and cache the FAISS index."""
    doc_texts = [doc["content"] for doc in DOCUMENTS]
    embeddings = _embedding_model.encode(doc_texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    return index, embeddings

def initialize_llm():
    """Initialize the language model."""
    # Check for API key in environment or Streamlit secrets
    api_key = os.getenv('GOOGLE_API_KEY') or st.secrets.get("GOOGLE_API_KEY", None)

    if not api_key:
        st.error("⚠️ Google API Key not found. Please set it in Streamlit secrets or environment variables.")
        st.stop()

    return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=api_key)

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
        with st.spinner("🔎 Searching knowledge base and generating answer..."):
            try:
                # Generate prompt and get relevant articles
                prompt, relevant_articles = generate_prompt(query, embedding_model, index)

                if prompt is None:
                    st.warning("I couldn't find any relevant information to answer your question.")
                else:
                    # Get LLM response
                    response = llm.invoke(prompt)

                    # Display answer
                    st.subheader("💬 Answer:")
                    st.markdown(response.content)

                    # Display sources
                    st.divider()
                    st.subheader("📖 Sources:")

                    for i, article in enumerate(relevant_articles, 1):
                        with st.expander(f"Source {i}: {article['title']} (Relevance Score: {1/(1+article['distance']):.2f})"):
                            st.markdown(f"**Date:** {article['date']}")
                            st.markdown(f"**Content:** {article['content']}")
                            st.caption(f"Distance: {article['distance']:.4f}")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
