import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

# Sample documents
documents = [
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

doc_contents = [doc["content"] for doc in documents]
doc_metadata = [{"title": doc["title"], "date": doc["date"]} for doc in documents]

# Step 1: Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Generate embeddings for all documents
doc_texts = [doc["content"] for doc in documents]
embeddings = embedding_model.encode(doc_texts)
print(f"✓ Generated embeddings for {len(documents)} documents")

# Step 3: Create FAISS index
dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
index.add(embeddings.astype('float32'))
print(f"✓ Created FAISS index with {index.ntotal} vectors")

# Step 4: Save the index (optional, for persistence)
faiss.write_index(index, "documents.index")
print("✓ Saved FAISS index to 'documents.index'")

print("✓ Vector database ready!")

def retrieve_relevant_articles(query, k=3):
    """
    Retrieves the most relevant articles from FAISS index based on the query.
    """
    # Step 1: Generate embedding for the query
    query_embedding = embedding_model.encode([query]).astype('float32')

    # Step 2: Search FAISS index for similar documents
    # Returns: distances (similarity scores) and indices (document IDs)
    distances, indices = index.search(query_embedding, k)

    # Step 3: Format results with metadata
    relevant_articles = []
    for idx in indices[0]:
        relevant_articles.append({
            'title': documents[idx]['title'],
            'content': documents[idx]['content'],
            'date': documents[idx]['date']
        })

    return relevant_articles

def generate_prompt(query):

    # Retrieve relevant articles from vector store
    relevant_articles = retrieve_relevant_articles(query, k=3)

    # Build context string from retrieved articles
    context = ""
    for i, article in enumerate(relevant_articles, 1):
        title = article.get('title', 'No Title')
        content = article.get('content', '')
        date_str = article.get('date', 'No Date')

        context += f"[Source {i}]\n"
        context += f"Title: {title}\n"
        context += f"Date: {date_str}\n"
        context += f"Content: {content}\n\n"

    # Handle case when no relevant context is found
    if not context.strip():
        return "I'm sorry, but I couldn't find any relevant information to answer your question."

    # Structured system prompt with clear instructions
    system_prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.

            **Instructions:**
            1. Answer the user's question using ONLY the information in the Context section below.
            2. Be concise and informative in your responses.
            3. If the answer is not in the context, respond with: "I don't have enough information to answer that question."
            4. When using information from the context, cite the source number (e.g., [Source 1]).

            **Question:** {query}

            **Context:**
            {context}

        """

    return system_prompt


sample_query = "What is Deep Learning and how does it work?"
custom_prompt = generate_prompt(sample_query)
print(custom_prompt)

# Use the prompt with the LLM
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
response = gemini_llm.invoke(custom_prompt)
print(response.content)




