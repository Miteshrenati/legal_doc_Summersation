import streamlit as st
from groq import Groq
import PyPDF2
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is set
if not api_key:
    st.error("GROQ API Key is missing! Please check your .env file.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=api_key)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# PDF extraction
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

# Text chunking
def split_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Generate embeddings
def generate_embeddings(chunks):
    return embedding_model.encode(chunks)

# Build FAISS index
def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Retrieve relevant chunks
def retrieve_chunks(query, index, chunks, k=3):
    query_emb = embedding_model.encode([query])
    _, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0]]

# Groq API response
def get_answer(input_text, query):
    try:
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "system", "content": "You are an AI assistant designed to answer questions based on the uploaded document."},
                      {"role": "user", "content": f"Document Context: {input_text}\nQuestion: {query}"}],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Failed to get response from Groq API: {e}")
        return ""

# Streamlit app
st.title("RAG Chartbot System")

uploaded_file = st.file_uploader("Upload a legal document (PDF)", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = split_text(text)
    embeddings = generate_embeddings(chunks)
    index = build_index(embeddings)

    query = st.text_input("Ask any question about the document")
    if query:
        relevant_chunks = retrieve_chunks(query, index, chunks)
        context = "\n".join(relevant_chunks)
        answer = get_answer(context, query)
        st.write(answer)