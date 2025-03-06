import streamlit as st
from groq import Groq
import PyPDF2
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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

# System prompt for legal document summarization
system_prompt = """
You are an AI assistant specialized in summarizing legal documents. Extract key points, obligations, risks, and important clauses in a professional and structured manner.
"""

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

# Groq API response for summarization
def get_summary(input_text):
    try:
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": input_text}],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Failed to get response from Groq API: {e}")
        return ""

# Retrieve relevant chunks
def retrieve_chunks(query, index, chunks, k=3):
    query_emb = embedding_model.encode([query])
    _, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0]]

# Groq API response for chatbot
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

# Visualize risk detection
risks = ['Liability', 'Breach', 'Penalty', 'Ambiguity']
risk_counts = np.random.randint(1, 10, size=len(risks))  # Placeholder counts

# Streamlit app
st.title("AI Legal Document Summarization and RAG Chatbot System")

uploaded_file = st.file_uploader("Upload a legal document (PDF)", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = split_text(text)
    embeddings = generate_embeddings(chunks)
    index = build_index(embeddings)

    # Summarize the document first
    summary = get_summary(text)
    st.subheader("Document Summary")
    st.write(summary)

    # Chart section at the top
    st.subheader("Risk Analysis Charts")
    col1, col2 = st.columns(2)

    # Risk frequency bar chart
    with col1:
        st.subheader("Risk Frequency")
        fig, ax = plt.subplots()
        ax.bar(risks, risk_counts, color='skyblue')
        ax.set_title('Frequency of Different Types of Risks')
        ax.set_xlabel('Risk Type')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    # Risk proportion pie chart
    with col2:
        st.subheader("Risk Proportion")
        risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_proportions = np.random.randint(1, 10, size=len(risk_labels))  # Placeholder proportions

        fig2, ax2 = plt.subplots()
        ax2.pie(risk_proportions, labels=risk_labels, autopct='%1.1f%%', startangle=140)
        ax2.set_title('Proportion of Risk Levels')
        st.pyplot(fig2)

    # Button to launch chatbot in sidebar
    with st.sidebar:
        st.subheader("Legal Document Chatbot")
        query = st.text_input("Ask any question about the document")
        if st.button("Run Chatbot"):
            if query:
                relevant_chunks = retrieve_chunks(query, index, chunks)
                context = "\n".join(relevant_chunks)
                answer = get_answer(context, query)
                st.write(answer)
            else:
                st.warning("Please enter a question before running the chatbot.")

    # Email integration
    st.subheader("Send Document Summary and Analysis via Email")
    recipient_email = st.text_input("Recipient Email Address")
    if st.button("Send Email"):
        if recipient_email:
            try:
                sender_email = os.getenv("SENDER_EMAIL")
                sender_password = os.getenv("SENDER_PASSWORD")

                message = MIMEMultipart()
                message['From'] = sender_email
                message['To'] = recipient_email
                message['Subject'] = "Legal Document Summary and Risk Analysis"

                email_body = f"""
                Document Summary:
                {summary}

                Risk Analysis:
                Risks: {', '.join(risks)}
                Frequency: {', '.join(map(str, risk_counts))}

                Risk Proportions:
                Low Risk: {risk_proportions[0]}
                Medium Risk: {risk_proportions[1]}
                High Risk: {risk_proportions[2]}
                """

                message.attach(MIMEText(email_body, 'plain'))

                with smtplib.SMTP('smtp.gmail.com', 587) as server:
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.sendmail(sender_email, recipient_email, message.as_string())

                st.success(f"Email sent successfully to {recipient_email}")
            except Exception as e:
                st.error(f"Failed to send email: {e}")
        else:
            st.warning("Please enter a valid email address.")
