# =========================
# Install Dependencies
# =========================

# !pip install streamlit hdbcli langchain_community langchain_openai ddgs python-docx pymupdf openai

import os
import pandas as pd
import fitz   # PyMuPDF
import docx
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st

# LangChain + HANA
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_openai import AzureOpenAIEmbeddings

# Database Libraries
from hdbcli import dbapi

# OpenAI
from openai import AzureOpenAI

# DuckDuckGo Search
from ddgs import DDGS


# =========================
# Helper Functions
# =========================
def google_search(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r["body"])
    return "\n".join(results)


def read_document(file_path):
    text = ""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    elif ext == ".pdf":
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()

    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])

    return text


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Video Script Generator", layout="wide")

st.title("🎬 AI-Powered Video Script Generator")

query = st.text_input("Enter your query (e.g., Create video script for Asus laptop)")
client_name = st.text_input("Enter client name (optional)")
uploaded_file = st.file_uploader("Upload a document (optional)", type=["txt", "pdf", "docx"])

generate_button = st.button("Generate Script")

if generate_button:
    with st.spinner("Fetching data and generating script..."):

        # --- Database Connection (⚠️ update credentials before running) ---
# ---------------------------
# Database & GPT Setup
# ---------------------------
# HANA connection
# --- Database Connection (⚠️ update credentials before running) ---
        connection = dbapi.connect(
            address=st.secrets["database"]["address"],
            port=st.secrets["database"]["port"],
            user=st.secrets["database"]["user"],
            password=st.secrets["database"]["password"],
            encrypt=True,
            autocommit=True,
            sslValidateCertificate=False,
        )

        # --- Azure Setup ---
        client = AzureOpenAI(
            azure_endpoint=st.secrets["azure"]["openai_endpoint"],
            api_key=st.secrets["azure"]["api_key"],
            api_version=st.secrets["azure"]["api_version"],
        )

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=st.secrets["azure"]["embeddings_deployment"],
            openai_api_version=st.secrets["azure"]["embeddings_api_version"],
            api_key=st.secrets["azure"]["api_key"],
            azure_endpoint=st.secrets["azure"]["openai_endpoint"],
        )

        db = HanaDB(
            embedding=embeddings,
            connection=connection,
            table_name="MARKETING_APP_CONTENT_GENERATION"
        )

        # --- HANA Docs ---
        docs = db.similarity_search(query, k=20)
        all_docs = "\n".join([doc.page_content for doc in docs])

        # --- Google Info ---
        google_info = ""
        if client_name.strip():
            google_info = google_search(f"{client_name} company profile")

        # --- Uploaded Document ---
        doc_text = ""
        if uploaded_file:
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            doc_text = read_document(file_path)

        # --- Build Context ---
        context_parts = [f"Query: {query}", f"Relevant Docs (HANA):\n{all_docs}"]
        if google_info:
            context_parts.append(f"Google Info:\n{google_info}")
        if doc_text:
            context_parts.append(f"Uploaded Document:\n{doc_text}")
        context = "\n\n".join(context_parts)

        # --- Script Writing Instructions ---
        specific_instructions = '''
        You are an expert video scriptwriter.
        Always create scene-by-scene scripts with visuals and narration.
        Use a professional but engaging tone.
        Include sections like: Problem Introduction, Product Introduction, Key Feature Highlight,
        Benefit Explanation, Real-Life Application, Call-to-Action, and Closing Shot.
        Structure each scene with:
        *Visual:* What is shown on screen
        *Narration:* Voiceover for that scene
        '''

        # --- LLM Call ---
        message_text = [
            {"role": "system", "content": specific_instructions},
            {"role": "user", "content": context}
        ]

        response = client.chat.completions.create(
            messages=message_text,
            model="Codetest",  # your Azure deployment model name
            max_tokens=1500,
            temperature=0.7
        )

        script = response.choices[0].message.content

        # --- Output ---
        st.subheader("Generated Video Script 🎥")
        st.markdown(script)
