import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import ollama
import time

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Ensure 'static/' folder exists
if not os.path.exists("static"):
    os.makedirs("static")

# Define the Ollama model to use
OLLAMA_MODEL = "llama3"  # Replace with the desired Ollama model

# Load SentenceTransformer model
@st.cache_data
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load articles and keep DOI information
@st.cache_data
def load_articles(filename="huntington_articles.json"):
    with open(filename, "r") as f:
        articles = json.load(f)
    docs = []
    for article in articles:
        title = article.get("TI", "")
        abstract = article.get("AB", "")
        text = title + ". " + abstract
        doi = ""
        # Try to extract DOI from the AID field if available
        aid = article.get("AID", "")
        if isinstance(aid, list):
            for item in aid:
                if "doi" in item.lower():
                    doi = item
                    break
        elif isinstance(aid, str) and "doi" in aid.lower():
            doi = aid
        docs.append({"text": text, "doi": doi})
    return docs

documents = load_articles()

# Build FAISS index using only the document text
@st.cache_data
def build_index(docs):
    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

index, embeddings = build_index(documents)

# Query FAISS index to retrieve relevant articles (including DOI info)
def query_documents(query_text, top_n=5):
    query_emb = model.encode([query_text]).astype("float32")
    distances, indices = index.search(query_emb, top_n)
    retrieved = [documents[i] for i in indices[0]]
    return retrieved

# Extract figures from PMC articles
def extract_figures(pmc_id):
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to fetch PMC article with ID {pmc_id}. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    figures = []
    for idx, fig in enumerate(soup.find_all('div', {'class': 'fig'})):
        img_tag = fig.find('img')
        if img_tag:
            img_url = img_tag['src']
            if not img_url.startswith("http"):
                img_url = "https://www.ncbi.nlm.nih.gov" + img_url  # Fix relative URLs
            
            img_response = requests.get(img_url)
            if img_response.status_code == 200:
                img_path = f"static/figure_{pmc_id}_{idx}.png"
                with open(img_path, "wb") as f:
                    f.write(img_response.content)
                figures.append(img_path)
            else:
                print(f"Failed to download image from {img_url}. Status code: {img_response.status_code}")
    
    return figures

# OCR to extract figure captions
def extract_caption(image_path):
    try:
        image = Image.open(image_path)
        caption_text = pytesseract.image_to_string(image)
        return caption_text.strip() or "Caption not detected."
    except Exception as e:
        return "Error reading caption."

# Improved LLM prompt that includes the retrieved DOI information in the context
def ask_llm(query_text, context_docs):
    """
    Generate an answer from the LLM based on the provided query and context documents.
    The prompt instructs the model to synthesize a concise answer directly from the given excerpts,
    which now include DOI information if available.
    """
    # Build context string: for each doc include its text and DOI if available
    context = "\n\n".join([
        f"{doc['text']}\nDOI: {doc['doi']}" if doc['doi'] else doc['text']
        for doc in context_docs
    ])
    
    # Construct a detailed prompt that guides the LLM
    prompt = f"""
You are a scientific research assistant. 
Based on the following excerpts from the literature, provide a concise, evidence-based answer to the question below.
Make sure your answer directly references and synthesizes the information in the context.

Context:
{context}

Question:
{query_text}

Answer:
    """
    
    # Query the Ollama model with the improved prompt
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract and return the model's answer
    result = response['message']['content'].strip()
    return result

# Streamlit interface
st.title("🧠 Huntington Disease Literature Query Interface")

# Dynamic PMC ID selection for figure extraction
pmc_id = st.text_input("Enter PMC ID for Figures (Optional):", "7252973")

user_query = st.text_input("Enter your question about Huntington disease:")

if st.button("Submit Query"):
    if user_query:
        st.write("🔍 Retrieving relevant literature excerpts...")
        retrieved_docs = query_documents(user_query, top_n=5)

        # Display text excerpts along with DOI information
        st.markdown("### 📄 Top Retrieved Excerpts:")
        for idx, doc in enumerate(retrieved_docs):
            st.markdown(f"**Excerpt {idx+1}:**")
            st.write(doc["text"])
            if doc["doi"]:
                st.write(f"**DOI:** {doc['doi']}")
        
        # Extract and display relevant figures
        st.markdown("### 🖼️ Related Figures:")
        figures = extract_figures(pmc_id=pmc_id.strip())  
        if figures:
            for fig in figures:
                st.image(fig, caption=extract_caption(fig), use_column_width=True)
        else:
            st.write("❗️ No figures found for the selected PMC ID.")

        # LLM response
        st.write("🤖 Querying the language model for an answer...")
        answer = ask_llm(user_query, retrieved_docs)
        st.markdown("### 🧠 LLM Answer:")
        st.write(answer)
    else:
        st.write("❗️ Please enter a query before submitting.")
