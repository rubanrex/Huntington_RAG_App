import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from openai import OpenAI   
import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import fitz  # PyMuPDF for PDF-based figure extraction
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Fetch the API key securely from environment
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

if not api_key:
    raise ValueError("⚠️ OPENAI_API_KEY not found. Please check your .env file.")

# Ensure 'static/' folder exists
if not os.path.exists("static"):
    os.makedirs("static")

# Extract figures from PMC articles
def extract_figures(pmc_id):
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    figures = []
    for idx, fig in enumerate(soup.find_all('div', {'class': 'fig'})):
        img_tag = fig.find('img')
        if img_tag:
            img_url = img_tag['src']
            img_response = requests.get(img_url)
            img_path = f"static/figure_{pmc_id}_{idx}.png"
            with open(img_path, "wb") as f:
                f.write(img_response.content)
            figures.append(img_path)
    return figures

# OCR to extract figure captions
def extract_caption(image_path):
    try:
        image = Image.open(image_path)
        caption_text = pytesseract.image_to_string(image)
        return caption_text.strip() or "Caption not detected."
    except Exception as e:
        return "Error reading caption."

# Load SentenceTransformer model
@st.cache_data
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Build FAISS index
@st.cache_data
def build_index(docs):
    embeddings = model.encode(docs, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Load articles
@st.cache_data
def load_articles(filename="huntington_articles.json"):
    with open(filename, "r") as f:
        articles = json.load(f)
    docs = []
    for article in articles:
        title = article.get("TI", "")
        abstract = article.get("AB", "")
        docs.append(title + ". " + abstract)
    return docs

documents = load_articles()
index, embeddings = build_index(documents)

# Query FAISS index to retrieve relevant articles
def query_documents(query_text, top_n=5):
    query_emb = model.encode([query_text]).astype("float32")
    distances, indices = index.search(query_emb, top_n)
    retrieved = [documents[i] for i in indices[0]]
    return retrieved

# Generate LLM answer
def ask_llm(query_text, context_docs):
    context = "\n\n".join(context_docs)
    prompt = f"""
    Based on the following literature excerpts and corresponding figures, answer the question:

    {context}

    Question: {query_text}
    Answer:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Switch to GPT-3.5-Turbo for compatibility
            messages=[
                {"role": "system", "content": "You are an expert in Huntington disease research."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❗️ Error generating response: {e}"


# Streamlit interface
st.title("🧠 Huntington Disease Literature Query Interface")

# Dynamic PMC ID selection
pmc_id = st.text_input("Enter PMC ID for Figures (Optional):", "7252973")

user_query = st.text_input("Enter your question about Huntington disease:")

if st.button("Submit Query"):
    if user_query:
        st.write("🔍 Retrieving relevant literature excerpts...")
        retrieved_docs = query_documents(user_query, top_n=5)

        # Display text excerpts
        st.markdown("### 📄 Top Retrieved Excerpts:")
        for idx, doc in enumerate(retrieved_docs):
            st.markdown(f"**Excerpt {idx+1}:**")
            st.write(doc)

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
