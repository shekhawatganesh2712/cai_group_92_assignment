import streamlit as st
import requests
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Load Pre-trained Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title("📄 RAG Chatbot - Apple's Financial Reports")
st.write("Enter a query to search Apple's financial reports using RAG.")

query = st.text_input("🔍 Enter your query:", "")

# Example Data (Replace with actual extracted text)
urls = {
    '2023': 'https://s2.q4cdn.com/470004039/files/doc_earnings/2023/q4/filing/_10-K-Q4-2023-As-Filed.pdf',
    '2024': 'https://s2.q4cdn.com/470004039/files/doc_earnings/2024/q4/filing/10-Q4-2024-As-Filed.pdf'
}

# Function to download PDFs
def download_pdf(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    return False

# Extract Text from PDFs
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Process and Structure Data
def clean_and_split_text(text):
    import re
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    sections = re.split(r'\n\s*\n', text)
    return sections

# Convert Text to Embeddings
def embed_chunks(chunks):
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings

# Retrieve Results
def retrieve_similar_chunks(query, index, chunks, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)
    results = [chunks[idx] for idx in indices[0]]
    return results

# Load Data & Index
@st.cache_data
def load_data():
    texts = {}
    for year, url in urls.items():
        pdf_path = f'apple_{year}_annual_report.pdf'
        if download_pdf(url, pdf_path):
            texts[year] = extract_text_from_pdf(pdf_path)

    structured_texts = {year: clean_and_split_text(text) for year, text in texts.items()}
    text_chunks = {year: [" ".join(sec.split()[:500]) for sec in sections] for year, sections in structured_texts.items()}
    
    # Create embeddings
    embeddings = {year: embed_chunks(chunks) for year, chunks in text_chunks.items()}
    
    # Build FAISS Index
    embedding_dim = embeddings['2023'].shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    
    all_chunks = []
    for year in text_chunks.keys():
        all_chunks.extend(text_chunks[year])
    
    all_embeddings = np.vstack([embeddings[year] for year in embeddings.keys()])
    index.add(all_embeddings)
    
    return index, all_chunks

index, all_chunks = load_data()

# Search on button click
if st.button("🔍 Search"):
    if query:
        results = retrieve_similar_chunks(query, index, all_chunks)
        st.write("### Results:")
        for i, res in enumerate(results, 1):
            st.write(f"**Result {i}:** {res}")
    else:
        st.warning("⚠️ Please enter a query.")
