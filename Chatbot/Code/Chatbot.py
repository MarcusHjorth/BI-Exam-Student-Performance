import os
import streamlit as st
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
import fitz  # PyMuPDF

# --- INITIALISERING ---
st.set_page_config(page_title="BI Chatbot", page_icon="")
st.title(" BI Chatbot – txt and web ")

# --- TRIN 1: Embeddings & LLM (samme let model) ---
embeddings = OllamaEmbeddings(model="llama3.2:3b")
vector_store = InMemoryVectorStore(embeddings)
llm = OllamaLLM(model="llama3.2:3b")

# --- TRIN 2: Indlæs PDF og TXT fra ../Data ---
texts = []
data_folder = os.path.join(os.path.dirname(__file__), '..', 'Data')
st.sidebar.header("📁 Læs filer fra /Data")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return [page.get_text() for page in doc]

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [f.read()]

if os.path.exists(data_folder):
    files = [f for f in os.listdir(data_folder) if f.endswith((".pdf", ".txt"))]
    if files:
        for file in files:
            file_path = os.path.join(data_folder, file)
            st.sidebar.write(f" Indlæser: {file}")
            if file.endswith(".pdf"):
                file_texts = extract_text_from_pdf(file_path)
            elif file.endswith(".txt"):
                file_texts = extract_text_from_txt(file_path)
            texts.extend(file_texts)
        st.sidebar.success(f"✅ {len(files)} filer indlæst med {len(texts)} tekststykker.")
    else:
        st.sidebar.warning("⚠️ Ingen PDF- eller TXT-filer fundet i Data-mappen.")
else:
    st.sidebar.error("❌ Mappen /Data blev ikke fundet.")

# --- TRIN 3: Indlæs fra webside ---
st.sidebar.header("🌐 Tilføj Webside")
url = st.sidebar.text_input("Indtast en URL", placeholder="https://...")
if url:
    loader = SeleniumURLLoader(urls=[url])
    docs = loader.load()
    web_texts = [doc.page_content for doc in docs]
    texts.extend(web_texts)
    st.sidebar.success(f"🌍 Indlæst tekst fra: {url}")

# --- TRIN 4: Split tekst og tilføj til vector store ---
if texts:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.create_documents(texts)
    vector_store.add_documents(chunks)
    st.sidebar.success(f"🧠 {len(chunks)} tekst-chunks tilføjet til søgebasen.")
else:
    st.sidebar.warning("⚠️ Ingen tekstdata indlæst endnu.")

# --- TRIN 5: Promptskabelon ---
prompt = ChatPromptTemplate.from_template("""
Du er en hjælpsom BI-assistent. Brug følgende kontekst til at svare:
{context}

Spørgsmål: {question}
""")

# --- TRIN 6: Chat Interface ---
st.header("🤖 Stil et spørgsmål")
user_input = st.text_input("Skriv dit spørgsmål her:")

if user_input:
    docs = vector_store.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.format(context=context, question=user_input)
    response = llm.invoke(final_prompt)
    st.markdown("### 🔍 Svar:")
    st.write(response)
