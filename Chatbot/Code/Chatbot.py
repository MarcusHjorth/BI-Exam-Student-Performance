import os
import streamlit as st
import fitz  # PyMuPDF
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

# --- STREAMLIT KONFIGURATION ---
st.set_page_config(page_title="BI Chatbot", page_icon="🤖")
st.title("BI Chatbot – fra filer og web")

# --- INITIALISERING ---
@st.cache_resource
def init_model():
    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    llm = OllamaLLM(model="llama3.2:3b")
    return embeddings, llm

embeddings, llm = init_model()
vector_store = InMemoryVectorStore(embeddings)

# --- FUNKTIONER TIL FILHÅNDTERING ---
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return [page.get_text() for page in doc]

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [f.read()]

def load_files(data_folder):
    texts = []
    if not os.path.exists(data_folder):
        st.sidebar.error("❌ Mappen /Data blev ikke fundet.")
        return texts

    files = [f for f in os.listdir(data_folder) if f.endswith((".pdf", ".txt"))]
    if not files:
        st.sidebar.warning("⚠️ Ingen PDF- eller TXT-filer fundet.")
        return texts

    with st.spinner("🔄 Indlæser filer..."):
        for file in files:
            path = os.path.join(data_folder, file)
            if file.endswith(".pdf"):
                texts.extend(extract_text_from_pdf(path))
            elif file.endswith(".txt"):
                texts.extend(extract_text_from_txt(path))
            st.sidebar.write(f"📄 {file}")
    st.sidebar.success(f"✅ Indlæst {len(files)} filer med {len(texts)} tekststykker.")
    return texts

# --- INDHENT TEKST FRA WEB ---
def fetch_web_text(url):
    loader = SeleniumURLLoader(urls=[url])
    docs = loader.load()
    return [doc.page_content for doc in docs]

# --- AUTOMATISK INDLÆSNING AF CHATBOT_DATA.TXT VED START ---
data_folder = os.path.join(os.path.dirname(__file__), '..', 'Chat')
chatbot_file = "Chatbot_data.txt"
file_path = os.path.join(data_folder, chatbot_file)

texts = []
if os.path.exists(file_path):
    texts = extract_text_from_txt(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.create_documents(texts)
    vector_store.add_documents(chunks)
    st.sidebar.success(f"✅ Automatisk indlæst {chatbot_file} med {len(chunks)} tekst-chunks.")
else:
    st.sidebar.warning(f"⚠️ Filen {chatbot_file} blev ikke fundet.")

# --- FILINDELÆSNING FRA MAPPE (PDF & TXT) ---
st.sidebar.header("📁 Indlæs flere lokale filer")
if st.sidebar.button("🔍 Indlæs filer fra /Chat"):
    loaded_texts = load_files(data_folder)
    if loaded_texts:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.create_documents(loaded_texts)
        vector_store.add_documents(chunks)
        st.sidebar.success(f"✅ Indlæst og tilføjet {len(chunks)} tekst-chunks fra mappe.")

# --- WEBINDLÆSNING ---
st.sidebar.header("🌐 Indlæs fra web")
url = st.sidebar.text_input("Indtast URL", placeholder="https://...")
if url and st.sidebar.button("🌍 Tilføj webtekst"):
    with st.spinner("Henter indhold..."):
        try:
            web_texts = fetch_web_text(url)
            texts.extend(web_texts)
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            chunks = splitter.create_documents(web_texts)
            vector_store.add_documents(chunks)
            st.sidebar.success(f"✅ Webtekst indlæst og tilføjet fra {url}")
        except Exception as e:
            st.sidebar.error(f"❌ Fejl ved webindlæsning: {e}")

# --- CHAT INTERFACE ---
st.header("🤖 Stil et spørgsmål")
user_input = st.text_input("Skriv dit spørgsmål:")

if user_input:
    with st.spinner("🔍 Finder svar..."):
        docs = vector_store.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = ChatPromptTemplate.from_template("""
Du er en hjælpsom BI-assistent. Brug følgende kontekst til at svare:

{context}

Spørgsmål: {question}
""")
        final_prompt = prompt.format(context=context, question=user_input)
        response = llm.invoke(final_prompt)
        st.markdown("### 💬 Svar:")
        st.write(response)
