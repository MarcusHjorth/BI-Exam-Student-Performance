import os
import io
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate



# --- STREAMLIT SETUP ---
st.set_page_config(page_title="BI Chatbot", page_icon="")
st.title(" BI Chatbot – fra filer og web")

# --- MODEL + VECTOR STORE INITIALISERING ---
@st.cache_resource
def init_all():
    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    llm = OllamaLLM(model="llama3.2:3b")
    vector_store = InMemoryVectorStore(embeddings)
    return embeddings, llm, vector_store

embeddings, llm, vector_store = init_all()

# --- LOAD DATA FOR DIAGRAMS ---
real_df = pd.read_csv("Data/CleanedData/real_data_cleaned.csv")
sim_df = pd.read_csv("Data/CleanedData/simulated_data_cleaned.csv")
sim_df["StudyTimeWeekly"] = sim_df["study_hours_per_day"] * 5
sim_df["GPA"] = (sim_df["exam_score"] / 100) * 4
combined_df = pd.concat([
    real_df[["StudyTimeWeekly", "GPA"]],
    sim_df[["StudyTimeWeekly", "GPA"]]
], ignore_index=True)

# --- PDF & TEXT FIL-HÅNDTERING ---
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_pil = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(img_pil, lang='eng')
            texts.append(ocr_text)
    return texts

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [f.read()]

# --- INDLÆS FLERE STANDARD CHATFILER ---
chat_dir = os.path.join(os.path.dirname(__file__), "..", "Chatbot", "Chat")
filnavne = ["4_Prediction.txt", "GitReadme.txt"]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

for fil in filnavne:
    sti = os.path.join(chat_dir, fil)
    if os.path.exists(sti):
        tekst = extract_text_from_txt(sti)
        chunks = splitter.create_documents(tekst)
        vector_store.add_documents(chunks)
    else:
        st.sidebar.warning(f"⚠️ Filen {fil} blev ikke fundet.")


# --- INTERAKTIV FILUPLOAD ---
st.sidebar.markdown("### 📂 Upload egne filer")
uploaded_files = st.sidebar.file_uploader(
    "Vælg .pdf eller .txt filer", type=["pdf", "txt"], accept_multiple_files=True
)
if uploaded_files:
    all_texts = []
    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.read())
            all_texts.extend(extract_text_from_pdf(temp_path))
            os.remove(temp_path)
        elif file.name.endswith(".txt"):
            all_texts.append(file.read().decode("utf-8"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.create_documents(all_texts)
    vector_store.add_documents(chunks)
    st.sidebar.success(f"✅ Indlæst {len(chunks)} tekstbidder fra uploadede filer.")

# --- WEBTEKST INDLÆSNING ---
def fetch_web_text(url):
    loader = SeleniumURLLoader(urls=[url])
    docs = loader.load()
    return [doc.page_content for doc in docs]

st.sidebar.markdown("### 🌐 Indlæs fra web")
url = st.sidebar.text_input("Indtast URL", placeholder="https://...")
if url and st.sidebar.button("🌍 Tilføj webtekst"):
    with st.spinner("🔗 Henter indhold..."):
        try:
            web_texts = fetch_web_text(url)
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            chunks = splitter.create_documents(web_texts)
            vector_store.add_documents(chunks)
            st.sidebar.success(f"✅ Tekst fra {url} indlæst")
        except Exception as e:
            st.sidebar.error(f"❌ Fejl: {e}")

# --- PLOT FUNCTIONS ---
def show_histogram(column, title, xlabel, note):
    st.markdown(f"<div class='chart-description'>{note}</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.histplot(combined_df[column], kde=True, bins=20, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    st.pyplot(fig)

def show_scatter(x, y, title, note):
    st.markdown(f"<div class='chart-description'>{note}</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.scatterplot(data=combined_df, x=x, y=y, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# --- CHAT INTERFACE ---
st.header("💬 Stil et spørgsmål")
user_input = st.text_input("Skriv dit spørgsmål:")

if user_input:
    lower_input = user_input.lower()

    if "gpa fordeling" in lower_input or "gpa diagram" in lower_input:
        show_histogram(
            "GPA",
            "GPA Distribution",
            "GPA (0–4)",
            "Most students have a GPA between **2.0 and 3.0**, with a peak in that range."
        )

    elif "study time fordeling" in lower_input or "study time diagram" in lower_input:
        show_histogram(
            "StudyTimeWeekly",
            "Study Time Distribution",
            "Hours/Week",
            "Most students study around **10–15 hours/week**. After 20 hours, the number drops sharply."
        )

    elif "sammenhæng" in lower_input or ("study time" in lower_input and "gpa" in lower_input):
        show_scatter(
            "StudyTimeWeekly", "GPA",
            "Study Time vs GPA",
            "This plot shows a **positive relationship** between study time and GPA."
        )

    else:
        with st.spinner("🧠 Finder svar..."):
            docs = vector_store.similarity_search(user_input, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = ChatPromptTemplate.from_template("""
Du er en hjælpsom BI-assistent. Brug følgende kontekst til at svare:

{context}

Spørgsmål: {question}
""")
            response = llm.invoke(prompt.format(context=context, question=user_input))
            st.markdown("###  Svar:")
            st.write(response)
