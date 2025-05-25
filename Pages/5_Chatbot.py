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
from sklearn.tree import plot_tree

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="BI Chatbot", page_icon="")
st.title("BI Chatbot – Files and Web")



# --- MODEL + VECTOR STORE INITIALIZATION ---
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




# --- PDF & TEXT FILE HANDLING ---
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
    


# --- LOAD DEFAULT CHAT FILES ---
chat_dir = os.path.join(os.path.dirname(__file__), "..", "Chatbot", "Chat")
filenames = ["4_Prediction.txt", "GitReadme.txt", "1_LoadAndClean.txt", "3_HabitFactors.txt"]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

for file in filenames:
    path = os.path.join(chat_dir, file)
    if os.path.exists(path):
        text = extract_text_from_txt(path)
        chunks = splitter.create_documents(text)
        vector_store.add_documents(chunks)
    else:
        st.sidebar.warning(f"⚠️ File {file} was not found.")


    

# --- INTERACTIVE FILE UPLOAD ---
st.sidebar.markdown("### 📂 Upload your own files")
uploaded_files = st.sidebar.file_uploader(
    "Choose .pdf or .txt files", type=["pdf", "txt"], accept_multiple_files=True
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
    st.sidebar.success(f" Loaded {len(chunks)} text chunks from uploaded files.")



# --- LOAD TEXT FROM WEB ---
def fetch_web_text(url):
    loader = SeleniumURLLoader(urls=[url])
    docs = loader.load()
    return [doc.page_content for doc in docs]

st.sidebar.markdown("### 🌐 Load from web")
url = st.sidebar.text_input("Enter URL", placeholder="https://...")
if url and st.sidebar.button(" Add web text"):
    with st.spinner("🔗 Fetching content..."):
        try:
            web_texts = fetch_web_text(url)
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            chunks = splitter.create_documents(web_texts)
            vector_store.add_documents(chunks)
            st.sidebar.success(f" Text from {url} loaded")
        except Exception as e:
            st.sidebar.error(f" Error: {e}")




# --- PLOT FUNCTIONS TO MAKE CHATBOT SHOW DIAGRAMS  ---
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

    from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

@st.cache_data
def load_exam_data():
    df = pd.read_csv('Data/CleanedData/simulated_data_cleaned.csv')
    df['exam_level'] = df['exam_score'].apply(lambda score:
        'very_low' if score <= 40 else
        'low' if score <= 55 else
        'medium' if score <= 70 else
        'high' if score <= 85 else 'very_high')
    return df

student_df = load_exam_data()

def show_interactive_habit_comparison():
    st.markdown("### 📊 Interactive Habit Comparison")
    numeric_cols = ['social_media_hours', 'netflix_hours', 'attendance_percentage', 'sleep_hours']
    category_cols = ['mental_health_rating', 'exercise_frequency']

    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select a feature for the X-axis", numeric_cols + category_cols, index=0)
    with col2:
        y_feature = st.selectbox("Select a feature for the Y-axis", ['exam_score'], index=0)

    fig, ax = plt.subplots()
    if x_feature in category_cols:
        sns.boxplot(x=x_feature, y=y_feature, data=student_df, ax=ax)
    else:
        sns.regplot(x=x_feature, y=y_feature, data=student_df, ax=ax)

    ax.set_title(f"{x_feature.replace('_', ' ').title()} vs {y_feature.title()}")
    st.pyplot(fig)

def show_decision_tree_model():
    st.markdown("### 🌳 Decision Tree Prediction Model")

    features = [
        'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating'
    ]

    # Filter rows with missing exam_level
    student_filtered = student_df.dropna(subset=["exam_level"])

    X = student_filtered[features]
    y = student_filtered['exam_level']

    # Sanity check
    if y.nunique() < 2:
        st.error("⚠️ Not enough classes in exam_level to train a decision tree.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)

    st.write("Classes:", list(tree.classes_))

    fig = plt.figure(figsize=(25, 10))
    plot_tree(tree, feature_names=features, class_names=[str(c) for c in tree.classes_], filled=True)
    st.pyplot(fig)

    st.markdown("🔍 This decision tree shows how student habits split into exam performance categories.")


# --- CHAT INTERFACE ---
st.header("💬 Ask a question")
user_input = st.text_input("Write your question:")



# --- SHOW DIAGRAMS ---
    #If user inputs something with the starte of the if - statment = that diagram is shown 

if user_input:
    lower_input = user_input.lower()

    # 4_Prediction"

    if "gpa distribution" in lower_input or "gpa chart" in lower_input:
        show_histogram(
            "GPA",
            "GPA Distribution",
            "GPA (0–4)",
            "Most students have a GPA between **2.0 and 3.0**, with a peak in that range."
        )

    elif "study time distribution" in lower_input or "study time chart" in lower_input:
        show_histogram(
            "StudyTimeWeekly",
            "Study Time Distribution",
            "Hours/Week",
            "Most students study around **10–15 hours/week**. After 20 hours, the number drops sharply."
        )

    elif "correlation" in lower_input or ("study time" in lower_input and "gpa" in lower_input):
        show_scatter(
            "StudyTimeWeekly", "GPA",
            "Study Time vs GPA",
            "This plot shows a **positive relationship** between study time and GPA."
        )
            # --- Load & Clean Data Preview ---
    elif "preview of real" in lower_input or "show real dataset" in lower_input:
        st.markdown("### Preview of Cleaned Real Dataset")
        st.dataframe(real_df.head())
        st.write("Shape:", real_df.shape)
        st.write("Missing values:", real_df.isnull().sum().sum())
        st.write("Duplicate entries:", real_df.duplicated().sum())

    elif "preview of simulated" in lower_input or "show simulated dataset" in lower_input:
        st.markdown("### Preview of Cleaned Simulated Dataset")
        st.dataframe(sim_df.head())
        st.write("Shape:", sim_df.shape)
        st.write("Missing values:", sim_df.isnull().sum().sum())
        st.write("Duplicate entries:", sim_df.duplicated().sum())


        # --- Habits -----
    elif "interactive habit" in lower_input or "habit comparison" in lower_input:
        show_interactive_habit_comparison()

    elif "decision tree" in lower_input or "prediction model" in lower_input:
        show_decision_tree_model()
    else:
        with st.spinner("Finding answer..."):
            docs = vector_store.similarity_search(user_input, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = ChatPromptTemplate.from_template("""
You are a helpful BI assistant. Use the following context to answer:

{context}

Question: {question}
""")
            response = llm.invoke(prompt.format(context=context, question=user_input))
            st.markdown("### Answer:")
            st.write(response)
if 'show_cheatsheet' not in st.session_state:
    st.session_state.show_cheatsheet = False

# Initialize state
if "show_cheatsheet" not in st.session_state:
    st.session_state.show_cheatsheet = False

# Toggle function
def toggle_cheatsheet():
    st.session_state.show_cheatsheet = not st.session_state.show_cheatsheet

# Knappen med on_click
toggle_label = " Hide Example Questions" if st.session_state.show_cheatsheet else " Show Example Questions"
st.button(toggle_label, on_click=toggle_cheatsheet)

# Cheat Sheet visning
if st.session_state.show_cheatsheet:
    st.markdown("### Questions you can ask:")
    st.markdown("""
- What is the GPA distribution?
- Show me a preview of real (or simulated) dataset
- Is there a correlation between study time and GPA?
- Show me Interactive Habit Comparison
- What does the prediction model do?
- What does the uploaded PDF say about data quality?
    """)
    st.markdown("---")
# 
st.markdown(
    """
    <p style='font-size: 0.75rem; color: gray;'>
     Answers will likely be in <strong>Danish</strong>.<br> 
     Please note that this is not ChatGPT. If something doesn't work the first time, try asking your question again.<br>
     The chatbot can be wrong. We recommend double-checking important information.
    </p>
    """,
    unsafe_allow_html=True
)