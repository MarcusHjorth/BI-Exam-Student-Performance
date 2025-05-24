# --- 📦 IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ---  PAGE SETUP ---
st.set_page_config(page_title=" GPA Predictor", layout="centered")
st.title("🎓 GPA Prediction Using Study Time")

st.markdown("""
    <style>
        /* General */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f8f9fc;
            color: #333;
        }

        /* Section headers */
        h1, h2, .stSubheader {
            color: #1a1a1a;
        }

        /* Chart description styling */
        .chart-description {
            background-color: #eef1f6;
            border-left: 5px solid #4c8bf5;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            font-size: 0.95rem;
        }

        /* Model performance container */
        .model-container {
            background-color: #ffffff;
            border: 1px solid #dce3ed;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }

        /* Space between plots */
        .element-container:has(> .stPyplotChart) {
            margin-bottom: 2rem;
        }

        /* Sliders + SelectBox spacing */
        .stSlider, .stRadio, .stSelectbox {
            margin-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)





# ---  LOAD DATA ---
real_df = pd.read_csv("Data/CleanedData/real_data_cleaned.csv")
sim_df = pd.read_csv("Data/CleanedData/simulated_data_cleaned.csv")


# ---  DATA PREP ---
def prepare_data(real, sim):
    sim["StudyTimeWeekly"] = sim["study_hours_per_day"] * 5
    sim["GPA"] = (sim["exam_score"] / 100) * 4

    real_model = real[["StudyTimeWeekly", "GPA"]]
    sim_model = sim[["StudyTimeWeekly", "GPA"]]

    return pd.concat([real_model, sim_model], ignore_index=True)

combined_df = prepare_data(real_df, sim_df)


# ---  PLOT FUNCTIONS ---
def plot_distribution(data, column, title, xlabel, note):
    st.markdown(f"<div class='chart-description'>{note}</div>", unsafe_allow_html=True)

    fig, ax = plt.subplots()
    sns.histplot(data[column], kde=True, bins=20, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.markdown("---")

def plot_scatter(data, x, y, title, note):
    st.markdown(f"<div class='chart-description'>{note}</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, data=data, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)
    st.markdown("---")


# ---  VISUAL EXPLORATION ---
st.subheader(" Data Distributions")

plot_distribution(
    combined_df, "GPA", "GPA Distribution", "GPA (0-4)",
    "Most students have a GPA between **2.0 and 3.0**, with a peak in that range. Only a few students score extremely low (0.0) or high (4.0)."
)

plot_distribution(
    combined_df, "StudyTimeWeekly", "Study Time Distribution", "Hours/Week",
    "Most students study around **10–15 hours per week**. After 20 hours, the number of students drops sharply"
)

plot_scatter(
    combined_df, "StudyTimeWeekly", "GPA", "Study Time vs GPA",
    "This plot shows a **positive relationship** between study time and GPA. Students who study more hours per week tend to have higher GPAs. However, the points are somewhat spread out — it’s not a perfect line."
)


# --- SIMPLE LINEAR REGRESSION ---
X = combined_df[["StudyTimeWeekly"]]
y = combined_df["GPA"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

y_pred_simple = model_simple.predict(X_test)


# ---  EXTENDED MODEL ---
def train_extended_model(real, sim):
    sim = sim.rename(columns={"parental_education_level": "ParentalEducation"})
    sim["Absences"] = np.random.randint(0, 15, len(sim))
    sim["GPA"] = (sim["exam_score"] / 100) * 4

    real_ext = real[["StudyTimeWeekly", "ParentalEducation", "Absences", "GPA"]]
    sim_ext = sim[["StudyTimeWeekly", "ParentalEducation", "Absences", "GPA"]]

    df = pd.concat([real_ext, sim_ext], ignore_index=True).dropna()
    df = df[df["ParentalEducation"].apply(lambda x: isinstance(x, str))]

    encoder = LabelEncoder()
    df["ParentalEducation"] = encoder.fit_transform(df["ParentalEducation"])

    X = df[["StudyTimeWeekly", "ParentalEducation", "Absences"]]
    y = df["GPA"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, encoder, y_test, y_pred

model_ext, encoder, y_test_ext, y_pred_ext = train_extended_model(real_df, sim_df)

st.subheader(" Model Performance")

st.markdown("#### Simple Model ")
col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2_score(y_test, y_pred_simple):.3f}")
col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_simple)):.3f}")

st.markdown("#### Extended Model ")
col3, col4 = st.columns(2)
col3.metric("R² Score", f"{r2_score(y_test_ext, y_pred_ext):.3f}")
col4.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test_ext, y_pred_ext)):.3f}")


# ---  INTERACTIVE PREDICTION ---
st.subheader("Try It Yourself")

model_option = st.radio("Select Model:", ["Simple", "Extended"])
study_hours = st.slider("Study Time per Week (hrs)", 0, 40, 10)

if model_option == "Simple":
    result = model_simple.predict([[study_hours]])[0]
    st.success(f"Predicted GPA: {result:.2f}")
else:
    absences = st.slider("Number of Absences", 0, 30, 5)
    education = st.selectbox("Parental Education", encoder.classes_)
    education_encoded = encoder.transform([education])[0]

    result = model_ext.predict([[study_hours, education_encoded, absences]])[0]
    st.success(f"Predicted GPA: {result:.2f}")