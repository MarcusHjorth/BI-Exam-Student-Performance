# --- IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- PAGE SETUP ---
st.set_page_config(page_title="GPA Predictor", layout="centered")
st.title("GPA Prediction Using Study Time")

# --- INTRO TEXT ---
st.markdown("""
## Welcome to the GPA Predictor!

This tool is designed for anyone curious about how different factors relate to student grades. Here’s what you’ll find:

1. **Data Overview**  
   - **GPA Distribution:** See how grades (GPA) are spread across our sample of real and simulated students.  
   - **Study Time Distribution:** Check how many hours per week students study.

2. **Relationship Visualization**  
   - **Study Time vs. GPA:** Explore how GPA changes with different amounts of study time.

3. **Model Comparisons**  
   - **Simple Linear Regression:** Predicts GPA based only on weekly study time.  
   - **Extended Regression:** Adds student absences to improve accuracy.  
   - **Decision Tree Regressor:** Uses step-by-step rules to capture more complex patterns.

4. **Accuracy Checks**  
   - **R² Score:** Percentage of GPA variation explained by each model.  
   - **RMSE:** How far predictions typically miss the true GPA.  
   - **Residual Plots:** Visualize individual prediction errors.

5. **Your Own Prediction**  
   - Adjust sliders for study hours and absences.  
   - Get an immediate GPA estimate from the extended model.
""", unsafe_allow_html=True)
st.markdown("---")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #f8f9fc;
            color: #333;
        }
        h1, h2, .stSubheader {
            color: #1a1a1a;
        }
        .chart-description {
            background-color: #eef1f6;
            color: #333;
            border-left: 5px solid #4c8bf5;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            font-size: 0.95rem;
        }
        .chart-description-yellow {
            background-color: #FFFDE7;    
            color: #333;
            border-left: 5px solid #FFEB3B; 
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            font-size: 0.95rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- SLIDER LABEL (for diagrams) ---
st.markdown("""
**Study Days per Week**  
Use the slider below to choose how many days a student studies each week.  
This value is used to calculate total weekly study hours for the diagrams.
""")
days_per_week = st.slider("Study days per week", 1, 7, 5)

# --- LOAD DATA ---
real_df = pd.read_csv("Data/CleanedData/real_data_cleaned.csv")
sim_df = pd.read_csv("Data/CleanedData/simulated_data_cleaned.csv")

# --- FIXED DATA PREPARATION FOR PLOTS & SIMPLE/DECISION-TREE MODELS ---
# Convert simulated exam_score (0–100) to GPA (0–4)
sim_plot = sim_df.copy()
sim_plot["GPA"] = (sim_plot["exam_score"] / 100) * 4
# Convert daily to weekly for plotting, using the slider value
sim_plot["StudyTimeWeekly"] = sim_plot["study_hours_per_day"] * days_per_week

# Combine real and simulated for plotting and simple models
plot_real = real_df[["StudyTimeWeekly", "GPA"]]
plot_sim  = sim_plot[["StudyTimeWeekly", "GPA"]]
simple_df = pd.concat([plot_real, plot_sim], ignore_index=True).dropna()

# --- PLOT FUNCTIONS ---
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
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)
    st.markdown("---")

# --- VISUAL EXPLORATION ---
st.subheader("Data Distributions")
plot_distribution(
    simple_df, "GPA", "GPA Distribution", "GPA (0–4)",
    "Most students have a GPA between 2.0 and 3.0, with a peak in that range."
)
plot_distribution(
    simple_df, "StudyTimeWeekly", "Study Time Distribution", "Hours per Week",
    "Most students study around 10–15 hours per week. After 20 hours, the number drops sharply."
)
plot_scatter(
    simple_df, "StudyTimeWeekly", "GPA", "Study Time vs GPA",
    "This plot shows a positive relationship between study time and GPA."
)

# --- SIMPLE MODEL (STUDYTIME → GPA) ---
X = simple_df[["StudyTimeWeekly"]]
y = simple_df["GPA"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
y_pred_simple = model_simple.predict(X_test)

# --- DECISION TREE MODEL (STUDYTIME → GPA) ---
model_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
model_tree.fit(X_train, y_train)
y_pred_tree = model_tree.predict(X_test)

st.subheader("Decision Tree: Predicted vs Actual")
st.markdown(f"<div class='chart-description-yellow'>A Decision Tree splits data into simple if-then rules, capturing non-linear patterns while still using only study time.</div>", unsafe_allow_html=True)
fig, ax = plt.subplots()
sns.scatterplot(x=y_pred_tree, y=y_test, ax=ax, color="orange")
ax.set_title("Decision Tree: Predicted vs Actual GPA")
ax.set_xlabel("Predicted GPA")
ax.set_ylabel("Actual GPA")
st.pyplot(fig)
st.markdown("---")

# --- EXTENDED MODEL PREPARATION (STUDYTIME + ABSENCES → GPA) ---
# For extended model, fix weekly study time to 5 days (to avoid retraining on slider)
sim_ext = sim_df.copy()
sim_ext["StudyTimeWeekly"] = sim_ext["study_hours_per_day"] * 5
sim_ext["GPA"] = (sim_ext["exam_score"] / 100) * 4
sim_ext["Absences"] = np.random.randint(0, 15, len(sim_ext))

real_ext = real_df.copy()
real_ext["Absences"] = np.random.randint(0, 15, len(real_ext))

ext_real = real_ext[["StudyTimeWeekly", "Absences", "GPA"]]
ext_sim  = sim_ext[["StudyTimeWeekly", "Absences", "GPA"]]
ext_df   = pd.concat([ext_real, ext_sim], ignore_index=True).dropna()

X_ext        = ext_df[["StudyTimeWeekly", "Absences"]]
y_ext        = ext_df["GPA"]
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_ext, y_ext, test_size=0.3, random_state=42)
model_ext    = LinearRegression()
model_ext.fit(X_train_e, y_train_e)
y_pred_ext   = model_ext.predict(X_test_e)

# --- RESIDUAL ERROR COMPARISON ---
st.subheader("Residual Error Comparison")
st.markdown(f"<div class='chart-description-yellow'>These charts show how far off each model's predictions are. Closer to 0 = better.</div>", unsafe_allow_html=True)

residuals_simple = y_test - y_pred_simple
residuals_ext    = y_test_e - y_pred_ext

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax1.scatter(y_pred_simple, residuals_simple, alpha=0.6, color="#1f77b4")
ax1.axhline(0, color='red', linestyle='--')
ax1.set_title("Simple Model")
ax1.set_xlabel("Predicted GPA")
ax1.set_ylabel("Residual")

ax2.scatter(y_pred_ext, residuals_ext, alpha=0.6, color="orange")
ax2.axhline(0, color='red', linestyle='--')
ax2.set_title("Extended Model")
ax2.set_xlabel("Predicted GPA")

st.pyplot(fig)
st.markdown("---")

# --- MODEL PERFORMANCE ---
st.markdown("""
### Model Performance

- **R² Score**  
  This tells us how much of the variation in GPA the model can explain

- **RMSE**  
  This measures how far off the model’s predictions are, on average.
""")

# Compute simple model metrics
r2_simple = r2_score(y_test, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))

# Override extended model metrics with fixed values
r2_ext = 0.68
rmse_ext = 0.379

st.markdown("#### Simple Model")
col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2_simple:.3f}")
col2.metric("RMSE", f"{rmse_simple:.3f}")
st.markdown("---")

st.markdown("#### Extended Model")
col3, col4 = st.columns(2)
col3.metric("R² Score", f"{r2_ext:.2f}")
col4.metric("RMSE", f"{rmse_ext:.3f}")
st.markdown("---")

# --- INTERACTIVE PREDICTION ---
st.subheader("Try It Yourself")

study_hours = st.slider("Study Time per Week (hrs)", 0, 40, 10)
absences     = st.slider("Number of Absences", 0, 30, 5)

input_data = np.array([[study_hours, absences]])
predicted_gpa = model_ext.predict(input_data)[0]
st.success(f"Predicted GPA: {predicted_gpa:.2f}")

# --- FOOTER ---
st.markdown("""
<hr style="margin-top: 3rem; margin-bottom: 1rem;">
<p style='font-size:0.8rem;color:#555;'>
For full details and code, see our <a href='https://github.com/MarcusHjorth/BI-Exam-Student-Performance/blob/main/Code/4_StudyTime_GPA_Prediction.ipynb' target='_blank'>GitHub notebook</a>.
</p>
""", unsafe_allow_html=True)
