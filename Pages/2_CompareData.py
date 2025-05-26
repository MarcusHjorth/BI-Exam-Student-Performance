import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set visual style
sns.set(style="whitegrid")

st.set_page_config(page_title="Student Performance Comparison", layout="wide")
st.title("📊 Student Performance Comparison: Real vs Simulated Data")

# Load datasets
@st.cache_data

def load_data():
    real_df = pd.read_csv("Data/CleanedData/real_data_cleaned.csv")
    sim_df = pd.read_csv("Data/CleanedData/simulated_data_cleaned.csv")

    sim_df['StudyTimeWeekly'] = sim_df['study_hours_per_day'] * 7
    sim_df['Absences'] = (100 - sim_df['attendance_percentage']) / 100 * 180
    edu_map = {
        'no formal': 0,
        'primary school': 1,
        'high school': 2,
        'bachelor': 3,
        'master': 4,
        'phd': 5
    }
    sim_df['ParentalEducation'] = sim_df['parental_education_level'].map(edu_map)

    real_df = real_df[['StudyTimeWeekly', 'Absences', 'ParentalEducation', 'GPA']]
    sim_df = sim_df[['StudyTimeWeekly', 'Absences', 'ParentalEducation', 'exam_score']]

    return real_df, sim_df

real, sim = load_data()

# Sidebar selection
plot_type = st.sidebar.selectbox("Select analysis:", [
    "Study Time vs Performance",
    "Absences vs Performance",
    "Parental Education vs Performance",
    "Correlation Heatmaps",
    "Barplots (Grouped Averages)"
])

# Plot functions
def plot_scatter_side_by_side(real_df, sim_df, x, y_real, y_sim, xlabel, ylabel):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real Dataset**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=real_df, x=x, y=y_real, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        st.pyplot(fig)
    with col2:
        st.markdown("**Simulated Dataset**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=sim_df, x=x, y=y_sim, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        st.pyplot(fig)

def plot_bar_side_by_side(real_df, sim_df, group_col, value_real, value_sim, xlabel, ylabel):
    avg_real = real_df.groupby(group_col)[value_real].mean().reset_index()
    avg_sim = sim_df.groupby(group_col)[value_sim].mean().reset_index()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Real Dataset**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=avg_real, x=group_col, y=value_real, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        st.pyplot(fig)
    with col2:
        st.markdown("**Simulated Dataset**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=avg_sim, x=group_col, y=value_sim, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        st.pyplot(fig)

def plot_heatmap(df, title, method):
    fig, ax = plt.subplots(figsize=(6, 4))
    corr = df.corr(method=method)
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(f"{title} (method: {method})")
    st.pyplot(fig)

# Study time slider
study_range = st.sidebar.slider("Select Study Time Range (hours/week)", 0, 60, (0, 60))
real_filtered = real[(real["StudyTimeWeekly"] >= study_range[0]) & (real["StudyTimeWeekly"] <= study_range[1])]
sim_filtered = sim[(sim["StudyTimeWeekly"] >= study_range[0]) & (sim["StudyTimeWeekly"] <= study_range[1])]

# Render selected view
if plot_type == "Study Time vs Performance":
    st.subheader("Study Time vs Academic Performance")
    plot_scatter_side_by_side(real_filtered, sim_filtered, "StudyTimeWeekly", "GPA", "exam_score", "Study Time (hours/week)", "Academic Performance")

elif plot_type == "Absences vs Performance":
    st.subheader("Absences vs Academic Performance")
    plot_scatter_side_by_side(real_filtered, sim_filtered, "Absences", "GPA", "exam_score", "Absences (days)", "Academic Performance")

elif plot_type == "Parental Education vs Performance":
    st.subheader("Parental Education vs Academic Performance")
    plot_scatter_side_by_side(real_filtered, sim_filtered, "ParentalEducation", "GPA", "exam_score", "Parental Education Level", "Academic Performance")

elif plot_type == "Correlation Heatmaps":
    method = st.selectbox("Choose correlation method", ["pearson", "spearman", "kendall"])
    if method != "pearson":
        st.markdown("""
        ⚠️ **Important Reminder:** Only **Pearson correlation** was used in our actual analysis and conclusions.  
        **Spearman** and **Kendall** are included here **just for exploration** and were **not used** to draw any final insights.
        """)
    st.subheader("Correlation Heatmap: Real Dataset")
    plot_heatmap(real, "Real Dataset Correlation", method)
    st.subheader("Correlation Heatmap: Simulated Dataset")
    plot_heatmap(sim, "Simulated Dataset Correlation", method)

elif plot_type == "Barplots (Grouped Averages)":
    bar_choice = st.radio("Select variable for grouped barplot:", ["Study Time", "Absences", "Parental Education"])

    if bar_choice == "Study Time":
        st.subheader("Barplot: Study Time Group vs Academic Performance")
        real_filtered['StudyTimeGroup'] = pd.cut(real_filtered['StudyTimeWeekly'], bins=[0, 5, 10, 15, 20], labels=['0-5', '5-10', '10-15', '15-20'])
        sim_filtered['StudyTimeGroup'] = pd.cut(sim_filtered['StudyTimeWeekly'], bins=[0, 10, 20, 30, 40, 60], labels=['0-10', '10-20', '20-30', '30-40', '40-60'])
        plot_bar_side_by_side(real_filtered, sim_filtered, "StudyTimeGroup", "GPA", "exam_score", "Study Time Group", "Average Academic Performance")

    elif bar_choice == "Absences":
        st.subheader("Barplot: Absence Group vs Academic Performance")
        real_filtered['AbsenceGroup'] = pd.cut(real_filtered['Absences'], bins=[0, 5, 10, 20, 40, 80], labels=['0-5', '6-10', '11-20', '21-40', '40+'])
        sim_filtered['AbsenceGroup'] = pd.cut(sim_filtered['Absences'], bins=[0, 5, 10, 20, 40, 80], labels=['0-5', '6-10', '11-20', '21-40', '40+'])
        plot_bar_side_by_side(real_filtered, sim_filtered, "AbsenceGroup", "GPA", "exam_score", "Absence Group", "Average Academic Performance")

    elif bar_choice == "Parental Education":
        st.subheader("Barplot: Parental Education vs Academic Performance")
        plot_bar_side_by_side(real_filtered, sim_filtered, "ParentalEducation", "GPA", "exam_score", "Parental Education Level", "Average Academic Performance")

st.markdown("---")
st.caption("Created by Group 12 – Business Intelligence, CPHbusiness")
