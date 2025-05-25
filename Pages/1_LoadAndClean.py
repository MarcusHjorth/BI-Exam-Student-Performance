import streamlit as st
import pandas as pd
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Load & Clean Data", layout="centered")

st.title("Dataset Loading & Cleaning Overview")
st.markdown("""
This page provides a transparent overview of how both the real and simulated student performance datasets were **loaded, cleaned, and preprocessed** before any analysis.  
We perform deduplication, missing value handling, column formatting, outlier filtering, and more.
""")

st.markdown("---")

# --- REAL DATA LOADING & CLEANING ---
st.header("1. Real Dataset Cleaning")

st.markdown("""
We begin by loading a dataset in ARFF-style format (from OpenML) and applying the following steps:

- Skip metadata lines
- Assign column names from documentation
- Remove duplicates and rows with missing values
- Convert numerical columns to proper numeric types
- Filter unrealistic study time values (more than 100 hours/week)
- Convert categorical "yes/no" columns to binary (0/1)
""")

column_names = [
    "StudentID", "Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly",
    "Absences", "Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music",
    "Volunteering", "GPA", "GradeClass"
]

from io import StringIO
with open("Data/RawData/real_data_openML", "r") as f:
    data_lines = [
        line.strip() for line in f
        if line.strip() and not line.startswith(("%", "@"))
    ]
realData = pd.read_csv(StringIO("\n".join(data_lines)), names=column_names)

# Cleaning
realData = realData.drop_duplicates()
realData = realData.dropna()
realData = realData.drop(columns=["StudentID"])

# Convert numeric columns
num_cols = ["Age", "StudyTimeWeekly", "Absences", "GPA"]
for col in num_cols:
    realData[col] = pd.to_numeric(realData[col], errors="coerce")

# Remove outliers
realData = realData[realData["StudyTimeWeekly"] <= 100]

# Encode binary columns
binary_cols = ["Tutoring", "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"]
# Ensure binary/ordinal columns are numeric (already encoded as integers like 0/1/2)
for col in binary_cols:
    realData[col] = pd.to_numeric(realData[col], errors="coerce").astype("Int64")


# Final status
st.markdown("**Preview of Cleaned Real Dataset**")
st.dataframe(realData.head())
st.write("Shape:", realData.shape)
st.write("Missing values:", realData.isnull().sum().sum())
st.write("Duplicate entries:", realData.duplicated().sum())

# Save cleaned data
realData_cleaned_path = "Data/CleanedData/real_data_cleaned.csv"
os.makedirs("Data/CleanedData", exist_ok=True)
realData.to_csv(realData_cleaned_path, index=False)

st.success(f"Cleaned real dataset saved to: `{realData_cleaned_path}`")

st.markdown("---")

# --- SIMULATED DATA CLEANING ---
st.header("2. Simulated Dataset Cleaning")

st.markdown("""
We repeat a similar process for the simulated dataset:

- Remove duplicates and missing rows
- Drop unnecessary columns (like 'StudentID')
- Standardize text formatting for object-type columns
- Convert numeric and binary fields appropriately
- Remove unrealistic study time outliers
""")

simulatedData = pd.read_csv("Data/RawData/simulated_data_kaggle.csv")
simulatedData = simulatedData.drop_duplicates()
simulatedData = simulatedData.dropna()

# Drop ID if present
if "StudentID" in simulatedData.columns:
    simulatedData = simulatedData.drop(columns=["StudentID"])

# Standardize object-type columns
for col in simulatedData.select_dtypes(include='object').columns:
    simulatedData[col] = simulatedData[col].str.strip().str.lower()

# Convert numeric columns
for col in num_cols:
    if col in simulatedData.columns:
        simulatedData[col] = pd.to_numeric(simulatedData[col], errors="coerce")

# Remove outliers
if "StudyTimeWeekly" in simulatedData.columns:
    simulatedData = simulatedData[simulatedData["StudyTimeWeekly"] <= 100]

# Ensure binary/ordinal columns are numeric
for col in binary_cols:
    if col in simulatedData.columns:
        simulatedData[col] = pd.to_numeric(simulatedData[col], errors="coerce").astype("Int64")


# Reset index
simulatedData = simulatedData.reset_index(drop=True)

# Final status
st.markdown("**Preview of Cleaned Simulated Dataset**")
st.dataframe(simulatedData.head())
st.write("Shape:", simulatedData.shape)
st.write("Missing values:", simulatedData.isnull().sum().sum())
st.write("Duplicate entries:", simulatedData.duplicated().sum())

# Save cleaned data
simulatedData_cleaned_path = "Data/CleanedData/simulated_data_cleaned.csv"
simulatedData.to_csv(simulatedData_cleaned_path, index=False)

st.success(f"Cleaned simulated dataset saved to: `{simulatedData_cleaned_path}`")

st.markdown("---")

# --- Summary ---
st.header("3. Summary & Outputs")
st.markdown("""
Both datasets have now been successfully cleaned and exported as CSV files, stored in the `Data/CleanedData/` folder.  
These cleaned versions will be used for further exploration and modeling throughout the application.

This ensures:
- Consistency of column types and naming
- Removal of noise (duplicates, missing, outliers)
- Binary variables are numeric and ready for modeling

Proceed to the analysis page to begin exploring relationships between student habits and academic performance.
""")

st.markdown("---")

st.markdown("### Interactive Filter For The Cleaned Real Dataset")

# GPA range slider
gpa_range = st.slider(
    "Select GPA Range",
    min_value=0.0,
    max_value=4.0,
    value=(0.0, 4.0),
    step=0.1
)

# Gender filter
gender_filter = st.selectbox("Gender", options=["All", "Female (0)", "Male (1)"])

# Parental education filter
parental_edu = st.multiselect(
    "Parental Education Level",
    options=sorted(realData["ParentalEducation"].unique())
)

# Apply filters
filtered_data = realData[
    (realData["GPA"] >= gpa_range[0]) & (realData["GPA"] <= gpa_range[1])
]

if gender_filter == "Female (0)":
    filtered_data = filtered_data[filtered_data["Gender"] == 0]
elif gender_filter == "Male (1)":
    filtered_data = filtered_data[filtered_data["Gender"] == 1]

if parental_edu:
    filtered_data = filtered_data[filtered_data["ParentalEducation"].isin(parental_edu)]

# Show result
st.dataframe(filtered_data)
st.write("Filtered rows:", filtered_data.shape[0])
