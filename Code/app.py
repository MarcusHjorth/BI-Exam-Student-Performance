import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

st.set_page_config(page_title="MACHINE LEARNING FOR ANALYSIS AND PREDICTION ")

# Sidebar
st.sidebar.title("Exams project skrrt")
page = st.sidebar.radio("Select a task", [
    "1. Intro to project",
    "2. Compare Data sets",
    "3. Fun Variables",
    "4. Predict Exam Score",
    "5. Answer to tasks questions"
])

if page == "1. Intro to project":
    st.title("Intro to project")


if page == "2. Compare Data sets":
    st.title("Compare Data sets")


if page == "3. Fun Variables":
    st.title("Fun Variables")


if page == "4. Predict Exam Score":
    st.title("Predict Exam Score")

if page == "5. Chatbot":
    st.title("Chatbot")
