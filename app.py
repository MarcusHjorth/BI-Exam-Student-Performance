import streamlit as st

# --- PAGE SETUP ---
st.set_page_config(page_title="Welcome", layout="centered")
st.title(" BI Exam: Student Performance")

# --- INTRO CONTENT ---
st.markdown("""
##  What’s This About?

This app is part of a final Business Intelligence project from CPHbusiness Group 12.  
We explore **what affects student performance** – using both **real and simulated data**.
            
If this sounds like something you'd like to explore, feel free to look around!
            
Have questions or need help? Try our built-in chatbot or check out the full project notebook on GitHub.
(You’ll find the link at the bottom of each page.)

---

###  Brief Annotation
This project explores what influences student success in school. Using both real and simulated datasets, we analyze key factors like study time, absences, and parental education, along with sleep habits and social media use and other interesting patterns we discover along the way.

Our goal is to understand which habits have the biggest impact on academic performance and to see if we can predict a student’s grades using machine learning.

The insights from this project can help students, parents, and educators focus on the daily routines and behaviors that truly support better learning outcomes.

---

##  Team

This project was built by:

- Sander Marcus Christensen  
- Marcus Hjorth Rasmussen  
- Mateen Jan Rafiq

---

##  Full Details & Code

Want to dig deeper?  
Explore our full notebooks, code, and documentation on GitHub:

👉 [Visit GitHub Repository](https://github.com/MarcusHjorth/BI-Exam-Student-Performance)
""", unsafe_allow_html=True)
