import streamlit as st

# Custom CSS for styled question titles
st.markdown("""
<style>
.question-title {
    font-size: 22px;
    font-weight: 800;
    color: #1f4e79;
    margin-top: 1.5em;
    margin-bottom: 0.3em;
}
</style>
""", unsafe_allow_html=True)

st.title("📘 Answers to Research Questions")
st.markdown("Here we present answers to the research questions explored in this project. Each section includes insights derived from our data analysis.")

# Q1
st.markdown("<div class='question-title'>1. How do study time, absences, and parental education relate to academic performance across both datasets?</div>", unsafe_allow_html=True)
with st.expander("View Answer"):
    st.markdown("""
    **Short Answer:**  
All three factors influence academic performance — but the strength and realism of the patterns vary depending on the dataset.

**Key Findings:**
- **Study time** shows a positive trend in both datasets, but much stronger in the simulated data (r = 0.82 vs. r = 0.18).
- **Absences** have a major negative impact in the real data (r = -0.92), but almost no effect in the simulated dataset.
- **Parental education** shows a mild upward trend, but the correlation is weak in both datasets.

**Conclusion:**  
Study time helps, absences hurt, and parental education plays a role — but only the real dataset reflects how complex and noisy academic performance really is.

---
    """)

# Q2
st.markdown("<div class='question-title'>2. To what extent does different habits impact a students grades *(Kaggle dataset only)*</div>", unsafe_allow_html=True)
with st.expander("View Answer"):
    st.markdown("""
    **Short Answer:**
                
    Not surprisingly, the most important factor, for the best exam score is **study time**. That said, other habits can also play a role.
    factors like **mental health**, **sleep**, **exercise** and **attendance** do all show a positive correlation in some degree,
    while habits **social media** and **Netflix usage** do in fact show a slight negative correlation with exam scores. 

    In genereal, consistent studying combined with healthy daily routines tends to lead to better academic performance.
    """)

# Q3
st.markdown("<div class='question-title'>3. Can we predict a student’s likely grade using a machine learning model based only on weekly study time?</div>", unsafe_allow_html=True)
with st.expander("View Answer"):
    st.markdown("""
    **Short Answer:**  
    Not reliably — using only weekly study time, our model explained just **29%** of GPA variation.  
    Adding factors like **absences** and **parental education** increased accuracy significantly (R² = **68%**).

    **Key Findings:**
    - Study time shows a clear positive trend with GPA, but it’s not enough alone.
    - Including absences and parental background improved prediction power.
    - Linear Regression outperformed Decision Tree due to better stability and interpretability.

    **Conclusion:**  
    A student’s GPA can't be predicted accurately with study time alone — real-life context matters.  
    More variables = better models.
    """)

# Q4
st.markdown("<div class='question-title'>4. Do the key academic patterns found in the simulated dataset align with those in the real-world dataset?</div>", unsafe_allow_html=True)
with st.expander("View Answer"):
    st.markdown("""
    **Short Answer:**  
Partially — some trends are realistic, but others are clearly exaggerated or missing.

**Key Differences:**
- Simulated data makes study time look like a superpower — it’s nearly a straight line to high grades.
- In the real world, even students who study a lot don’t always get top grades.
- The real dataset shows strong effects of absence and more variation in general.

**Conclusion:**  
The simulated dataset is useful for training and testing models, but not for drawing conclusions about real students. Real-world data is messier — and more honest.
    """)

# Q5
st.markdown("<div class='question-title'>5. Can a simple chatbot help users navigate our findings and better understand which habits influence academic success?</div>", unsafe_allow_html=True)
with st.expander("View Answer"):
    st.markdown("""
    **Short Answer:**  
    Yes — while it's far from perfect, our chatbot does a decent job guiding users through our findings and answering questions.

    - **Chatbot Implementation:**  
      We built a simple Q&A-style chatbot based on our project’s insights. It's able to answer common questions about habits like study time, sleep, and absences — and how they relate to academic performance.

    - **User Feedback:**  
      Testers said it was “okay” — it wasn’t always perfect or super detailed, but it usually gave relevant answers. Considering this is only the **second chatbot** we've ever built, we think it turned out pretty well.

    - **Effectiveness:**  
      It’s not the next **ChatGPT** — but for a basic helper built on our own data, it works. Don’t take our word for it though — **try it yourself and see what you think!**
    """)

# Footer
st.markdown(
    """
    <div style='background-color: #f0f2f6; padding: 10px; border-left: 5px solid #1f77b4; border-radius: 4px; margin-top: 30px; color: #333;'>
        This page is part of our final BI exam project.  
        See the full repository for code, data, and model documentation 👉  
        <a href='https://github.com/MarcusHjorth/BI-Exam-Student-Performance/tree/main/Code' target='_blank'>
            GitHub Repository
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
