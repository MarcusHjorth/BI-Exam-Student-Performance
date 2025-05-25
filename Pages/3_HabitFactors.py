import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('Data/CleanedData/simulated_data_cleaned.csv')
    df['exam_level'] = df['exam_score'].apply(lambda score:
        'very_low' if score <= 40 else
        'low' if score <= 55 else
        'medium' if score <= 70 else
        'high' if score <= 85 else 'very_high')
    return df

student = load_data()

# Title
st.title("Student Habits Exam Score Analysis")
st.markdown("This app explores factors affecting exam scores using data visualization and a decision tree model.")

# --- INTRO TEXT ---
st.markdown("""

This tool explores how different lifestyle and academic factors influence students’ exam performance, using data visualization and machine learning.

Here’s what you’ll find in this interactive dashboard:


1. **Correlation Analysis**
   - One-hot encoded heatmap to reveal relationships between all features.
   - Interpretation of strongest and weakest correlations with `exam_score`.

2. **Visual Exploration**
   - **Boxplots**: Mental health and exercise vs. exam scores.
   - **Regplots**: Sleep, Netflix, and study hours vs. exam performance.

3. **Score Categorization**
   - Converts numerical exam scores into categories like `very_low`, `medium`, and `very_high`.

4. **Feature Importance**
   - Ranks input variables using ANOVA F-test to determine which factors are most predictive.

5. **Decision Tree Classifier**
   - Predicts exam score categories based on selected features to further see the correlations between some of the stronger features from the heatmap.
   - Full visual representation of the tree logic.

6. **Model Performance**
   - Precision, recall, and F1-scores for each class.
   - Overall accuracy and explanation of model strengths/weaknesses.

---

Whether you're an educator, data enthusiast, or student, this tool helps reveal which habits truly drives academic performance—and where improvement is possible!
""", unsafe_allow_html=True)

st.markdown("---")



# --- Correlation Analysis ---
st.header("1. Correlation Analysis")
st.markdown("""
To start, we use a heatmap to examine how features such as sleep, screen time, and mental health relate to exam scores.
This helps identify strong or weak relationships that can inform later modeling.
""")

student_encoded = pd.get_dummies(student.drop(columns=["student_id"]), drop_first=True)
fig_corr, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(student_encoded.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig_corr)

st.markdown("---")

# --- Visual Exploration ---
st.header("2. Visual Exploration of Student Habits")
st.markdown("""
Below are plots showing how individual habits correlate with exam performance.

Not to any surprising `study_hours_per_day` is the strongest predictor. But other factors like 
`mental_health_rating`, `exercise_frequency`, `attendance_percentage`, `sleep_hours`, and screen time(`social_media_hours`, `netflix_hours`),
did also have a noticeable impact on exam scores.
            
- **Higher `mental_health_ratings`** and **`exercise_frequency`** show a modest boost in scores.
- **More sleep** tends to associate with better results.
- **High screentime usage** show a slight negative trends.
""")


# --- Interactive Habit Comparison ---
st.header("Interactive Habit Comparison")
st.markdown("Use the dropdowns below to explore how two different variables relate to exam scores.")

numeric_cols = ['social_media_hours', 'netflix_hours', 'attendance_percentage', 'sleep_hours']
category_cols = ['mental_health_rating', 'exercise_frequency' ]

col1, col2 = st.columns(2)
with col1:
    x_feature = st.selectbox("Select a feature for the X-axis", numeric_cols + category_cols, index=0)
with col2:
    y_feature = st.selectbox("Select a feature for the Y-axis", ['exam_score'], index=0)

fig, ax = plt.subplots()
if x_feature in category_cols:
    sns.boxplot(x=x_feature, y=y_feature, data=student, ax=ax)
else:
    sns.regplot(x=x_feature, y=y_feature, data=student, ax=ax)

ax.set_title(f"{x_feature.replace('_', ' ').title()} vs {y_feature.title()}")
st.pyplot(fig)

st.markdown("---")


row1 = st.columns(3)
with row1[0]:
    st.markdown("""###### Mental health rating""")
    fig, ax = plt.subplots()
    sns.boxplot(x='mental_health_rating', y='exam_score', data=student, ax=ax)
    ax.set_title("Mental Health vs Exam Score")
    st.pyplot(fig)

with row1[1]:
    st.markdown("""###### Exercise frequency""")
    fig, ax = plt.subplots()
    sns.boxplot(x='exercise_frequency', y='exam_score', data=student, ax=ax)
    ax.set_title("Exercise Frequency vs Exam Score")
    st.pyplot(fig)

with row1[2]:
    st.markdown("""###### Sleep hours""")
    fig, ax = plt.subplots()
    sns.regplot(x='sleep_hours', y='exam_score', data=student, ax=ax)
    ax.set_title("Sleep Hours vs Exam Score")
    st.pyplot(fig)


row2 = st.columns(3)
with row2[0]:
    st.markdown("""###### Netflix hours""")
    fig, ax = plt.subplots()
    sns.regplot(x='netflix_hours', y='exam_score', data=student, ax=ax)
    ax.set_title("Netflix Hours vs Exam Score")
    st.pyplot(fig)

with row2[1]:
    st.markdown("""###### Social media hours""")
    fig, ax = plt.subplots()
    sns.regplot(x='social_media_hours', y='exam_score', data=student, ax=ax)
    ax.set_title("Social Media Hours vs Exam Score")
    st.pyplot(fig)

with row2[2]:
    st.markdown("""###### Attendance percentage""")
    fig, ax = plt.subplots()
    sns.regplot(x='attendance_percentage', y='exam_score', data=student, ax=ax)
    ax.set_title("Attendance Percentage vs Exam Score")
    st.pyplot(fig)



st.markdown("---")


# --- Correlation Values ---
st.header("3. Numeric Correlation with Exam Score")
st.markdown("""
This table ranks how strongly each numeric habit correlates with exam performance.
Study time and attendance show the highest positive correlations.
""")
exam_corr = student.corr(numeric_only=True)['exam_score'].sort_values(ascending=False)
st.dataframe(exam_corr)

st.markdown("---")

# --- Decision Tree ---
st.header("4. Decision Tree Prediction Model")
st.markdown("""
We use a Decision Tree Classifier to categorize students into performance levels like 'very low', 'low', 'medium', 'high' or 'very high' based on their daily habits.

This helps us visualize **which habits the model considers most important** when predicting academic outcomes.
""")

features = [
    'study_hours_per_day', 'social_media_hours', 'netflix_hours',
    'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating'
]
X = student[features]
y = student['exam_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

st.subheader("Tree Logic Visualization")
fig_tree = plt.figure(figsize=(30, 15))
plot_tree(tree, feature_names=features, class_names=tree.classes_, filled=True)
st.pyplot(fig_tree)


st.markdown("---")


# --- Model Performance ---
st.header("5. Model Evaluation")
st.markdown("""
We assess how well the model performs in classifying students.  
While it performs best at detecting 'very high' and 'very low' scores, other categories like 'low' are harder to distinguish.
""")
y_pred = tree.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

st.markdown("---")

# --- Summary ---
st.header("6. Summary & Takeaways")
st.markdown("""
**Key Findings:**
- **Study hours** are the strongest predictor of high exam scores.
- **More sleep** and regular **exercise** are modestly beneficial.
- **Screen time (Netflix, social media)** shows a weak negative impact.
- **Mental health rating** has a small but noticeable effect.

**Model Accuracy:** ~59% — room for improvement, especially with more balanced class distributions.

This tool shows that while no single habit guarantees success, a combination of consistent studying, healthy routines, and lower screen time aligns with higher performance.
""")





