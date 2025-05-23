#  BI-Exam-Student-Performance

###  Brainstorm
After exploring datasets from OpenML, Kaggle, and World Bank Data, we chose two for our project:

- Student Performance Dataset from OpenML
- Student Habits vs Academic Performance from Kaggle

Both contain key academic indicators such as:

- Weekly study time
- Absences
- Parental education
- GPA or equivalent scores

The Kaggle dataset also includes behavioral data like sleep and social media usage — often overlooked factors that we believe are crucial when examining daily habits and academic outcomes.

We aim to:

- Compare patterns between the two datasets
- Investigate how sleep and social media habits influence grades
- Build a machine learning model to predict a student’s likely grade based on study time

Finally, we’re developing a simple chatbot to help answer common questions about our project for users or readers.

---

###  Brief Annotation
This project explores what influences student success in school. Using both real and simulated datasets, we analyze key factors like study time, absences, and parental education, along with sleep habits and social media use and other interesting patterns we discover along the way.

Our goal is to understand which habits have the biggest impact on academic performance and to see if we can predict a student’s grades using machine learning.

The insights from this project can help students, parents, and educators focus on the daily routines and behaviors that truly support better learning outcomes.

---


###  Hypotheses
1.  Students who study more hours per week will achieve higher GPA or exam scores.
(Measured through the studytimeWeekly variable across both datasets.)

2.  Higher absence rates are associated with lower academic performance.
(Analyzed using the absences variable and grade outcomes.)

3. Students with more educated parents perform better academically.
(Based on the ParentalEducation variable.)

---

###  Research for Hypotheses 
To make sure our ideas are based on real facts, we looked for research studies that support what we believe.

1.  Students who study more hours per week will achieve higher GPA or exam scores: [ The Relationship Between Students' Study Time and Academic Performance.](https://pdfs.semanticscholar.org/fdf4/53c5603d7af401953a5eaa49e9ec228d3aaa.pdf)
This study shows a clear positive correlation between weekly study time and academic achievement.

<br>


2.  Higher absence rates are associated with lower academic performance: 
[ School Absences, Academic Achievement, and Adolescents’ Post-School Destinations.](https://doi.org/10.1080/03054985.2024.2308520)
Absenteeism—both truancy and illness—was found to significantly reduce students’ chances of continuing in further education.

<br>

3. Students with more educated parents perform better academically 
[ Parent Involvement and Student Academic Performance: A Multiple Mediational Analysis.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3020099/)
The study found that parental involvement, closely linked to education level, positively influences student outcomes through increased confidence and better student-teacher relationships.


<br> 
These articles help show that our ideas make sense and give us a good starting point for understanding our results.

---



###  Research Questions for project 
1. How do study time, absences, and parental education relate to academic performance across both datasets?

2. To what extent do sleep patterns and social media use impact students' grades? (Kaggle dataset only)

3. Can we predict a student’s likely grade using a machine learning model based on study habits and background factors?

4. Do the key academic patterns found in the simulated dataset align with those in the real-world dataset?

5. Can a simple chatbot help users navigate our findings and better understand which habits influence academic success?

---


### Team Engagement
Each team member is responsible for one or more areas:
- Data cleaning & preparation  
- EDA & visualization  
- Model building & validation  
- Documentation & presentation  

---

###  🛠 Tools & Platforms
- **GitHub** – Version control and documentation  
- **Jupyter Notebooks / VS Code** – Coding and experimentation 
- **ollama and gemma3:4b**  - Chatbot 
- **Data from OpenML and kaggle**

**Link to dataset on OpenML:**  
[Student_Performance_Dataset (ID: 46255)](https://www.openml.org/search?type=data&status=active&id=46255&sort=runs)

**Link to dataset on kaggle:**  
[Student Habits vs Academic Performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance)
