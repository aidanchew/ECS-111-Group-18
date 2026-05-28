import streamlit as st
import matplotlib.pyplot as plt

from data_loader import load_data

#For consistent styling:
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

/* =========================================================
   1. GLOBAL FONT (Fixed to prevent "keyboard_arrow" bug)
   ========================================================= */
p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    font-family: 'system-ui', sans-serif !important;
    color: white !important;
}

/* =========================================================
   2. UNIFIED TEXTBOX COLORS
   ========================================================= */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="datepicker"] > div {
    background-color: #ffffff !important; 
    border-radius: 8px !important;
    border: 1px solid rgba(0,0,0,0.2) !important;
    box-shadow: none !important;
}

input, textarea, div[data-baseweb="select"] * {
    color: black !important;
}

/* =========================================================
   3. BACKGROUND IMAGE
   ========================================================= */
[data-testid="stAppViewContainer"] {
    background-image: url('https://static.vecteezy.com/system/resources/previews/012/086/236/non_2x/back-to-school-doodles-in-chalkboard-background-free-vector.jpg');
    background-attachment: fixed;
    background-position: center; 
    background-repeat: no-repeat; 
    background-size: cover;  
    background-size: 120%;    

/* =========================================================
   4. SIDEBAR STYLING
   ========================================================= */
    [data-testid="stSidebar"] {
        background-color: #2e3b4e;
        padding: 20px;
    }
    
    [data-testid="stSidebar"] div {
        color: white;
        font-family: sans-serif;
    }
    
    [data-testid="stSidebar"] a {
        color: #ffcc00 !important;
        text-decoration: none;
        font-weight: bold;
    }
}""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>📊Exploratory Data Analysis</h1>", unsafe_allow_html=True)
st.divider()

df = load_data()

# Dataset Overview
st.subheader("Dataset Overview")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head())

st.markdown("""
Our dataset has 382 rows and 57 columns. Each row is one student, with their info pulled from both the Math and Portuguese versions of the UCI Student Performance dataset. The columns cover demographics (school, sex, age, address), family stuff (parental education, jobs, family size), lifestyle (going out, alcohol use, free time, health), and academics (study time, past failures, absences, and grades from all three grading periods).

We also built four target variables on top of the original data. final_gpa is the average G3 across both subjects on the 20-point Portuguese scale. Combined_GPA is the same thing but on a 4.0 scale so its easier to read. Pass_Fail just flags whether a student passed or not. GPA_Group puts students into letter buckets (A, B, C, D/F).

Setting it up this way means we can run the same prediction problem three different ways. Regression on Combined_GPA, pass/fail classification on Pass_Fail, and a multi-class version on GPA_Group. Then we compare which one actually gives us useful results.
""")
st.divider()

# GPA Distribution
st.subheader("GPA Distribution")
fig, ax = plt.subplots()
ax.hist(df['Combined_GPA'], bins=20, edgecolor='black')
ax.set_xlabel("Combined GPA (4.0 scale)")
ax.set_ylabel("Count")
st.pyplot(fig)

st.markdown("""
So when we plotted out the GPAs, we honestly werent that surprised by what we got. Theres a giant spike at 2.3 (54 kids land right there) and the bulk of our students are sitting between like 1.8 and 2.8. Once you get past 3.0 the numbers thin out quick, and on the low end weve only got a couple students under a 1.0.

Basically were working with a dataset full of average kids. We dont have a ton of straight A students or kids who are really failing out, which kinda sucks for modeling. Our models are gonna have plenty of C and B examples to chew on but barely anything to learn from when it comes to the top and bottom of the scale. We already know going in that predicting an A student or a D/F student is gonna be way harder than predicting another C.
""")
st.divider()

# Pass/Fail Balance
st.subheader("Pass/Fail Balance")
st.bar_chart(df['Pass_Fail'].value_counts())

st.markdown("""
The bar chart shows us that 272 of the students passed and 110 had failed. This gives us a distribution of 71% passing and 29% failing. The ratio for this is about 2.5 to 1. This seems like it is quite unbalanced. If we had used accuracy to grade the classifiers, the numbers would be inaccurate. A model would spew out pass every time and still hit the 71% threshold for passing. So for the classification we pulled F1, and ROC-AUC. F1 tends to keep the model honest about the fail group since it has to do well on both sides, and the ROC-AUC is more about if the probabilities actually mean something at the different cutoffs. We figured that between accuracy, F1, and ROC-AUC, we would have enough to tell if a model was real or just faking results.
""")
st.divider()

# GPA Group Balance
st.subheader("GPA Group Balance")
st.bar_chart(df['GPA_Group'].value_counts())

st.markdown("""
The bar chart for GPA groups is much more skewed than the pass-fail one. The C group was the biggest by far with 223 students, B was the second largest with 122, and then the D/F range was at only 31 students. The A range has the least amount of people with 6; this is about 2% of the overall. This causes a few problems for the multi-class classifier. The first issue being that the A group is so small that there is very little to learn from. For precision and recall, A students will have a rough outcome. The second issue is similar to the pass-fail issue: if a model guesses C every time, we would get around 58% accuracy without it learning anything useful. So accuracy on its own would mislead us. This is why it is paired with a weighted F1 for the GPA group. The F1 takes the score from each of the 4 classes and averages them; it adjusts them into each group. This gives a more accurate read on how each model is going across all the GPA buckets instead of grouping most of the students into the C.
""")
st.divider()

# Numeric Feature Correlations
st.subheader("Numeric Feature Correlations with GPA")
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()['Combined_GPA'].sort_values(ascending=False)
st.dataframe(corr)

st.markdown("""
The table above led our group to choose two conditions rather than one. When running correlations, every single one of the top values turned out to be another grade column. G2_mat at 0.90. G1_mat and G3_mat both at 0.88. G2_por at 0.83 and G1_por at 0.79. These numbers are high because G1 and G2 are part of how G3 is calculated. So if we leave them in, the model is just looking at grades to predict more grades. We want to actually see if non school stuff can predict GPA, so for Condition 1 we cut G1 and G2. For Condition 2 we cut every single thing pertaining to academics. After cutting all that out the numbers got way smaller. The top 4 left are mothers education at 0.25, study time in Portuguese at 0.21, study time in Math at 0.21, and fathers education at 0.21. Not great, since nothing even passes 0.25. This means Condition 2 is not gonna have any one feature doing the heavy lifting. The models are going to need to combine a bunch of smaller signals to get anywhere. Failures_mat at -0.44 is the one exception. Past math failures are the biggest non grade signal by a ton. We are betting this shows up first in the SHAP plot for Random Forest. Everything else negative is way smaller. Past Portuguese failures at -0.26. Portuguese absences at -0.19. Travel time between -0.17 and -0.19. Alcohol use on weekdays and weekends sits around -0.14 to -0.16. Going out at -0.15.
""")