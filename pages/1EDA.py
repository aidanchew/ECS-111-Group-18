import streamlit as st
import matplotlib.pyplot as plt

from data_loader import load_data

#For consistent styling:
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

/*1. GLOBAL FONT */
/* We target specific text tags but specifically EXCLUDE "span" tags. 
   Streamlit uses spans for Material Icons. If you force Poppins on spans, 
   the icons break and turn into text. */
p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    font-family: 'system-ui', sans-serif !important;
    color: white !important;
}

/*2. UNIFIED TEXTBOX COLORS*/
/* Forces every single input type to be the exact same pure white */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="datepicker"] > div {
    background-color: #ffffff !important; 
    border-radius: 8px !important;
    border: 1px solid rgba(0,0,0,0.2) !important;
    box-shadow: none !important;
}

/* Ensures text typed inside the boxes is black */
input, textarea, div[data-baseweb="select"] * {
    color: black !important;
}

/*3. BACKGROUND IMAGE */
[data-testid="stAppViewContainer"] {
    background-image: url('https://static.vecteezy.com/system/resources/previews/012/086/236/non_2x/back-to-school-doodles-in-chalkboard-background-free-vector.jpg');
    background-attachment: fixed;
    background-position: center; 
    background-repeat: no-repeat; 
    background-size: cover;      

            
/*4. SIDEBAR STYLING*/
/* Target the sidebar container */
    [data-testid="stSidebar"] {
        background-color: #2e3b4e; /* Change to your brand color */
        padding: 20px;
    }
    
    /* Style the text inside the sidebar */
    [data-testid="stSidebar"] div {
        color: white;
        font-family: sans-serif;
    }
    
    /* Style sidebar links */
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
Our newly cleaned dataset has 382 rows and 57 columns. These columns include:
- **final_gpa**: the average G3 across both subjects on the 20-point Portuguese scale. 
- **Combined_GPA**: Same as final_gpa, but now on a 4.0 scale, so it's easier to read
- **Pass_Fail**: Flags whether a student passed or not.
- **GPA_Group**: puts students into letter buckets (A, B, C, D/F).

Setting it up this way means we can run the same prediction problem three different ways:
1) **Regression** on Combined_GPA
2) **Pass/fail classification** on Pass_Fail, 
3) **A multi-class version** on GPA_Group. 

**Then we compare which one actually gives us useful results**
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
When plotting GPA’s, we weren’t surprised by the results. A **majority of kids** landed **between 1.8 and 2.8** (most at 2.3 [54 kids]), and only a few scored past 3.0 and below 1.0. 
            
Because the dataset takes a real sample of two schools, naturally, there is a **large class imbalance** between Excellent/Failing Students vs Passing ones. Therefore, (Taking lessons from our 2nd homework), we know **predicting an A student or a D/F student is gonna be way harder than predicting a C student.**
""")
st.divider()

# Pass/Fail Balance
st.subheader("Pass/Fail Balance")
st.bar_chart(df['Pass_Fail'].value_counts())

st.markdown("""
The bar chart shows **71% of students passed** while **29% failed**. Again, this is another imbalancing issue we will need to consider. 
            
Therefore, for our **classification task**, we will use **F1 and ROC-AUC**. F1 tends to keep the model honest about the fail group since it has to do well on both sides, and the ROC-AUC is more about if the probabilities actually mean something at the different cutoffs.
""")
st.divider()

# GPA Group Balance
st.subheader("GPA Group Balance")
st.bar_chart(df['GPA_Group'].value_counts())

st.markdown("""
The bar chart for GPA groups is much more skewed than the pass-fail one. Similar to the GPA Distribution earlier, the B/C groups dominate the overall distribution, with only **8% of the students being accounted for D/F grades**, and a measly **2% accounting for A grades.** 

This causes a few problems for the multi-class classifier. 
1) **A group is so small** that there is **very little to learn from.** For precision and recall, A students will have a rough outcome. 
2) If a model guesses C every time, **we would get around 58% accuracy** without it learning anything useful.

This is why we will **prioritize weighted F1** for the GPA group. The F1 takes the score from each of the 4 classes and averages them; it adjusts them into each group. This gives a more accurate read on how each model is going across all the GPA buckets instead of grouping most of the students into the C.
""")
st.divider()

# Numeric Feature Correlations
st.subheader("Numeric Feature Correlations with GPA")
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()['Combined_GPA'].sort_values(ascending=False)
st.dataframe(corr)

st.markdown("""
Using the table above, we came to an important conclusion: **we will be using two conditions for each predictive model**

When running correlations, every single one of the top values turned out to be another grade column. **The grading period + subject numbers are far too correlated with G3**, a warning we received before even loading the dataset. Because we want to actually see if non-school stuff can predict GPA, we will perform the following conditions on each model:
1) Condition 1 will cut G1 and G2. 
2) Condition 2 will cut every single variable related to academics.

After cutting all that out, the correlations became way smaller. Because almost all conditions fail to pass a correlation of 0.25(except Failures_mat at -0.44 [likely to show on SHAP plot]), **condition 2 will not have a single feature that does all of the “heavy lifting”**.

""")

#Next Page:
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    if st.button("▶️Next Page", type="secondary", use_container_width=True):
        st.switch_page("pages/2Regression.py")