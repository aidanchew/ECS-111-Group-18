import streamlit as st
import pandas as pd


#Logo and styling:
    # st.page_link("pages/1EDA.py", label="EDA", icon="📊")
    # st.page_link("pages/2Regression.py", label="Regression", icon="📈")
    # st.page_link("pages/3RidgeAndLasso.py", label="Ridge/Lasso", icon="🔢")
    # st.page_link("pages/4Classification.py", label="Classification", icon="🔍")
    # st.page_link("pages/5Prediction.py", label="Prediction", icon="🔮")'


#Configure Page Settings:
st.set_page_config(layout="centered")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

/*1. GLOBAL FONT*/
/* We target specific text tags but specifically EXCLUDE "span" tags. 
   Streamlit uses spans for Material Icons. If you force Poppins on spans, 
   the icons break and turn into text. */
p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    font-family: 'system-ui', sans-serif !important;
    color: white !important;
}

/*2. UNIFIED TEXTBOX COLORS */
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

/* 3. BACKGROUND IMAGE*/
[data-testid="stAppViewContainer"] {
    background-image: url('https://static.vecteezy.com/system/resources/previews/012/086/236/non_2x/back-to-school-doodles-in-chalkboard-background-free-vector.jpg');
    background-attachment: fixed;
    background-position: center; 
    background-repeat: no-repeat; 
    background-size: cover;  
    background-size: 120%;    

/* 4. SIDEBAR STYLING */

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



#Title:
st.markdown("<h1 style='text-align: center;'>🏡Home Page</h1>", unsafe_allow_html=True)
st.divider()
st.markdown("<h3 style='text-align: center;'><b> Group 18: Failing Kids</b></h3>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Pavish Sanghu, Jinghan Sun, Aidan Chew, Alexander Cai</h5>", unsafe_allow_html=True)
st.markdown("")
st.divider()

#Mission Statement:
st.markdown("<h3 style='text-align: center;'><b> Mission Statement</b></h3>", unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center; font-style: italic; '>Allow a user to predict a their final grade (G3) using historical data from the 'Student Performance' dataset</h5>", 
    unsafe_allow_html=True)
st.divider()

#Spacing:
st.markdown("")
st.markdown("")
st.markdown("")

#Intoduction:
with st.container(border=True):
    st.subheader("Introduction")
    st.markdown("A student’s academic performance is one of the most important assets of an individual. It defines someone’s commitment to school, academic intelligence, and, most importantly, their future. Grades are the defining factor for college administrations, which means a student’s academic ability in Middle/High school is incredibly important.")
    st.markdown("While a student’s grade can often be attributed to the amount of time studying or paid tutoring lessons, it’s more complicated than academic factors. Personal habits and lifestyles can affect a student’s performance. Family life, interests, and accessibility to resources can all alter how well a child may score, muddying what should’ve been a simple correlation (grade|studying)")
    st.markdown("Therefore, our goal is to determine whether it’s possible to predict students' scores, accounting for all external factors. This process is solved using various machine learning models taught in the UC Davis course, ECS 111 [Applied Machine Learning for Non-Majors]. ")

#Dataset:
with st.container(border=True):
    st.subheader("Dataset")
    st.markdown("To ensure data authenticity and reliability, we sourced our dataset from our fellow University of California, UC Irvine. Their machine learning repository is filled with quality dataset that have been authorized before being published. Our project used the 'Student Performance' dataset, containing information on the academic achievements of two Portugese secondary schools.")
    st.markdown("The grades of each grading period are based on both Portugese (English equivalent) and Math. Additionally, final grades are based on both G1 and G2, which are the first two grading periods.")
    st.markdown("Below is a brief summary of the dataset, which is categorized into four main subcategories: Demographics, Personal Information, Habits/Lifestyle, and Academic Performance.")
    records, features, target = st.columns(3)
    records.metric("Total Records", "649")
    features.metric("Features", "30")
    target.metric("Target Variable", "Grade (G3)")

#Summary of the dataset:
summary = pd.DataFrame({
    'Subcategory': ['Demographics','Personal Information', 'Habits/Lifestyle','Academic Performance'],
    'Features': [2, 10, 18, 3],
    'Description': ['Characteristics of a student', 'Details about a student\'s background', 'Daily habits and living conditions', 'Grades and performance metrics of a student [Target Variable]'],
    'Dimensions': [
    'Sex, Age', 
    'School, Address, Family size, Parental status, Parents education level, Parents job, Guardian, Education support, Health', 
    'Reason for school, Travel time, Study time, Failures, Extra educational support [schoolsup], Family support, Extra paid classes [paid], Activities, Nursery, Interest in college [higher], Internet, Romantic, Family relationships, Freetime, Going out, Alcohol consumption (weekday/weekend), Absences', 
    'G1, G2, G3']
})



st.dataframe(summary,
             column_config={
                 'Subcategory': st.column_config.TextColumn("Subcategory", width="medium"),
                 'Features': st.column_config.TextColumn("Features", width="small"),
                 'Description': st.column_config.TextColumn("Description", width="large"),
                 'Dimensions': st.column_config.TextColumn("Dimensions", width="large")
                },
              hide_index=True,
              use_container_width=True)


#Link to the dataset:
st.info("📩 **Original Resource:** [Access the original Student Performance dataset on UCI's ML Repository](https://archive.ics.uci.edu/dataset/320/student+performance)")
st.markdown("")



#Pages:
with st.container(border=True):
    st.subheader("Pages")
    st.markdown("Our project has been split into six main pages. Each page focuses on a specific aspect of our analysis and modeling process:")
    with st.expander("Click here to see the different pages of our project"):
        st.markdown("- **📊EDA**: Explores the dataset through visualizations with brief analysis to identify trends.")
        st.markdown("- **📈Regression**: Builds and evaluates various regression models to predict student performance.")
        st.markdown("- **🔢Ridge and Lasso**: Further explores regression techniques by implementing Ridge and Lasso regression to (potentially) enhance model performance and interpretability.")
        st.markdown("- **🔍Pass/Fail**: Runs Logistic Regression and Random Forest to predict whether a student passes or fails.")
        st.markdown("- **🎓GPA Groups**: Runs Logistic Regression and Random Forest to predict which letter grade group (A, B, C, D/F) a student lands in.")
        st.markdown("- **🔮Prediction**: Uses the best performing model to predict a student's grade based on user input.")

# st.markdown("Our project has been split into five main pages. Each page focuses on a specific aspect of our analysis and modeling process:")
# st.markdown("- **EDA**: Explores the dataset through visualizations and brief statsitical analysis to identify trends and patterns.")
# st.markdown("- **Regression**: Builds and evaluates various regression models to predict student performance.")
# st.markdown("- **Ridge and Lasso**: Further explores regression techniques by implementing Ridge and Lasso regression to (potentially) enhance model performance and interpretability.")
# st.markdown(" - **Classification**: Runs Logistic Regression and Random Forest Classifier to predict a student's grade.")
# st.markdown("- **Prediction**: Uses the best performing model to finally predict a student's grade based on user input.")




# links = [
#     {"label": "EDA", "icon": "📊", "page": "pages/1EDA.py"},
#     {"label": "Regression", "icon": "📈", "page": "pages/2Regression.py"},
#     {"label": "Ridge/Lasso", "icon": "ℹ️", "page": "pages/3RidgeAndLasso.py"},
#     {"label": "Classification", "icon": "🔍", "page": "pages/4Classification.py"},
#     {"label": "Prediction", "icon": "🔮", "page": "pages/5Prediction.py"}
# ]
st.markdown("")
st.markdown("<h4 style='text-align: center;'>For more information, click on the links below:</h4>", unsafe_allow_html=True)
cols = st.columns(6)

with cols[0]: st.page_link("pages/1EDA.py", label="EDA", icon="📊")
with cols[1]: st.page_link("pages/2Regression.py", label="Regression", icon="📈")
with cols[2]: st.page_link("pages/3RidgeAndLasso.py", label="Ridge/Lasso", icon="🔢")
with cols[3]: st.page_link("pages/4PassFail.py", label="Pass/Fail", icon="🔍")
with cols[4]: st.page_link("pages/5GPAGroups.py", label="GPA Groups", icon="🎓")
with cols[5]: st.page_link("pages/6Prediction.py", label="Prediction", icon="🔮")


# st.page_link("pages/1EDA.py", label="Data Exploration", icon="📊")
# st.page_link("pages/2Regression.py", label="Regression Models", icon="📈")
# st.page_link("pages/3RidgeAndLasso.py", label="Ridge and Lasso Regression", icon="ℹ️")
# st.page_link("pages/4Classification.py", label="Classification Models", icon="🔍")
# st.page_link("pages/5Prediction.py", label="Prediction", icon="🔮")
# if st.button("Click to go to Data Page"):
#     st.switch_page("pages/1EDA.py")