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
/* We target specific text tags but specifically EXCLUDE "span" tags. 
   Streamlit uses spans for Material Icons. If you force Poppins on spans, 
   the icons break and turn into text. */
p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    font-family: 'system-ui', sans-serif !important;
    color: white !important;
}

/* =========================================================
   2. UNIFIED TEXTBOX COLORS
   ========================================================= */
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

#This chunk displays the shape of the dataset and then tells us how many rows and cols are in there
#Then we check the first 5 rows to ensure that the data looks correct
st.subheader("Dataset Overview")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head())
st.divider()

#Will create a histrogram and a mathplotlib figure 
#Will help show the distribution of the target variable (GPA) 
# Then it will push it into streamlit
st.subheader("GPA Distribution")
fig, ax = plt.subplots()
ax.hist(df['Combined_GPA'], bins=20, edgecolor='black')
ax.set_xlabel("Combined GPA (4.0 scale)")
ax.set_ylabel("Count")
st.pyplot(fig)
st.divider()


st.subheader("Pass/Fail Balance")
st.bar_chart(df['Pass_Fail'].value_counts())
st.divider()

st.subheader("GPA Group Balance")
st.bar_chart(df['GPA_Group'].value_counts())
st.divider()

#Select the numeric cols from the dataset 
#Then we will calculate the correlation of each numeric feature with the target variable (GPA) and display it in a table
st.subheader("Numeric Feature Correlations with GPA")
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()['Combined_GPA'].sort_values(ascending=False)
st.dataframe(corr)