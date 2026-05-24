import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


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



from data_loader import load_data, get_features_and_conditions

st.markdown("<h1 style='text-align: center;'>🔢Ridge and Lasso Regression</h1>", unsafe_allow_html=True)
st.divider()

#We will load the dataset and separate it into the feature sets and target variables
df = load_data()
X_cond1, X_cond2, y_clf, y_reg, y_group = get_features_and_conditions(df)

#We will define a function that runs Ridge and Lasso regression for a given condition
#This lets us reuse the same code for both feature sets
def run_ridge_lasso(X_data, y_reg, condition_name):
    st.header(condition_name)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_reg, test_size=0.2, random_state=42)

#Ridge
    #We will train a Ridge regression model and use it to predict GPA on the test data
    ridge_model = RidgeCV(cv=5)
    ridge_model.fit(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)

    st.subheader("Ridge Regression")
    st.write(f"RMSE: {root_mean_squared_error(y_test, ridge_preds):.4f}")
    st.write(f"MAE: {mean_absolute_error(y_test, ridge_preds):.4f}")
    st.write(f"R²: {r2_score(y_test, ridge_preds):.4f}")

# Lasso
    #We will train a Lasso regression model and use it to predict GPA on the test data
    lasso_model = LassoCV(cv=5, max_iter=10000)
    lasso_model.fit(X_train, y_train)
    lasso_preds = lasso_model.predict(X_test)

    st.subheader("Lasso Regression")
    st.write(f"RMSE: {root_mean_squared_error(y_test, lasso_preds):.4f}")
    st.write(f"MAE: {mean_absolute_error(y_test, lasso_preds):.4f}")
    st.write(f"R²: {r2_score(y_test, lasso_preds):.4f}")

#We will run the Ridge and Lasso models on both feature conditions
#Condition 1 uses all features except G1/G2, while Condition 2 uses only non-academic features
run_ridge_lasso(X_cond1, y_reg, "Condition 1: All Features Excluding G1/G2")
st.divider()
run_ridge_lasso(X_cond2, y_reg, "Condition 2: Non-Academic Features Only")