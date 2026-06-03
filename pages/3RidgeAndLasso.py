import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from data_loader import load_data, get_features_and_conditions

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



st.markdown("<h1 style='text-align: center;'>🔢Ridge and Lasso Regression</h1>", unsafe_allow_html=True)
st.divider()

# Intro paragraph explaining models and metrics
st.header("Introduction")
st.markdown("""
The following page runs **Ridge** and **Lasso** Regression to predict a student’s final GPA. Both methods can be almost seen as modified linear regression, adding a penalty to prevent coefficients from overfitting. 

We will use RidgeCV and LassoCV to pick the best regularization strength, using 5-fold cross-validation. The models will be evaluated using **RMSE**, **MAE**, and **R²**, similar to the previous Linear Regression Page. 
""")

with st.expander("Click here to see the full detailed introduction:"):
    st.markdown("""
For the Ridge and Lasso page we ran 2 regularized regression models to predict the Combined GPA. Both of these are basically Linear Regression but with a penalty added to keep the coefficients from getting too large. The point of the penalty is to stop the model from overfitting when there are a lot of features that might not all matter. Ridge uses an L2 penalty which shrinks the coefficients toward zero but never actually sets any of them to zero, so it keeps every feature in the model just at a smaller weight. Lasso uses an L1 penalty which can shrink coefficients all the way to zero, so it ends up dropping features entirely and acting like a built in feature selector. We used RidgeCV and LassoCV which automatically pick the best regularization strength using 5 fold cross validation.
                
For the metrics we are looking at the same RMSE, MAE, and R² as the regression page. RMSE is the average prediction error in GPA points with big misses punished harder. MAE is the plain average of how off our predictors are. R² is the percent of GPA variance the model can explain. Lower is better for RMSE and MAE, higher is better for R².
  
                """)

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

    # condition specific interpretation paragraph
    if "Condition 1" in condition_name:
        st.write("")
        st.subheader("📋Analysis")
        st.markdown("""
**Key Insight**: Ridge and Lasso came out almost identical.

1) Ridge and Lasso had **similar RMSE/MAE and R².**

2) Both beat regular Linear Regression (OLS: R²= 0.12) 

3) The fact that Ridge and Lasso are this close to each other tells us Lasso did not need to drop any features to win, which means **most of the features are at least pulling a little weight.**

**Takeaway:** Predictions are still off by around 0.42 to 0.53 grade points and these models are explaining around 18% of the variance.

""")
    else:
        st.write("")
        st.subheader("📋Analysis")
        st.markdown("""
**Insight:** Ridge and Lasso stayed close to each other again

1) The drop from Condition 1 to Condition 2 was way smaller here than it was for Random Forest.

2) The regularized models are more stable across conditions, but they also have a lower ceiling overall. 

3) Regularized Models explain about **11%  of the variance**, which is not great, and both of them are still beaten by the Random Forest in Condition 2 at 0.14.

**Takeaway:** Regularization helps these linear models avoid overfitting, but it cannot manufacture a signal that the features do not have. 

""")

#We will run the Ridge and Lasso models on both feature conditions
#Condition 1 uses all features except G1/G2, while Condition 2 uses only non-academic features
run_ridge_lasso(X_cond1, y_reg, "Condition 1: All Features Excluding G1/G2")
st.divider()
run_ridge_lasso(X_cond2, y_reg, "Condition 2: Non-Academic Features Only")

#Next Page:
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    if st.button("▶️Next Page", type="secondary", use_container_width=True):
        st.switch_page("pages/4PassFail.py")