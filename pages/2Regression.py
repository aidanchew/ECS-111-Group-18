import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from data_loader import load_data, get_features_and_conditions


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


st.markdown("<h1 style='text-align: center;'>📈Regression Analysis</h1>", unsafe_allow_html=True)
st.divider()

#Load the cleaned dataset and extract the features and target variables for both regression and classification tasks. 
#The get_features_and_conditions function will return the appropriate feature sets and target variables based on the conditions defined in the data_loader module.
df = load_data()
X_cond1, X_cond2, y_clf, y_reg, y_group = get_features_and_conditions(df)

# Hyperparameter Grids for decision tree and random forest
# define search space for GridSearchCV. 
# linear regression has no meaningful hyperparams to tune, so it's not included here. 
PARAM_GRIDS = {
    "Decision Tree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5],
    },
}

# Cached so tuning does not re-run on every interaction
# underscore prefix on _X_train, _y_train skips Streamlit's hashing for large arrays
@st.cache_resource
def fit_models(condition_name, _X_train, _y_train):
    """Fit all regression models for a given condition, using GridSearchCV
    for Decision Tree and Random Forest. Returns a dict where each key is
    the model name and each value has 'model' (the fitted estimator) and
    'best_params' (the best hyperparameters found, or None for Linear Regression)."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
    }
 
    fitted = {}
    for name, model in models.items(): 
        if name in PARAM_GRIDS:
            search = GridSearchCV(
                model,
                PARAM_GRIDS[name],
                cv = 5,
                # optimizes for RMSE (lead metric)
                scoring = "neg_mean_squared_error",
                n_jobs=-1,
            )
            search.fit(_X_train, _y_train)

            fitted[name] = {
                "model": search.best_estimator_,
                "best_params": search.best_params_,
            }
        else:
            model.fit(_X_train, _y_train)
            fitted[name] = {"model": model, "best_params": None}
 
    return fitted

#This chunk will display the first few rows of the target variables to give us a preview of what we are trying to predict in the regression and classification tasks.

def run_regression(X_data, y_reg, condition_name):
    st.header(condition_name)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_reg, test_size=0.2, random_state=42)
    
    # fit models with tuning (cached)
    fitted = fit_models(condition_name, X_train, y_train)
    
#We will run three different regression models: Linear Regression, Decision Tree Regressor, and Random Forest Regressor. 
#For each model, we will fit it to the training data, make predictions on the test set, and then evaluate the performance using RMSE, MAE, and R² metrics. The results will be displayed in the Streamlit app for easy comparison.
    for name, result in fitted.items():
        model = result["model"]
        preds = model.predict(X_test)
 
        st.subheader(name)
        st.write(f"RMSE: {root_mean_squared_error(y_test, preds):.4f}")
        st.write(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
        st.write(f"R²: {r2_score(y_test, preds):.4f}")
 
        if result["best_params"]:
            st.write("**Best Hyperparameters:**", result["best_params"])

run_regression(X_cond1, y_reg, "Condition 1: All Features Excluding G1/G2")
st.divider()
run_regression(X_cond2, y_reg, "Condition 2: Non-Academic Features Only")