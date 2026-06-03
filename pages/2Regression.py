import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# For consistent styling:
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    font-family: 'system-ui', sans-serif !important;
    color: white !important;
}

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

[data-testid="stAppViewContainer"] {
    background-image: url('https://static.vecteezy.com/system/resources/previews/012/086/236/non_2x/back-to-school-doodles-in-chalkboard-background-free-vector.jpg');
    background-attachment: fixed;
    background-position: center; 
    background-repeat: no-repeat; 
    background-size: cover;  
    background-size: 120%;    

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


from data_loader import load_data, get_features_and_conditions

st.markdown("<h1 style='text-align: center;'>📈 Regression Models</h1>", unsafe_allow_html=True)
st.divider()

# Intro paragraph explaining models and metrics
st.header("Introduction")
st.markdown("""
Linear Regression is the “backbone” of most machine learning models. Therefore, it is naturally the first subset of models we will attempt. We used 3 different algorithms: **Linear Regression**, **Decision Trees**, and **Random Forest**. Together, these 3 models will be used to predict the final GPA on a 4.0 scale. 

Additionally, we will use **RMSE**, **MAE**, and **R²** to evaluate these models. 
""")
st.write("")


with st.expander("Click here to see the full detailed introduction:"):
    st.markdown("""
For the Regression page we ran 3 different models to predict the Combined GPA as a continuous number on the 4.0 scale. Linear Regression is used as our baseline. It assumes there is a straight line relationship between the features and GPA which is rarely true in real data but it gives us something to compare against. Decision Tree splits the data into branches based on feature thresholds, which lets it capture non-linear relationships and feature interactions that Linear Regression cannot. The downside is that a single tree tends to overfit. Random Forest fixes that by training many decision trees on slightly different samples of the data and averaging their predictions. That averaging step usually reduces variance and gives better generalization than a single tree.

For the metrics we use RMSE, MAE, and R². RMSE is the average prediction error in GPA points with big misses punished harder because the errors get squared. MAE is the plain average of how off our predictions are without the squaring. R² is the percent of GPA variance the model can explain, so it tells us how much of the pattern in grades the model is actually capturing. Lower is better for RMSE and MAE, higher is better for R².
                    """)
st.divider()

# Load dataset and split into features and targets
df = load_data()
X_cond1, X_cond2, y_clf, y_reg, y_group = get_features_and_conditions(df)


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


@st.cache_resource
def fit_models(condition_name, _X_train, _y_train):
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
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            search.fit(_X_train, _y_train)
            fitted[name] = {"model": search.best_estimator_, "best_params": search.best_params_}
        else:
            model.fit(_X_train, _y_train)
            fitted[name] = {"model": model, "best_params": None}

    return fitted


# Define a function that runs all three regression models for a given condition
def run_regression(X_data, y_reg, condition_name):
    st.header(condition_name)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_reg, test_size=0.2, random_state=42)
    fitted = fit_models(condition_name, X_train, y_train)

    for name, result in fitted.items():
        model = result["model"]
        preds = model.predict(X_test)

        st.subheader(name)
        st.write(f"RMSE: {root_mean_squared_error(y_test, preds):.4f}")
        st.write(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
        st.write(f"R²: {r2_score(y_test, preds):.4f}")

        if result["best_params"]:
            st.write("**Best Hyperparameters:**", result["best_params"])

    # Condition specific interpretation paragraph
    if "Condition 1" in condition_name:
        st.write("")
        st.subheader("📋Analysis")
        st.markdown("""
**Key Insight**: The three models showed a clear gap between linear and tree based approaches. 

1) **Linear Regression had the weakest R²** of the three because the relationship between features and GPA is not actually a straight line. 

2) **Decision Tree did better** at picking up non-linear patterns but its R² was held back by overfitting since a single tree memorizes the training data and struggles to generalize. 

3) **Random Forest was the best performer** here with the lowest RMSE and the highest R². That makes sense because averaging across many trees smooths out the overfitting that hurt the Decision Tree on its own. 

**Takeaway:**
The fact that the tree based models beat Linear Regression tells us there are non-linear relationships and feature interactions in the data that a straight line cannot capture. 
""")
    else:
        st.write("")
        st.subheader("📋Analysis")
        st.markdown("""
**Key Insight:** Every model dropped in performance once we removed the academic features like study time, failures, and absences. 

1) The drop confirms that **academic behavior** is the strongest signal for predicting GPA and that demographic or family features alone do not carry enough information. 

2) **Random Forest still came out as the best**, but its R² fell significantly compared to Condition 1, which means even the most flexible model cannot make up for missing the most predictive features.


**Takeaway**: Non-academic features have some predictive value but they are not enough on their own to reliably predict final GPA. 

""")


# Run the regression models on both feature conditions
run_regression(X_cond1, y_reg, "Condition 1: All Features Excluding G1/G2")
st.divider()
run_regression(X_cond2, y_reg, "Condition 2: Non-Academic Features Only")

#Next Page:
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    st.page_link("pages/3RidgeAndLasso.py", label="Next Page", icon="▶️", use_container_width="content")