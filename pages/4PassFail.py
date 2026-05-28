import streamlit as st
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_loader import load_data, get_features_and_conditions


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


# Title
st.markdown("<h1 style='text-align: center;'>🔍Pass/Fail Classification</h1>", unsafe_allow_html=True)
st.divider()

# Intro paragraph
st.markdown("""
For the Pass/Fail page we ran 2 classification models to predict whether a student passes or fails. Logistic Regression sounds like a regression model but it is actually a classifier. It takes the features and runs them through a sigmoid function to spit out a probability between 0 and 1 of the student passing. We tuned the C hyperparameter which controls how much regularization gets applied, smaller C means more regularization and larger C means less. Random Forest is the same model from the regression page, just used for classification this time where the trees vote on a class instead of averaging numbers. We tuned n_estimators, max_depth, max_features, and min_samples_split.

For the metrics we are looking at Accuracy, F1 Score, ROC AUC, and Cross-Val F1. Accuracy is just the percent of correct predictions but it can be misleading when the classes are imbalanced like ours are. F1 Score balances precision and recall, which is why it is more honest than accuracy when one class is bigger than the other. ROC AUC measures how well the model separates the two classes across all probability thresholds, 0.5 is random guessing and 1.0 is perfect. Cross-Val F1 is the F1 score averaged across the 5 GridSearchCV folds during tuning, which tells us if the model is consistent or just got lucky on the test set.

We also included a SHAP feature importance plot under each Random Forest. SHAP shows which features had the biggest impact on the model's predictions by calculating how much each feature pushed the output up or down. The EDA showed us that failures_mat had the strongest non-grade correlation with GPA, so we expect that feature to show up high on the SHAP plot.
""")
st.divider()

# Load data
df = load_data()
X_cond1, X_cond2, y_clf, y_reg, y_group = get_features_and_conditions(df)

# Hyperparameter grids
LOG_REG_GRID = {
    "C": [0.01, 0.1, 1, 10],
}

RF_CLF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5],
}


# Pass/Fail tuning (f1 scoring since it is binary)
@st.cache_resource
def fit_pass_fail_models(condition_name, _X_train, _y_train):
    log_search = GridSearchCV(
        LogisticRegression(max_iter=5000, random_state=42),
        LOG_REG_GRID, cv=5, scoring="f1", n_jobs=-1,
    )
    log_search.fit(_X_train, _y_train)

    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        RF_CLF_GRID, cv=5, scoring="f1", n_jobs=-1,
    )
    rf_search.fit(_X_train, _y_train)

    return {
        "log": {"model": log_search.best_estimator_, "best_params": log_search.best_params_, "cv_f1": log_search.best_score_},
        "rf": {"model": rf_search.best_estimator_, "best_params": rf_search.best_params_, "cv_f1": rf_search.best_score_},
    }


# Run pass/fail classification for one condition
def run_pass_fail(X_data, y_clf, condition_name):
    st.header(f"{condition_name} — Pass/Fail")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_clf, test_size=0.2, random_state=42)
    fitted = fit_pass_fail_models(condition_name, X_train, y_train)

    # Logistic Regression
    log_model = fitted["log"]["model"]
    log_preds = log_model.predict(X_test)
    log_probs = log_model.predict_proba(X_test)[:, 1]

    st.subheader("Logistic Regression")
    st.write(f"Accuracy: {accuracy_score(y_test, log_preds):.2%}")
    st.write(f"F1 Score: {f1_score(y_test, log_preds):.4f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, log_probs):.4f}")
    st.write(f"Cross-Val F1: {fitted['log']['cv_f1']:.4f}")
    st.write("**Best Hyperparameters:**", fitted["log"]["best_params"])

    # Random Forest
    rf_model = fitted["rf"]["model"]
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    st.subheader("Random Forest")
    st.write(f"Accuracy: {accuracy_score(y_test, rf_preds):.2%}")
    st.write(f"F1 Score: {f1_score(y_test, rf_preds):.4f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    st.write(f"Cross-Val F1: {fitted['rf']['cv_f1']:.4f}")
    st.write("**Best Hyperparameters:**", fitted["rf"]["best_params"])

    # SHAP feature importance for Random Forest
    st.subheader("SHAP Feature Importance (Random Forest)")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[:, :, 1], X_test, show=False)
    st.pyplot(fig, bbox_inches='tight')

    # Condition specific interpretation
    if "Condition 1" in condition_name:
        st.markdown("""
For Condition 1 Logistic Regression came out on top with an Accuracy of 79.22%, an F1 Score of 0.8644, a ROC AUC of 0.7170, and a Cross-Val F1 of 0.8477. Random Forest had a lower accuracy of 74.03% and a worse F1 of 0.8361, but its ROC AUC of 0.8137 was way higher than Logistic Regression. So Random Forest is actually better at separating the two classes when you look across all probability thresholds, even if its single threshold predictions came out worse. The Cross-Val F1 for Random Forest at 0.8508 is also higher than the test F1, which suggests the test set just happened to be a slightly harder split. Logistic Regression picked C of 0.1 after tuning which is decent regularization. The SHAP plot should be dominated by failures_mat just like the EDA was warning us about.
""")
    else:
        st.markdown("""
For Condition 2 the gap between the two models basically closed. Logistic Regression dropped to 72.73% accuracy with an F1 of 0.8320 and a ROC AUC of 0.7036. Random Forest stayed about the same at 74.03% accuracy and an F1 of 0.8413, but its ROC AUC dropped to 0.7138. The Random Forest Cross-Val F1 of 0.8503 stayed very close to its Condition 1 score, which is interesting because it means the model is finding consistent signal even without the academic features. The pass/fail task is actually holding up way better than the regression task did in Condition 2, which makes sense since predicting a binary outcome is a much easier ask than predicting an exact GPA number. Both models are still hitting around 72-74% accuracy which beats just guessing pass every time at 71%, so the models are doing real work but not by a huge margin. The SHAP plot here should rely more on study time, parental education, and the failures_mat that survives the academic cut.
""")


# Run pass/fail for both conditions
run_pass_fail(X_cond1, y_clf, "Condition 1: All Features Excluding G1/G2")
st.divider()
run_pass_fail(X_cond2, y_clf, "Condition 2: Non-Academic Features Only")