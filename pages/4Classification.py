import streamlit as st
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from data_loader import load_data, get_features_and_conditions

#We will create the title for the classification models page
st.title("Classification Models")

#We will load the cleaned dataset and separate it into feature sets and target variables
df = load_data()
X_cond1, X_cond2, y_clf, y_reg, y_group = get_features_and_conditions(df)

#We will preview the target variables used in the project
#This includes the numeric GPA, the pass/fail target, and the GPA group target
st.subheader("Targets Preview")
st.dataframe(df[['Combined_GPA', 'Pass_Fail', 'GPA_Group']].head())

# hyperparam grids for logistics regression and random forest 
#logistics: tune regularization strength C (default ibfgs solver only supports 12)
# random forest: same grid as the regression page

LOG_REG_GRID = {
    "C": [0.01, 0.1, 1, 10],
}
 
RF_CLF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5],
}

# separate functions: pass/fail uses f1 and GPA groups use f1_weighted
# underscore prefix skip hashing for large arrays

@st.cache_resource
def fit_pass_fail_models(condition_name, _X_train, _y_train):
    """GridSearchCV tuning for pass/fail classification. Returns fitted models,
    best params, and best CV F1 scores."""
 
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
        "log": {
            "model": log_search.best_estimator_,
            "best_params": log_search.best_params_,
            "cv_f1": log_search.best_score_,
        },
        "rf": {
            "model": rf_search.best_estimator_,
            "best_params": rf_search.best_params_,
            "cv_f1": rf_search.best_score_,
        },
    }
    
@st.cache_resource
def fit_gpa_group_models(condition_name, _X_train, _y_train):
    """GridSearchCV tuning for GPA letter group classification. Uses f1_weighted
    scoring since this is a multi-class task."""
 
    log_search = GridSearchCV(
        LogisticRegression(max_iter=5000, random_state=42),
        LOG_REG_GRID, cv=5, scoring="f1_weighted", n_jobs=-1,
    )
    log_search.fit(_X_train, _y_train)
 
    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        RF_CLF_GRID, cv=5, scoring="f1_weighted", n_jobs=-1,
    )
    rf_search.fit(_X_train, _y_train)
 
    return {
        "log": {
            "model": log_search.best_estimator_,
            "best_params": log_search.best_params_,
            "cv_f1": log_search.best_score_,
        },
        "rf": {
            "model": rf_search.best_estimator_,
            "best_params": rf_search.best_params_,
            "cv_f1": rf_search.best_score_,
        },
    }

#We will create a function to run pass/fail classification for a given feature condition
#This lets us test Logistic Regression and Random Forest on both feature sets
def run_pass_fail(X_data, y_clf, condition_name):
    st.header(f"{condition_name} — Pass/Fail")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_clf, test_size=0.2, random_state=42)

#Logistic Regression
    #We will train a Logistic Regression model to predict whether a student passes or fails
    #We will also calculate prediction probabilities so ROC AUC can be measured
    
    # models now fitted via cached GridSearchCV
    fitted = fit_pass_fail_models(condition_name, X_train, y_train)
    
    log_model = fitted["log"]["model"]
    log_preds = log_model.predict(X_test)
    log_probs = log_model.predict_proba(X_test)[:, 1]
    
    # change
    log_cv_f1 = fitted["log"]["cv_f1"]

    st.subheader("Logistic Regression")
    st.write(f"Accuracy: {accuracy_score(y_test, log_preds):.2%}")
    st.write(f"F1 Score: {f1_score(y_test, log_preds):.4f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, log_probs):.4f}")
    st.write(f"Cross-Val F1: {log_cv_f1:.4f}")
    
    # added: display best hyperparam
    st.write("**Best Hyperparameters:**", fitted["log"]["best_params"])


#Random Forest
    #We will train a Random Forest model to predict pass/fail outcomes
    #We will compare its performance to Logistic Regression using the same metrics
    
    # Change: using tuned model from GridSearchCV instead of default
    rf_model = fitted["rf"]["model"]
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    # CV F1 now comes from GridSearchCV instead of separate cross_val_score
    rf_cv_f1 = fitted["rf"]["cv_f1"]

    st.subheader("Random Forest")
    st.write(f"Accuracy: {accuracy_score(y_test, rf_preds):.2%}")
    st.write(f"F1 Score: {f1_score(y_test, rf_preds):.4f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    st.write(f"Cross-Val F1: {rf_cv_f1:.4f}")
    
    # display best hyperparams
    
    st.write("**Best Hyperparameters:**", fitted["rf"]["best_params"])

#SHAP
    #We will use SHAP to show which features had the biggest impact on the Random Forest predictions
    #This helps explain the model instead of only showing performance scores
    st.subheader("SHAP Feature Importance (Random Forest)")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[:, :, 1], X_test, show=False)
    st.pyplot(fig, bbox_inches='tight')


#We will create a function to classify students into GPA letter groups
#This is a multi-class classification problem instead of a binary pass/fail problem
def run_gpa_groups(X_data, y_group, condition_name):
    st.header(f"{condition_name} — GPA Letter Groups")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_group, test_size=0.2, random_state=42)

    # models are now fitted via cached GridSearchCV function
    fitted = fit_gpa_group_models(condition_name, X_train, y_train)

#Logistic Regression
    #We will train a Logistic Regression model to predict GPA groups
    #Because there are multiple classes, we will use weighted F1 score
    log_model = fitted["log"]["model"]
    log_preds = log_model.predict(X_test)
    
    # 
    log_cv_f1 = fitted["log"]["cv_f1"]

    st.subheader("Logistic Regression")
    st.write(f"Accuracy: {accuracy_score(y_test, log_preds):.2%}")
    st.write(f"F1 Score (weighted): {f1_score(y_test, log_preds, average='weighted'):.4f}")
    st.write(f"Cross-Val F1 (weighted): {log_cv_f1:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, log_preds))
    
    # display hyperparam
    st.write("**Best Hyperparameters:**", fitted["log"]["best_params"])

#Random Forest
    #We will train a Random Forest model to predict GPA groups
    #We will compare it to Logistic Regression using accuracy, weighted F1, and cross-validation
    rf_model = fitted["rf"]["model"]
    rf_preds = rf_model.predict(X_test)
    
    # 
    rf_cv_f1 = fitted["rf"]["cv_f1"]

    st.subheader("Random Forest")
    st.write(f"Accuracy: {accuracy_score(y_test, rf_preds):.2%}")
    st.write(f"F1 Score (weighted): {f1_score(y_test, rf_preds, average='weighted'):.4f}")
    st.write(f"Cross-Val F1 (weighted): {rf_cv_f1:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, rf_preds))
    
    # 
    st.write("**Best Hyperparameters:**", fitted["rf"]["best_params"])


run_pass_fail(X_cond1, y_clf, "Condition 1: All Features Excluding G1/G2")
run_pass_fail(X_cond2, y_clf, "Condition 2: Non-Academic Features Only")

st.divider()

run_gpa_groups(X_cond1, y_group, "Condition 1: All Features Excluding G1/G2")
run_gpa_groups(X_cond2, y_group, "Condition 2: Non-Academic Features Only")