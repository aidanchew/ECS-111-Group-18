import streamlit as st
import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

st.title("Group 18: Student Performance")

# ==================================================================================================================
# Loading the Data
# ==================================================================================================================

df_mat = pd.read_csv("student+performance/student/student-mat.csv", sep = ";")

df_por = pd.read_csv("student+performance/student/student-por.csv", sep = ";")

# ==================================================================================================================
# Merging the two Datasets
# ==================================================================================================================

identity_columns = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason",
    "nursery", "internet"
]

df_merged = pd.merge(df_mat, df_por, on = identity_columns, suffixes = ("_mat", "_por"))

df_cleaned = df_merged.drop_duplicates()

# ==================================================================================================================
# Target
# ==================================================================================================================

grade_cols = [
    'G1_mat', 'G2_mat', 'G3_mat',
    'G1_por', 'G2_por', 'G3_por'
]

# Average grades

df_cleaned['final_gpa'] = df_cleaned[grade_cols].mean(axis = 1)

# Convert to 4.0 GPA scale from Porteguese

df_cleaned['Combined_GPA'] = (df_cleaned['final_gpa'] / 20) * 4.0

# Pass or Fail

df_cleaned['Pass_Fail'] = (df_cleaned['Combined_GPA'] >= 2.0).astype(int)

st.subheader("Targets Preview")
st.dataframe(df_cleaned[['Combined_GPA', 'Pass_Fail']].head())

# ==================================================================================================================
# Feature Selection
# ==================================================================================================================

columns_to_drop = [
    'final_gpa',
    'Combined_GPA',
    'Pass_Fail',

    # Original grades

    'G1_mat', 'G2_mat', 'G3_mat',
    'G1_por', 'G2_por', 'G3_por'
]

X = df_cleaned.drop(columns = columns_to_drop)

# Handles categorical values

X = pd.get_dummies(X, drop_first = True)


# For Logistic Regression and Random Forest

y_clf = df_cleaned['Pass_Fail']

# For Linear Regression

y_reg = df_cleaned["Combined_GPA"]


# ==================================================================================================================
#  Conditions
# ==================================================================================================================

# Condition 1
# All features excluding G1 and G2

X_cond1 = X.drop(columns = [col for col in X.columns if ('G1' in col or 'G2' in col)], errors = 'ignore')

# Condition 2
# Non academic features only
# Remove academic related variables

academic_keywords = [
    'G1',
    'G2',
    'G3',
    'absences',
    'studytime',
    'failures'
]

academic_features = [col for col in X.columns if any(keyword in col for keyword in academic_keywords)]

X_cond2 = X.drop(columns = academic_features, errors = 'ignore')

# ==================================================================================================================
# Helper Function
# ==================================================================================================================

def evaluate_models(X_data, y_clf, y_reg, condition_name):

    st.header(condition_name)
    
    # ==================================================================================================================
    # Train Test Split
    # ==================================================================================================================

    X_train, X_test, y_train_clf, y_test_clf = train_test_split(X_data, y_clf, test_size = 0.2, random_state = 42)

    _, _, y_train_reg, y_test_reg = train_test_split(X_data, y_reg, test_size = 0.2, random_state = 42)

    # Logistic Regression

    log_model = LogisticRegression(max_iter = 1000, random_state = 42)

    log_model.fit(X_train, y_train_clf)

    log_preds = log_model.predict(X_test)
    log_probs = log_model.predict_proba(X_test)[:, 1]

    log_accuracy = accuracy_score(y_test_clf, log_preds)
    log_f1 = f1_score(y_test_clf, log_preds)
    log_auc = roc_auc_score(y_test_clf, log_probs)

    # Cross Validation
    log_cv_f1 = cross_val_score(log_model, X_data, y_clf, cv = 5, scoring = 'f1').mean()

    # Random Forest

    rf_model = RandomForestClassifier(random_state=42)

    rf_model.fit(X_train, y_train_clf)

    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    rf_accuracy = accuracy_score(y_test_clf, rf_preds)
    rf_f1 = f1_score(y_test_clf, rf_preds)
    rf_auc = roc_auc_score(y_test_clf, rf_probs)

    # Cross Validation F1

    rf_cv_f1 = cross_val_score(rf_model, X_data, y_clf, cv = 5, scoring = 'f1').mean()


    # Linear Regression

    lin_model = LinearRegression()

    lin_model.fit(X_train, y_train_reg)

    lin_preds = lin_model.predict(X_test)

    lin_rmse = root_mean_squared_error(y_test_reg, lin_preds)

    lin_mae = mean_absolute_error(y_test_reg, lin_preds)

    lin_r2 = r2_score(y_test_reg, lin_preds)

   # Cross Validation RMSE
   
    lin_cv_rmse = -cross_val_score(lin_model, X_data, y_reg, cv = 5, scoring = 'neg_root_mean_squared_error').mean()

    # ==================================================================================================================
    # Results
    # ==================================================================================================================

    st.subheader("Logistic Regression")

    st.write(f"Accuracy: {log_accuracy:.2%}")
    st.write(f"F1 Score: {log_f1:.4f}")
    st.write(f"ROC AUC: {log_auc:.4f}")
    st.write(f"Cross-Val F1: {log_cv_f1:.4f}")

    st.subheader("Random Forest")

    st.write(f"Accuracy: {rf_accuracy:.2%}")
    st.write(f"F1 Score: {rf_f1:.4f}")
    st.write(f"ROC AUC: {rf_auc:.4f}")
    st.write(f"Cross-Val F1: {rf_cv_f1:.4f}")

    st.subheader("Linear Regression")

    st.write(f"RMSE: {lin_rmse:.4f}")
    st.write(f"MAE: {lin_mae:.4f}")
    st.write(f"R²: {lin_r2:.4f}")

    # ==================================================================================================================
    # SHAP Analysis
    # ==================================================================================================================

    st.subheader("SHAP Feature Importance")

    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values[:, :, 1], X_test, show = False)

    st.pyplot(bbox_inches = 'tight')

# ==================================================================================================================
# Run Conditions
# ==================================================================================================================

evaluate_models(X_cond1, y_clf, y_reg, "Condition 1: All Features Excluding G1/G2")

evaluate_models(X_cond2, y_clf, y_reg, "Condition 2: Non-Academic Features Only")