import streamlit as st
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
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

#We will create a function to run pass/fail classification for a given feature condition
#This lets us test Logistic Regression and Random Forest on both feature sets
def run_pass_fail(X_data, y_clf, condition_name):
    st.header(f"{condition_name} — Pass/Fail")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_clf, test_size=0.2, random_state=42)

#Logistic Regression
    #We will train a Logistic Regression model to predict whether a student passes or fails
    #We will also calculate prediction probabilities so ROC AUC can be measured
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)
    log_preds = log_model.predict(X_test)
    log_probs = log_model.predict_proba(X_test)[:, 1]
    log_cv_f1 = cross_val_score(log_model, X_data, y_clf, cv=5, scoring='f1').mean()

    st.subheader("Logistic Regression")
    st.write(f"Accuracy: {accuracy_score(y_test, log_preds):.2%}")
    st.write(f"F1 Score: {f1_score(y_test, log_preds):.4f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, log_probs):.4f}")
    st.write(f"Cross-Val F1: {log_cv_f1:.4f}")

#Random Forest
    #We will train a Random Forest model to predict pass/fail outcomes
    #We will compare its performance to Logistic Regression using the same metrics
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_cv_f1 = cross_val_score(rf_model, X_data, y_clf, cv=5, scoring='f1').mean()

    st.subheader("Random Forest")
    st.write(f"Accuracy: {accuracy_score(y_test, rf_preds):.2%}")
    st.write(f"F1 Score: {f1_score(y_test, rf_preds):.4f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    st.write(f"Cross-Val F1: {rf_cv_f1:.4f}")

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

#Logistic Regression
    #We will train a Logistic Regression model to predict GPA groups
    #Because there are multiple classes, we will use weighted F1 score
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train, y_train)
    log_preds = log_model.predict(X_test)
    log_cv_f1 = cross_val_score(log_model, X_data, y_group, cv=5, scoring='f1_weighted').mean()

    st.subheader("Logistic Regression")
    st.write(f"Accuracy: {accuracy_score(y_test, log_preds):.2%}")
    st.write(f"F1 Score (weighted): {f1_score(y_test, log_preds, average='weighted'):.4f}")
    st.write(f"Cross-Val F1 (weighted): {log_cv_f1:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, log_preds))

#Random Forest
    #We will train a Random Forest model to predict GPA groups
    #We will compare it to Logistic Regression using accuracy, weighted F1, and cross-validation
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_cv_f1 = cross_val_score(rf_model, X_data, y_group, cv=5, scoring='f1_weighted').mean()

    st.subheader("Random Forest")
    st.write(f"Accuracy: {accuracy_score(y_test, rf_preds):.2%}")
    st.write(f"F1 Score (weighted): {f1_score(y_test, rf_preds, average='weighted'):.4f}")
    st.write(f"Cross-Val F1 (weighted): {rf_cv_f1:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, rf_preds))


run_pass_fail(X_cond1, y_clf, "Condition 1: All Features Excluding G1/G2")
run_pass_fail(X_cond2, y_clf, "Condition 2: Non-Academic Features Only")

st.divider()

run_gpa_groups(X_cond1, y_group, "Condition 1: All Features Excluding G1/G2")
run_gpa_groups(X_cond2, y_group, "Condition 2: Non-Academic Features Only")