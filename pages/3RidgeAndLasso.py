import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from data_loader import load_data, get_features_and_conditions

st.title("Ridge and Lasso Regression")

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
run_ridge_lasso(X_cond2, y_reg, "Condition 2: Non-Academic Features Only")