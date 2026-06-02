import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from data_loader import load_data, get_features_and_conditions

st.title("Regression Models")

df = load_data()
X_cond1, X_cond2, y_clf, y_reg, y_group = get_features_and_conditions(df)


def run_regression(X_data, y_reg, condition_name):
    st.header(condition_name)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_reg, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.subheader(name)
        st.write(f"RMSE: {root_mean_squared_error(y_test, preds):.4f}")
        st.write(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
        st.write(f"R²: {r2_score(y_test, preds):.4f}")


run_regression(X_cond1, y_reg, "Condition 1: All Features Excluding G1/G2")
run_regression(X_cond2, y_reg, "Condition 2: Non-Academic Features Only")
