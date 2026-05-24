#List of all packages:
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import RidgeCV, LassoCV
from data_loader import load_data, get_features_and_conditions

#List of Pages for Navigation:
pages = [
    st.Page("home.py", title="Home", icon="🏡"),
    st.Page("pages/1EDA.py", title="EDA", icon="📊"),
    st.Page("pages/2Regression.py", title="Regression", icon="📈"),
    st.Page("pages/3RidgeAndLasso.py", title="Ridge/Lasso", icon="🔢"),
    st.Page("pages/4Classification.py", title="Classification", icon="🔍"),
    st.Page("pages/5Prediction.py", title="Prediction", icon="🔮")
]

# Run Navigation:
pg = st.navigation(pages)
pg.run()