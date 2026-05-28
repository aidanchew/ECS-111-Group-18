#List of all packages:
import streamlit as st
import pandas as pd


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