import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from data_loader import load_data, get_features_and_conditions


#For consistent styling:
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

/* 1. GLOBAL FONT */
/* We target specific text tags but specifically EXCLUDE "span" tags. 
   Streamlit uses spans for Material Icons. If you force Poppins on spans, 
   the icons break and turn into text. */
p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    font-family: 'system-ui', sans-serif !important;
    color: white !important;
}

/*2. UNIFIED TEXTBOX COLORS*/
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

/* 3. BACKGROUND IMAGE */
[data-testid="stAppViewContainer"] {
    background-image: url('https://static.vecteezy.com/system/resources/previews/012/086/236/non_2x/back-to-school-doodles-in-chalkboard-background-free-vector.jpg');
    background-attachment: fixed;
    background-position: center; 
    background-repeat: no-repeat; 
    background-size: cover;  
    background-size: 120%;    

            
/* 4. SIDEBAR STYLING*/

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


# Title
st.markdown("<h1 style='text-align: center;'>🎓GPA Letter Group Classification</h1>", unsafe_allow_html=True)
st.divider()

# Intro paragraph
st.header("Introduction")
st.markdown("""
Once again, we will run **Logistic Regression** and **Random Forest**, but the task will be multi-classed rather than binary. The model has to decide between **4 different classes** (A, B, C, D/F) instead of just pass or fail.

For the metrics we are looking at, **Accuracy**, **weighted F1 Score**, and **Cross-Val F1**. We will use **weighted F1 Score** due to GPA groups being unbalanced (demonstrated in EDA). We also display the full classification report which breaks down precision, recall, and F1 per class so we can see exactly where the model is winning and losing.
""")

with st.expander("Click here to see the full detailed introduction:"):
    st.markdown("""
For the GPA Letter Group page we ran the same two classification models (Logistic Regression and Random Forest) but this time the task is multi-class instead of binary. The model has to decide between 4 different classes (A, B, C, D/F) instead of just pass or fail. For Logistic Regression this means the sigmoid function gets swapped out for a softmax that gives probabilities across all 4 classes. The Random Forest works the same way as before, just with more class options to vote on. We tuned the same hyperparameters as the Pass/Fail page.

For the metrics we are looking at Accuracy, weighted F1 Score, and Cross-Val F1. Weighted F1 is different from regular F1 because it calculates F1 for each individual class and then averages them based on how many students are in each class. We need this because the EDA showed our GPA Groups are severely imbalanced (C is 58% of the data and A is only 2%) so a regular F1 would not give us an honest read. We also display the full classification report which breaks down precision, recall, and F1 per class so we can see exactly where the model is winning and losing. We are expecting the A and D/F classes to come out the worst since those are the smallest groups.
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


# GPA Group tuning (f1_weighted since it is multi-class)
@st.cache_resource
def fit_gpa_group_models(condition_name, _X_train, _y_train):
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
        "log": {"model": log_search.best_estimator_, "best_params": log_search.best_params_, "cv_f1": log_search.best_score_},
        "rf": {"model": rf_search.best_estimator_, "best_params": rf_search.best_params_, "cv_f1": rf_search.best_score_},
    }


# Run GPA letter group classification for one condition
def run_gpa_groups(X_data, y_group, condition_name):
    st.header(f"{condition_name} — GPA Letter Groups")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_group, test_size=0.2, random_state=42)
    fitted = fit_gpa_group_models(condition_name, X_train, y_train)

    # Logistic Regression
    log_model = fitted["log"]["model"]
    log_preds = log_model.predict(X_test)

    st.subheader("Logistic Regression")
    st.write(f"Accuracy: {accuracy_score(y_test, log_preds):.2%}")
    st.write(f"F1 Score (weighted): {f1_score(y_test, log_preds, average='weighted'):.4f}")
    st.write(f"Cross-Val F1 (weighted): {fitted['log']['cv_f1']:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, log_preds))
    st.write("**Best Hyperparameters:**", fitted["log"]["best_params"])

    # Random Forest
    rf_model = fitted["rf"]["model"]
    rf_preds = rf_model.predict(X_test)

    st.subheader("Random Forest")
    st.write(f"Accuracy: {accuracy_score(y_test, rf_preds):.2%}")
    st.write(f"F1 Score (weighted): {f1_score(y_test, rf_preds, average='weighted'):.4f}")
    st.write(f"Cross-Val F1 (weighted): {fitted['rf']['cv_f1']:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, rf_preds))
    st.write("**Best Hyperparameters:**", fitted["rf"]["best_params"])

    # Condition specific interpretation
    if "Condition 1" in condition_name:
        st.write("")
        st.subheader("📋Analysis")
        st.markdown("""
**Insight:** Random Forest seemed to perform slightly better.

1) Random Forest had the **higher accuracy**, **weighted F1 scores**, and **Cross-Val F1**. 
                    
2) However, Random Forest hit 86% recall on the C class and 52% on B, but got a flat **0% for D/F**. No recall recorded for failing grades.
                    
3) **Neither model could predict any A students** at all because there were 0 in the test set after the split

**Takeaway:** This is exactly what the EDA was warning us about; the imbalance is so severe that the smaller classes basically get ignored.

""")
    else:
        st.write("")
        st.subheader("📋Analysis")
        st.markdown("""
**Insight:** Scores across both models dropped across the board. 
1) Logistic Regression was able to keep 12% of recall on the D/F grades. 
                    
2) Although with the academic features gone, the **models are basically just guessing C for almost everyone** and hoping for the best.

**Takeaway:** Without grade information, predicting exact GPA letter buckets is too hard for this dataset, especially for anything outside the middle.

""")


# Run GPA letter groups for both conditions
run_gpa_groups(X_cond1, y_group, "Condition 1: All Features Excluding G1/G2")
st.divider()
run_gpa_groups(X_cond2, y_group, "Condition 2: Non-Academic Features Only")

#Next Page:
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col5:
    st.page_link("pages/6Prediction.py", label="Next Page", icon="▶️", use_container_width="content")