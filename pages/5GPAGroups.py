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

p, h1, h2, h3, h4, h5, h6, label, .stMarkdown, pre, code {
    font-family: 'system-ui', sans-serif !important;
    color: white !important;
}

/* Force Streamlit text elements to white (classification report fix) */
[data-testid="stText"],
[data-testid="stText"] pre,
[data-testid="stCodeBlock"],
[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code,
.stText,
.stCode,
div[data-testid="stText"] *,
div[data-testid="stCodeBlock"] * {
    color: white !important;
    background-color: transparent !important;
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


# Title
st.markdown("<h1 style='text-align: center;'>🎓GPA Letter Group Classification</h1>", unsafe_allow_html=True)
st.divider()

# Intro paragraph
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
        st.markdown("""
For Condition 1 Random Forest came out slightly ahead with an Accuracy of 64.94% compared to Logistic Regression at 59.74%. The weighted F1 scores were closer with Random Forest at 0.6056 and Logistic Regression at 0.5955. Random Forest also had the better Cross-Val F1 at 0.6277. Looking at the classification report tells the real story though. Random Forest hit 86% recall on the C class and 52% on B, but got a flat 0% for D/F. Logistic Regression managed 25% recall on D/F which is at least something. Neither model could predict any A students at all because there were 0 in the test set after the split (with only 6 A students in the whole dataset, the train/test split landed all of them in training). This is exactly what the EDA was warning us about, the imbalance is so severe that the smaller classes basically get ignored.
""")
    else:
        st.markdown("""
For Condition 2 the scores dropped across the board. Logistic Regression actually did a little better than Random Forest this time with 61.04% accuracy compared to 59.74%, but the weighted F1 scores were both lower at around 0.55 to 0.59. Logistic Regression managed to keep 12% recall on the D/F class which is something, while Random Forest dropped to 0% for D/F again. The C class is still the only one being predicted with any real success at around 79-81% recall for both models. With the academic features gone, the models are basically just guessing C for almost everyone and hoping for the best. The accuracy hovering around 60% looks decent until you remember that just guessing C every time would give us around 58% accuracy by itself. So the models are only barely beating the dumbest possible strategy. The honest takeaway is that without grade information, predicting exact GPA letter buckets is too hard for this dataset, especially for anything outside the middle.
""")


# Run GPA letter groups for both conditions
run_gpa_groups(X_cond1, y_group, "Condition 1: All Features Excluding G1/G2")
st.divider()
run_gpa_groups(X_cond2, y_group, "Condition 2: Non-Academic Features Only")