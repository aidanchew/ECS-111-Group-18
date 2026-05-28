import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
 
from data_loader import load_data, get_features_and_conditions
 

#For consistent styling:
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

/* =========================================================
   1. GLOBAL FONT (Fixed to prevent "keyboard_arrow" bug)
   ========================================================= */
/* We target specific text tags but specifically EXCLUDE "span" tags. 
   Streamlit uses spans for Material Icons. If you force Poppins on spans, 
   the icons break and turn into text. */
p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    font-family: 'system-ui', sans-serif !important;
    color: white !important;
}

/* =========================================================
   2. UNIFIED TEXTBOX COLORS
   ========================================================= */
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



 
st.markdown("<h1 style='text-align: center;'>🔮Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style = 'text-align: center;'>Enter student characteristics to predict GPA and pass/fail outcome</h3>", unsafe_allow_html=True)
st.divider()

# load data and get the training columns so user input can be aligned to match
df = load_data()
X_cond1, X_cond2, y_clf, y_reg, y_group = get_features_and_conditions(df)
TRAINING_COLUMNS = X_cond1.columns.tolist()
 
# hyperparams grids (same as regression and classification pages)
RF_REG_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5],
}
 
RF_CLF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5],
}

# 
@st.cache_resource
def fit_prediction_models(_X_train_reg, _y_train_reg, _X_train_clf, _y_train_clf):
    """Fit tuned Random Forest models for GPA regression and pass/fail classification."""
 
    reg_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        RF_REG_GRID, cv=5, scoring="neg_mean_squared_error", n_jobs=-1,
    )
    reg_search.fit(_X_train_reg, _y_train_reg)
 
    clf_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        RF_CLF_GRID, cv=5, scoring="f1", n_jobs=-1,
    )
    clf_search.fit(_X_train_clf, _y_train_clf)
 
    return reg_search.best_estimator_, clf_search.best_estimator_

# train/test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_cond1, y_reg, test_size=0.2, random_state=42
)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_cond1, y_clf, test_size=0.2, random_state=42
)
 
reg_model, clf_model = fit_prediction_models(
    X_train_reg, y_train_reg, X_train_clf, y_train_clf
)

# Student Input Form
st.markdown("<h2 style='text-align: center;'>Personal Information</h2>", unsafe_allow_html=True)
st.markdown("")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=15, max_value=22, value=17)
    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
with col2:
    address = st.selectbox("Address Type", ["U", "R"], help="U = Urban, R = Rural")
    famsize = st.selectbox("Family Size", ["GT3", "LE3"], help="GT3 = >3, LE3 = ≤3")
    Pstatus = st.selectbox("Parent Cohabitation", ["T", "A"], help="T = Together, A = Apart")
with col3:
    Medu = st.slider("Mother's Education (0–4)", 0, 4, 2)
    Fedu = st.slider("Father's Education (0–4)", 0, 4, 2)

st.divider()


st.markdown("<h2 style='text-align: center;'>Family & Background</h2>", unsafe_allow_html=True)
st.markdown("")
 
col1, col2, col3 = st.columns(3)
with col1:
    Mjob = st.selectbox("Mother's Job", ["at_home", "health", "other", "services", "teacher"])
    Fjob = st.selectbox("Father's Job", ["at_home", "health", "other", "services", "teacher"])
with col2:
    reason = st.selectbox("Reason for School Choice", ["course", "home", "other", "reputation"])
    nursery = st.selectbox("Attended Nursery", ["yes", "no"])
    internet = st.selectbox("Internet Access", ["yes", "no"])
with col3:
    pass  # keeps layout balanced

st.divider() 


st.markdown("<h2 style='text-align: center;'>Math Course</h2>", unsafe_allow_html=True)
st.markdown("")

col1, col2, col3 = st.columns(3)
with col1:
    guardian_mat = st.selectbox("Guardian (Math)", ["father", "mother", "other"])
    traveltime_mat = st.slider("Travel Time — Math (1–4)", 1, 4, 1)
    studytime_mat = st.slider("Study Time — Math (1–4)", 1, 4, 2)
    failures_mat = st.number_input("Past Failures — Math", 0, 4, 0)
with col2:
    schoolsup_mat = st.selectbox("School Support (Math)", ["yes", "no"])
    famsup_mat = st.selectbox("Family Support (Math)", ["yes", "no"])
    paid_mat = st.selectbox("Paid Classes (Math)", ["yes", "no"])
    activities_mat = st.selectbox("Extracurriculars (Math)", ["yes", "no"])
with col3:
    higher_mat = st.selectbox("Wants Higher Ed (Math)", ["yes", "no"])
    romantic_mat = st.selectbox("In a Relationship (Math)", ["yes", "no"])
    famrel_mat = st.slider("Family Relationship — Math (1–5)", 1, 5, 3)
    freetime_mat = st.slider("Free Time — Math (1–5)", 1, 5, 3)
 
col1, col2 = st.columns(2)
with col1:
    goout_mat = st.slider("Going Out — Math (1–5)", 1, 5, 3)
    Dalc_mat = st.slider("Weekday Alcohol — Math (1–5)", 1, 5, 1)
    Walc_mat = st.slider("Weekend Alcohol — Math (1–5)", 1, 5, 1)
with col2:
    health_mat = st.slider("Health — Math (1–5)", 1, 5, 3)
    absences_mat = st.number_input("Absences — Math", 0, 93, 0)

st.divider()


st.markdown("<h2 style='text-align: center;'>Portuguese Course</h2>", unsafe_allow_html=True)
st.markdown("")

col1, col2, col3 = st.columns(3)
with col1:
    guardian_por = st.selectbox("Guardian (Portuguese)", ["father", "mother", "other"])
    traveltime_por = st.slider("Travel Time — Portuguese (1–4)", 1, 4, 1)
    studytime_por = st.slider("Study Time — Portuguese (1–4)", 1, 4, 2)
    failures_por = st.number_input("Past Failures — Portuguese", 0, 4, 0)
with col2:
    schoolsup_por = st.selectbox("School Support (Portuguese)", ["yes", "no"])
    famsup_por = st.selectbox("Family Support (Portuguese)", ["yes", "no"])
    paid_por = st.selectbox("Paid Classes (Portuguese)", ["yes", "no"])
    activities_por = st.selectbox("Extracurriculars (Portuguese)", ["yes", "no"])
with col3:
    higher_por = st.selectbox("Wants Higher Ed (Portuguese)", ["yes", "no"])
    romantic_por = st.selectbox("In a Relationship (Portuguese)", ["yes", "no"])
    famrel_por = st.slider("Family Relationship — Portuguese (1–5)", 1, 5, 3)
    freetime_por = st.slider("Free Time — Portuguese (1–5)", 1, 5, 3)
 
col1, col2 = st.columns(2)
with col1:
    goout_por = st.slider("Going Out — Portuguese (1–5)", 1, 5, 3)
    Dalc_por = st.slider("Weekday Alcohol — Portuguese (1–5)", 1, 5, 1)
    Walc_por = st.slider("Weekend Alcohol — Portuguese (1–5)", 1, 5, 1)
with col2:
    health_por = st.slider("Health — Portuguese (1–5)", 1, 5, 3)
    absences_por = st.number_input("Absences — Portuguese", 0, 93, 0)
    
# Prediction
if st.button("Predict"):
 
    # Build a single-row dict with the raw values matching the original dataset's column names
    raw_input = {
        "age": age, "Medu": Medu, "Fedu": Fedu,
        "school": school, "sex": sex, "address": address,
        "famsize": famsize, "Pstatus": Pstatus,
        "Mjob": Mjob, "Fjob": Fjob, "reason": reason,
        "nursery": nursery, "internet": internet,
        # Math
        "traveltime_mat": traveltime_mat, "studytime_mat": studytime_mat,
        "failures_mat": failures_mat, "famrel_mat": famrel_mat,
        "freetime_mat": freetime_mat, "goout_mat": goout_mat,
        "Dalc_mat": Dalc_mat, "Walc_mat": Walc_mat,
        "health_mat": health_mat, "absences_mat": absences_mat,
        "guardian_mat": guardian_mat, "schoolsup_mat": schoolsup_mat,
        "famsup_mat": famsup_mat, "paid_mat": paid_mat,
        "activities_mat": activities_mat, "higher_mat": higher_mat,
        "romantic_mat": romantic_mat,
        # Portuguese
        "traveltime_por": traveltime_por, "studytime_por": studytime_por,
        "failures_por": failures_por, "famrel_por": famrel_por,
        "freetime_por": freetime_por, "goout_por": goout_por,
        "Dalc_por": Dalc_por, "Walc_por": Walc_por,
        "health_por": health_por, "absences_por": absences_por,
        "guardian_por": guardian_por, "schoolsup_por": schoolsup_por,
        "famsup_por": famsup_por, "paid_por": paid_por,
        "activities_por": activities_por, "higher_por": higher_por,
        "romantic_por": romantic_por,
    }
 
    input_df = pd.DataFrame([raw_input])
 
    # Apply the same dummy encoding as data_loader.py
    input_encoded = pd.get_dummies(input_df, drop_first=True)
 
    # Align columns to match training data: add any missing dummy columns as 0,
    # drop any extra columns that the training data doesn't have
    input_aligned = input_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
 
    # Predict
    predicted_gpa = reg_model.predict(input_aligned)[0]
    predicted_pass_fail = clf_model.predict(input_aligned)[0]
 
    # Convert GPA to letter grade (same logic as data_loader.py)
    if predicted_gpa >= 3.5:
        letter = "A"
    elif predicted_gpa >= 2.5:
        letter = "B"
    elif predicted_gpa >= 1.5:
        letter = "C"
    else:
        letter = "D/F"
 
    # Display results
    st.divider()
    st.header("Prediction Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted GPA", f"{predicted_gpa:.2f}")
    with col2:
        st.metric("Letter Grade", letter)
    with col3:
        st.metric("Pass/Fail", "Pass" if predicted_pass_fail == 1 else "Fail")