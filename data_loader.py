import streamlit as st
import pandas as pd


@st.cache_data
def load_data():
    #We will read both student datasets, one for math and one for Portuguese
    #The files use semicolons as separators, so we include sep=";"
    df_mat = pd.read_csv("dataset/student+performance/student/student-mat.csv", sep=";")
    df_por = pd.read_csv("dataset/student+performance/student/student-por.csv", sep=";")

    identity_columns = [
        "school", "sex", "age", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason",
        "nursery", "internet"
    ]
    #We will merge the math and Portuguese datasets using the identity columns
    #Then we remove duplicate rows so the cleaned dataset does not repeat the same student records
    df_merged = pd.merge(df_mat, df_por, on=identity_columns, suffixes=("_mat", "_por"))
    df_cleaned = df_merged.drop_duplicates()

    #We will combine the grade columns from both subjects to create one average final grade
    #Then we convert that grade from a 20-point scale to a 4.0 GPA scale
    #We also create a Pass_Fail target where GPA 2.0 or higher is passing
    grade_cols = ['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por']
    df_cleaned['final_gpa'] = df_cleaned[grade_cols].mean(axis=1)
    df_cleaned['Combined_GPA'] = (df_cleaned['final_gpa'] / 20) * 4.0
    df_cleaned['Pass_Fail'] = (df_cleaned['Combined_GPA'] >= 2.0).astype(int)

    def gpa_to_letter(gpa):
        if gpa >= 3.5:
            return 'A'
        elif gpa >= 2.5:
            return 'B'
        elif gpa >= 1.5:
            return 'C'
        else:
            return 'D/F'
    #We will apply the GPA letter function to create the GPA_Group column
    #Then we return the cleaned dataset so it can be used in the Streamlit pages
    df_cleaned['GPA_Group'] = df_cleaned['Combined_GPA'].apply(gpa_to_letter)

    return df_cleaned


def get_features_and_conditions(df_cleaned):
    columns_to_drop = [
        'final_gpa', 'Combined_GPA', 'Pass_Fail', 'GPA_Group',
        'G1_mat', 'G2_mat', 'G3_mat',
        'G1_por', 'G2_por', 'G3_por'
    ]
 #We will create the feature matrix by dropping the target variables and grade columns
    #Then we convert categorical columns into dummy variables so the models can use them
    X = df_cleaned.drop(columns=columns_to_drop)
    X = pd.get_dummies(X, drop_first=True)

    y_clf = df_cleaned['Pass_Fail']
    y_reg = df_cleaned['Combined_GPA']
    y_group = df_cleaned['GPA_Group']

    X_cond1 = X.drop(columns=[col for col in X.columns if ('G1' in col or 'G2' in col)], errors='ignore')
    #We will create Condition 2 by removing academic-related features
    #This lets us test how well the models work using mostly non-academic information
    academic_keywords = ['G1', 'G2', 'G3', 'absences', 'studytime', 'failures']
    academic_features = [col for col in X.columns if any(keyword in col for keyword in academic_keywords)]
    X_cond2 = X.drop(columns=academic_features, errors='ignore')

    return X_cond1, X_cond2, y_clf, y_reg, y_group