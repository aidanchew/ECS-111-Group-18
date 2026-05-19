import streamlit as st
import matplotlib.pyplot as plt

from data_loader import load_data

st.title("Exploratory Data Analysis")

df = load_data()

#This chunk displays the shape of the dataset and then tells us how many rows and cols are in there
#Then we check the first 5 rows to ensure that the data looks correct
st.subheader("Dataset Overview")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head())

#Will create a histrogram and a mathplotlib figure 
#Will help show the distribution of the target variable (GPA) 
# Then it will push it into streamlit
st.subheader("GPA Distribution")
fig, ax = plt.subplots()
ax.hist(df['Combined_GPA'], bins=20, edgecolor='black')
ax.set_xlabel("Combined GPA (4.0 scale)")
ax.set_ylabel("Count")
st.pyplot(fig)


st.subheader("Pass/Fail Balance")
st.bar_chart(df['Pass_Fail'].value_counts())

st.subheader("GPA Group Balance")
st.bar_chart(df['GPA_Group'].value_counts())

#Select the numeric cols from the dataset 
#Then we will calculate the correlation of each numeric feature with the target variable (GPA) and display it in a table
st.subheader("Numeric Feature Correlations with GPA")
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()['Combined_GPA'].sort_values(ascending=False)
st.dataframe(corr)