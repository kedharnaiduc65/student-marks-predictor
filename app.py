import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("🎓 Student Marks Predictor")

st.write("Enter your details to predict final marks.")

# Dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Previous_Marks": [40,45,50,55,60,65,70,75,80,85],
    "Sleep_Hours": [8,7,7,6,6,6,5,5,5,4],
    "Final_Marks": [42,48,54,58,63,69,73,78,85,90]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied","Previous_Marks","Sleep_Hours"]]
y = df["Final_Marks"]

model = LinearRegression()
model.fit(X, y)

hours = st.number_input("Hours Studied", min_value=0.0)
previous = st.number_input("Previous Marks", min_value=0.0)
sleep = st.number_input("Sleep Hours", min_value=0.0)

if st.button("Predict"):
    prediction = model.predict([[hours, previous, sleep]])
    st.success(f"Predicted Final Marks: {prediction[0]:.2f}")
