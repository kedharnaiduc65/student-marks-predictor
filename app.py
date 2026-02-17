import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Student Marks Predictor", layout="centered")

# ---------------- TITLE ----------------
st.title("🎓 Student Performance Predictor")
st.markdown("### 📊 Machine Learning Based Academic Analysis")
st.markdown("---")

# ---------------- DATASET ----------------
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10,2,4,6,8,3,5,7,9,1,10],
    "Previous_Marks": [40,45,50,55,60,65,70,75,80,85,38,52,66,78,49,61,72,84,35,90],
    "Sleep_Hours": [8,7,7,6,6,6,5,5,5,4,7,6,6,5,8,7,5,4,8,4],
    "Final_Marks": [42,48,54,58,63,69,73,78,85,90,44,57,70,82,53,65,76,88,40,95]
}

df = pd.DataFrame(data)

# ---------------- MODEL TRAINING ----------------
X = df[["Hours_Studied", "Previous_Marks", "Sleep_Hours"]]
y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

st.success(f"Model Accuracy (R² Score): {accuracy:.2f}")

# ---------------- USER INPUT ----------------
st.subheader("🔍 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    hours = st.number_input("Hours Studied", min_value=0.0)

with col2:
    previous = st.number_input("Previous Marks", min_value=0.0)

sleep = st.slider("Sleep Hours", 0, 12, 6)

# ---------------- PREDICTION ----------------
if st.button("Predict Final Marks"):
    prediction = model.predict([[hours, previous, sleep]])
    st.success(f"🎯 Predicted Final Marks: {prediction[0]:.2f}")

    result_df = pd.DataFrame({
        "Hours Studied": [hours],
        "Previous Marks": [previous],
        "Sleep Hours": [sleep],
        "Predicted Marks": [prediction[0]]
    })

    st.download_button("⬇ Download Prediction Report", 
                       result_df.to_csv(index=False), 
                       "prediction_report.csv")

# ---------------- VISUALIZATION ----------------
st.subheader("📈 Study Hours vs Final Marks")

fig, ax = plt.subplots()
ax.scatter(df["Hours_Studied"], df["Final_Marks"])
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Final Marks")
st.pyplot(fig)

