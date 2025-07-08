import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
target_names = iris.target_names


model = RandomForestClassifier(random_state=42)
model.fit(X, y)


st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower measurements below to predict the species using a trained Random Forest model.")


st.sidebar.header("Input Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(X.iloc[:, 0].min()), float(X.iloc[:, 0].max()), float(X.iloc[:, 0].mean()))
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  float(X.iloc[:, 1].min()), float(X.iloc[:, 1].max()), float(X.iloc[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(X.iloc[:, 2].min()), float(X.iloc[:, 2].max()), float(X.iloc[:, 2].mean()))
petal_width  = st.sidebar.slider("Petal Width (cm)",  float(X.iloc[:, 3].min()), float(X.iloc[:, 3].max()), float(X.iloc[:, 3].mean()))



input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]



st.subheader("🔍 Prediction:")
st.write(f"The predicted species is: **{target_names[prediction].capitalize()}**")


st.subheader("📊 Prediction Probabilities:")
prob_df = pd.DataFrame([probabilities], columns=target_names)
st.dataframe(prob_df.style.highlight_max(axis=1, color="lightgreen"))



st.subheader("🔎 Feature Importance")
importances = model.feature_importances_
fig, ax = plt.subplots()
sns.barplot(x=importances, y=X.columns, palette="viridis", ax=ax)
plt.xlabel("Importance")
plt.ylabel("Feature")
st.pyplot(fig)


st.markdown("---")
st.markdown(
    "**App created by MURUGAYESU A**  \n"
    "[LinkedIn](https://www.linkedin.com/in/muruga-yesu-754034326) | "
    "[Instagram](https://www.instagram.com/murugayesu/)"
)


