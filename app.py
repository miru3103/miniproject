import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


st.set_page_config(page_title="Food Adulteration Detection", layout="wide")
st.markdown("""
    <style>
    body { background-color: #0e1117; color: white; }
    .main { background-color: #0e1117; }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("FOODSAFE: Food Adulteration Detection Dashboard")
st.markdown("This dashboard predicts whether a given food sample is **Safe (Pure)** or **Risk (Adulterated)** using machine learning.")


DATA_PATH = r"C:\Users\mruna\OneDrive\Documents\FINAL\food_adulteration_dataset.csv"  
df = pd.read_csv(DATA_PATH)


X = df.drop(columns=["label", "Food_Item"])
y = df["label"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

model = Pipeline([
    ("preprocess", num_pipe),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
model.fit(X_train, y_train)


st.sidebar.header("🔍 Select Food Sample")
food_choice = st.sidebar.selectbox("Choose a Food Item:", df["Food_Item"].unique())

sample_data = df[df["Food_Item"] == food_choice].sample(1).iloc[0]

st.sidebar.markdown("### Selected Sample Data:")
st.sidebar.dataframe(pd.DataFrame(sample_data).T)



input_data = pd.DataFrame([sample_data.drop(labels=["Food_Item", "label"])])
prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]


st.markdown("---")
st.header("🔬 Prediction Result")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Sample Information")
    st.write(f"**Food Item:** {sample_data['Food_Item']}")
    st.write(f"**Moisture Content:** {sample_data['Moisture_Content']:.2f}")
    st.write(f"**Acidity:** {sample_data['Acidity']:.2f}")
    st.write(f"**Starch Content:** {sample_data['Starch_Content']:.2f}")
    st.write(f"**Sugar Content:** {sample_data['Sugar_Content']:.2f}")
    st.write(f"**Ash Content:** {sample_data['Ash_Content']:.2f}")
    st.write(f"**Density:** {sample_data['Density']:.2f}")
    st.write(f"**Conductivity:** {sample_data['Conductivity']:.2f}")

with col2:
    st.subheader("📊 Adulteration Status")
    if prediction == 0:
        st.success(" This sample is predicted to be **SAFE (Pure)**.")
        fig, ax = plt.subplots()
        ax.pie([1, 0], labels=["Safe", ""], colors=["#00C853", "#1e1e1e"], startangle=60)
        ax.text(0, 0, "SAFE", color="white", ha="center", va="center", fontsize=20, fontweight='bold')
        st.pyplot(fig)
    else:
        st.error(" This sample is predicted to be **RISK (Adulterated)**.")
        fig, ax = plt.subplots()
        ax.pie([1, 0], labels=["Risk", ""], colors=["#ff1744", "#1e1e1e"], startangle=60)
        ax.text(0, 0, "RISK", color="white", ha="center", va="center", fontsize=20, fontweight='bold')
        st.pyplot(fig)


if "history" not in st.session_state:
    st.session_state["history"] = []


if st.button("💾 Save Prediction to History"):
    st.session_state["history"].append({
        "Food_Item": food_choice,
        "Prediction": "Safe" if prediction == 0 else "Adulterated",
        "Probability (Risk)": round(prob, 3),
        "Moisture_Content": round(sample_data["Moisture_Content"], 2),
        "Acidity": round(sample_data["Acidity"], 2),
        "Starch_Content": round(sample_data["Starch_Content"], 2),
        "Sugar_Content": round(sample_data["Sugar_Content"], 2),
        "Ash_Content": round(sample_data["Ash_Content"], 2),
        "Density": round(sample_data["Density"], 2),
        "Conductivity": round(sample_data["Conductivity"], 2),
    })
    st.success(" Prediction saved to history!")

st.markdown("---")
st.header("📜 Prediction History")

if len(st.session_state["history"]) > 0:
    hist_df = pd.DataFrame(st.session_state["history"])
    st.dataframe(hist_df, use_container_width=True)
    csv = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Prediction History as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions saved yet. Make a prediction and click **Save Prediction to History**.")


st.markdown("---")
st.subheader("📈 Model Performance on Test Data")

y_pred = model.predict(X_test)
st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
st.metric("ROC AUC", f"{roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.2f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)



