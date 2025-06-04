import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# -------------------- Title & Intro -------------------- #
st.set_page_config(page_title="Breast Cancer Diagnosis", layout="centered")

st.title("🔬 Breast Cancer Diagnosis Predictor (CSV Upload)")

st.markdown("""
This app predicts whether a breast tumor is **benign** or **malignant** based on clinical features.  
It uses a machine learning model trained on the **Breast Cancer Wisconsin Diagnostic Dataset**.

🔹 Upload a CSV file with **30 numerical features** (no headers required).  
🔹 Click **Predict** to classify the tumor profile.
""")

# -------------------- Load Model -------------------- #
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/trained_model.pkl")
    except FileNotFoundError:
        st.error("❌ Trained model not found. Make sure 'trained_model.pkl' exists in the models/ directory.")
        st.stop()

model = load_model()

# -------------------- CSV Upload -------------------- #
uploaded_file = st.file_uploader("📁 Upload a CSV file with 30 features", type="csv")

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)

        # Drop unnecessary columns if they exist
        columns_to_drop = ['id', 'Unnamed: 32', 'diagnosis']
        input_df.drop(columns=[col for col in columns_to_drop if col in input_df.columns], inplace=True)

        # Check shape
        if input_df.shape[1] != 30:
            st.error(f"❌ Expected 30 features after cleaning, but got {input_df.shape[1]}.")
            st.stop()

        # Predict
        prediction = model.predict(input_df)
        prediction_label = ["Benign" if p == 0 else "Malignant" for p in prediction]

        st.success("✅ Prediction Result:")
        st.write(prediction_label)

        # Allow CSV download of result
        result_df = input_df.copy()
        result_df["Prediction"] = prediction_label

        st.download_button(
            label="📥 Download Input + Prediction as CSV",
            data=result_df.to_csv(index=False),
            file_name="prediction_result.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")

else:
    st.info("📄 Please upload a CSV file with numerical feature columns.")

# -------------------- Confusion Matrix -------------------- #
@st.cache_resource
def get_confusion_plot():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    return fig

st.subheader("📊 Model Evaluation")
st.pyplot(get_confusion_plot())

# -------------------- Feature Importance -------------------- #
@st.cache_resource
def get_feature_importance():
    data = load_breast_cancer()
    clf = DecisionTreeClassifier()
    clf.fit(data.data, data.target)
    importance = clf.feature_importances_

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_df = pd.DataFrame({'Feature': data.feature_names, 'Importance': importance})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax)
    plt.title("Feature Importance")
    return fig

st.subheader("🔍 Feature Importance")
st.pyplot(get_feature_importance())
