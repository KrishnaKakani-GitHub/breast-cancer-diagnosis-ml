# 🧠 Breast Cancer Diagnosis using Machine Learning

This project demonstrates a complete pipeline for classifying breast cancer tumors using **scikit-learn**, from Exploratory Data Analysis (EDA) to model deployment using **Streamlit**. It is based on the **Breast Cancer Wisconsin Diagnostic Dataset** and includes visualizations, model training, evaluation, and a user-friendly app interface.

---

## 📁 Project Directory Structure

```bash
├── data/
│   └── breast_cancer_data.csv        # Local version of dataset
├── scripts/
│   ├── eda.py                        # EDA logic, saves figures
│   ├── model_training.py             # Train & save model, output classification report
│   ├── feature_importance.py         # Visualizes model interpretation
│   ├── predict_sample.py             # Script for testing on mock data
│   └── streamlit_app.py              # Streamlit interface
├── images/
│   ├── decision_tree.png             # Decision tree visualization
│   └── classification_report.png     # Model performance report
├── models/
│   └── trained_model.pkl             # Saved scikit-learn model
├── requirements.txt
├── .gitignore
└── README.md


📊 Classification Report


🌲 Decision Tree Visualization


📚 Dataset Information

Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: UCI Machine Learning Repository
Citation:
Wolberg, W. H., & Mangasarian, O. L. (1992). Breast Cancer Wisconsin (Diagnostic) Data Set. UCI Machine Learning Repository. Retrieved from https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic




