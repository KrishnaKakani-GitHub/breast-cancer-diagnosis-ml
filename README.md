### 

# Breast Cancer Diagnosis with Machine Learning

This project demonstrates a complete pipeline for classifying breast cancer tumors using scikit-learn, from EDA to model deployment using Streamlit.

---

## Project Directory Structure

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
│   └── eda_output.png                # Any saved EDA visuals
├── models/
│   └── trained_model.pkl             # Saved scikit-learn model
├── requirements.txt
├── .gitignore
└── README.md


**Breast Cancer Wisconsin Diagnostic Data Set**  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

Citation:  
Wolberg, W. H., & Mangasarian, O. L. (1992). *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic


