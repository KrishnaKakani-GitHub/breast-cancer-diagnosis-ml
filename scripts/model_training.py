# model_training.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

def load_and_prepare_data(filepath='data/breast_cancer_data.csv'):
    data = pd.read_csv(filepath)
    data.drop(columns=['id', 'Unnamed: 32'], inplace=True)
    data['diagnosis'] = np.where(data['diagnosis'] == 'M', 1, 0)
    return data

def train_decision_tree_entropy(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\nDecision Tree (Entropy Criterion) Report:")
    print(f"📊 Model Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save tree visualization
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(12, 8))
    tree.plot_tree(dt, filled=True, feature_names=X_train.columns, class_names=['Benign', 'Malignant'])
    plt.savefig("images/decision_tree.png", dpi=200)
    print("🖼️ Decision tree image saved to images/decision_tree.png")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(dt, "models/trained_model.pkl")
    print("✅ Model saved to models/trained_model.pkl")

    return y_pred


    print("\nDecision Tree (Entropy Criterion) Report:")
    print(classification_report(y_test, y_pred))

    # Save tree visualization
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(12, 8))
    tree.plot_tree(dt, filled=True, feature_names=X_train.columns, class_names=['Benign', 'Malignant'])
    plt.savefig("images/decision_tree.png", dpi=200)
    print("🖼️ Decision tree image saved to images/decision_tree.png")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(dt, "models/trained_model.pkl")
    print("✅ Model saved to models/trained_model.pkl")

    return y_pred

def calculate_error_rates(y_test, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    type_1 = FP / (FP + TN)
    type_2 = FN / (FN + TP)
    print(f"Type 1 Error Rate (FPR): {type_1:.4f}")
    print(f"Type 2 Error Rate (FNR): {type_2:.4f}")

def perform_pca_analysis(data, n_components=10):
    features = data.drop(columns=['diagnosis'])
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_
    print("\nExplained Variance by Component:")
    for i, var in enumerate(explained):
        print(f"PC{i+1}: {var:.4f}")
    return pcs, data['diagnosis']

def train_decision_tree_on_pca(X_pca, y):
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("\nPCA + Decision Tree Report:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    calculate_error_rates(y_test, y_pred)

def train_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=30, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("\nRandom Forest Classification Report:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    return rf.feature_importances_, X_train.columns

def display_feature_importance(importances, columns):
    df = pd.DataFrame({'Feature': columns, 'Importance': importances})
    df = df.sort_values(by='Importance', ascending=False)
    print("\nImportant Features:")
    print(df)

if __name__ == '__main__':
    data = load_and_prepare_data()

    # Classic Decision Tree (Entropy)
    X = data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred_entropy = train_decision_tree_entropy(X_train, y_train, X_test, y_test)
    calculate_error_rates(y_test, y_pred_entropy)

    # PCA + Decision Tree
    X_pca, y_pca = perform_pca_analysis(data, n_components=7)
    train_decision_tree_on_pca(X_pca, y_pca)

    # Random Forest + Feature Importance
    rf_importances, feature_names = train_random_forest(X_train, y_train, X_test, y_test)
    display_feature_importance(rf_importances, feature_names)
