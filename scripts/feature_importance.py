# feature_importance.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report

# === FUNCTION DEFINITIONS === #

def load_data(filepath='data/breast_cancer_data.csv'):
    """Load and return the cleaned, preprocessed dataset."""
    df = pd.read_csv(filepath)
    df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
    df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    return df

def split_data(df):
    """Split the dataset into training and testing sets."""
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_decision_tree(X_train, y_train):
    """Train and return a Decision Tree classifier."""
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    return dt

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on the test set."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {accuracy:.4f}")
    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred))

def plot_decision_tree(model, feature_names):
    """Visualize the trained decision tree."""
    plt.figure(figsize=(24, 16))
    tree.plot_tree(model, filled=True, feature_names=feature_names, class_names=['Benign', 'Malignant'])
    plt.title("Decision Tree Visualization")
    plt.show()

# === MAIN EXECUTION === #

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_decision_tree(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    plot_decision_tree(model, feature_names=X_train.columns)
