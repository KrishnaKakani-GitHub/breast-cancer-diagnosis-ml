# eda.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 30)

# === FUNCTION DEFINITIONS === #

def load_data(filepath='data/breast_cancer_data.csv'):
    return pd.read_csv(filepath)

def clean_data(df):
    df = df.drop(columns=['id', 'Unnamed: 32'])
    return df

def encode_target(df):
    df['diagnosis'] = np.where(df['diagnosis'] == 'M', 1, 0)
    return df

def check_class_balance(df):
    print("\n✅ Class Balance:")
    print(df['diagnosis'].value_counts(normalize=True))

def plot_correlation_matrix(df):
    corr = df.corr()
    plt.figure(figsize=(24,24))
    sns.heatmap(corr, cbar=True, square=True, fmt='.1f', annot=True,
                annot_kws={'size':15}, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def plot_boxplots(df):
    melted_data = pd.melt(df, id_vars="diagnosis",
                          value_vars=['radius_worst', 'texture_worst', 'perimeter_worst'])
    plt.figure(figsize=(15, 10))
    sns.boxplot(x="variable", y="value", hue="diagnosis", data=melted_data)
    plt.title("Boxplot: Worst Features by Diagnosis")
    plt.show()

def plot_pairplot(df):
    columns = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
               'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
               'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
    sns.pairplot(data=df[columns], hue="diagnosis", palette='rocket')
    plt.show()

def explore_data(df):
    print("✅ Data preview:")
    print(df.head())
    print("\n✅ Dataset info:")
    print(df.info())

# === MAIN EXECUTION === #

if __name__ == "__main__":
    data = load_data()
    data = clean_data(data)
    explore_data(data)
    check_class_balance(data)
    data = encode_target(data)
    plot_correlation_matrix(data)
    plot_boxplots(data)
    plot_pairplot(data)
