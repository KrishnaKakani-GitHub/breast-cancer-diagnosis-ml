import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(data_path='data/breast_cancer_data.csv'):
    df = pd.read_csv(data_path)
    sns.pairplot(df, hue="diagnosis")
    plt.savefig('images/eda_output.png')

if __name__ == "__main__":
    run_eda()
