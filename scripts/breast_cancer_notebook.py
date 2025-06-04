# Import the necessary libraries
# Data manipulation and numerical operations
import pandas as pd
import numpy as np

# Data visualization
import seaborn as sns  # For creating attractive and informative statistical graphics
import matplotlib.pyplot as plt  # For plotting graphs and charts

# Machine learning model metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

# Setting maximum limit of the number of columns visible
pd.set_option('display.max_columns', 30)

# Importing the dataset
data = pd.read_csv('data/breast_cancer_data.csv')

# View the data
data.head()

# Drop the unwanted columns
data.drop(columns=['id','Unnamed: 32'],inplace=True)
data.head()

# Check the dataset
data.info()

# EDA

# Checking class imbalance
data.diagnosis.value_counts(normalize=True)

# Encoding the target variable

data['diagnosis'] = np.where(data['diagnosis'] == 'M', 1, 0)
data['diagnosis'][:5]

# Finding out the correlation between the features (using a heatmap)
# Find the correlation
corr = data.corr()
corr.shape

# Plotting the heatmap of correlation between features
plt.figure(figsize=(24,24))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='coolwarm')
plt.show()

# Transform the DataFrame from wide to long format
melted_data = pd.melt(data, id_vars="diagnosis", value_vars=['radius_worst', 'texture_worst', 'perimeter_worst'])

# Set the figure size for better visualization
plt.figure(figsize=(15, 10))

# Create a box plot to compare the distributions of 'radius_worst', 'texture_worst', and 'perimeter_worst'
# for each 'diagnosis' (M = malignant, B = benign)
sns.boxplot(x="variable", y="value", hue="diagnosis", data=melted_data)

# Display the plot
plt.show()

# List down all the columns
data.columns

# List down the 'mean' columns:

columns = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

# Visualize using the pairplot
sns.pairplot(data=data[columns], hue="diagnosis", palette='rocket')

# Model+ Building

# Import the required libraries
from sklearn.tree import DecisionTreeClassifier

# Separate features and target variable
X = data.drop(columns=['diagnosis'])  # Features
y = data['diagnosis']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
dt.fit(X_train, y_train)

# Predict on the test set
y_pred = dt.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize the decision tree
from sklearn import tree

# Plot the decision tree
plt.figure(figsize=(24,16))
tree.plot_tree(dt, filled=True, feature_names=X.columns, class_names=['Benign', 'Malignant'])
plt.show()


# Using Entropy as the criterion

# Build a decision tree classifier which uses 'entropy' as the splitting criterion
from sklearn.tree import DecisionTreeClassifier

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Fit the model
dt_entropy.fit(X_train, y_train)

# Predict using the model
y_pred = dt_entropy.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualize the decision tree
from sklearn import tree

# Plot the decision tree
plt.figure(figsize=(24,16))
tree.plot_tree(dt_entropy, filled=True, feature_names=X.columns, class_names=['Benign', 'Malignant'])
plt.show()

# Import the required libraries
from sklearn.metrics import confusion_matrix

# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Extract the components of the confusion matrix
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate Type 1 Error Rate (False Positive Rate)
type_1_error_rate = FP / (FP + TN)

# Calculate Type 2 Error Rate (False Negative Rate)
type_2_error_rate = FN / (FN + TP)

print(f'Type 1 Error Rate (False Positive Rate): {type_1_error_rate:.4f}')
print(f'Type 2 Error Rate (False Negative Rate): {type_2_error_rate:.4f}')

# Performing PCA & reducing the dimensions of the data

# Import the required module
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # Import StandardScaler for feature scaling

# Scale the features using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['diagnosis']))

# Applying PCA
pca = PCA(n_components=10)  # Starting with 10 components to visualize
principal_components = pca.fit_transform(scaled_features)

# Explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_

# Create a list of (Principal Component, Variability) tuples and sort them in descending order
components_variability = [(f'PC{i+1}', var) for i, var in enumerate(explained_variance)]
components_variability.sort(key=lambda x: x[1], reverse=True)

# Print the variability captured by each component
for component, variability in components_variability:
    print(f'{component}: {variability:.4f}')

# PCA + Decision Tree Classifier

# Apply PCA to reduce to the first 7 principal components
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Predict on the test set
y_pred = dt.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Calculating the error metrics
# Create the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Extract the components of the confusion matrix
TN, FP, FN, TP = conf_matrix.ravel()

# Calculate Type 1 Error Rate (False Positive Rate)
type_1_error_rate = FP / (FP + TN)

# Calculate Type 2 Error Rate (False Negative Rate)
type_2_error_rate = FN / (FN + TP)

print(f'Type 1 Error Rate (False Positive Rate): {type_1_error_rate:.4f}')
print(f'Type 2 Error Rate (False Negative Rate): {type_2_error_rate:.4f}')

# Import required libraries
from sklearn.ensemble import RandomForestClassifier

# Separate features and target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=30, random_state=42)

# Train the classifier on the training data
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))


# Get feature importances from the model
feature_importances = rf.feature_importances_

# Create a DataFrame for easy interpretation
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the most important features
print("Important Features:")
print(importance_df)
