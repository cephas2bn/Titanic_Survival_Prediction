import seaborn as sns
import numpy as pd
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Load the Titanic datasets
train_data = pd.read_csv('../data/TitanicTrain.csv')
test_data = pd.read_csv('../data/TitanicTest.csv')

# Display the first few rows of each dataset to verify
print("Training Data:")
train_data.head()


print("\nTest Data:")
test_data.head()

# Check for missing values in the training data
missing_train = train_data.isnull().sum()
print("Missing values in Training Data:")
print(missing_train[missing_train > 0])

# Check for missing values in the test data
missing_test = test_data.isnull().sum()
print("\nMissing values in Test Data:")
print(missing_test[missing_test > 0])

# Fill missing values for 'Age' with median age in each 'Pclass'
train_data['Age'].fillna(train_data.groupby('Pclass')['Age'].transform('median'), inplace=True)
test_data['Age'].fillna(test_data.groupby('Pclass')['Age'].transform('median'), inplace=True)

# Fill missing values for 'Fare' in test data with median fare of the corresponding 'Pclass'
test_data['Fare'].fillna(test_data.groupby('Pclass')['Fare'].transform('median'), inplace=True)

# Fill missing values for 'Embarked' with the most common value ('S')
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to high number of missing values
train_data.drop(columns=['Cabin'], inplace=True)
test_data.drop(columns=['Cabin'], inplace=True)

# Verify that there are no more missing values
print("Missing values in Training Data after handling:")
print(train_data.isnull().sum())

print("\nMissing values in Test Data after handling:")
print(test_data.isnull().sum())

# Analyze survival rates by gender
survival_by_gender = train_data.groupby('Sex')['Survived'].sum()
total_by_gender = train_data['Sex'].value_counts()
print("\nSurvival counts by Gender:")
print(survival_by_gender)

print("\nTotal counts by Gender:")
print(total_by_gender)

# Analyze survival rates by class
survival_by_class = train_data.groupby('Pclass')['Survived'].sum()
total_by_class = train_data['Pclass'].value_counts()

print("\nSurvival counts by Class:")
print(survival_by_class)

print("\nTotal counts by Class:")
total_by_class

# Set plot style
sns.set(style="whitegrid")

# Plot 1: Survivors Count by Port of Embarkation
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data, x='Survived', hue='Embarked', palette='Set2')
plt.title("Survivors Count by Port of Embarked")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Embarked", labels=['Q', 'S', 'C'])
plt.show()

# Plot 2: Survivors Count by Gender
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data, x='Survived', hue='Sex', palette='Set1')
plt.title("Survivors Count by Gender")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Sex", labels=['Male', 'Female'])
plt.show()

# Set plot style
sns.set(style="whitegrid")

# 1. Missing Values Before and After Handling
# Count missing values before handling
missing_before = pd.DataFrame({
    "Training Data": [177, 687, 2],
    "Test Data": [86, 327, 1]
}, index=['Age', 'Cabin', 'Embarked'])

# Count missing values after handling
missing_after = pd.DataFrame({
    "Training Data": train_data.isnull().sum(),
    "Test Data": test_data.isnull().sum()
}).loc[['Age', 'Fare', 'Embarked']]

# Plot missing values before and after handling
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
missing_before.plot(kind='bar', ax=ax[0], color=['#1f77b4', '#ff7f0e'])
ax[0].set_title("Missing Values Before Handling")
ax[0].set_ylabel("Count of Missing Values")
missing_after.plot(kind='bar', ax=ax[1], color=['#1f77b4', '#ff7f0e'])
ax[1].set_title("Missing Values After Handling")
ax[1].set_ylabel("Count of Missing Values")
plt.tight_layout()
plt.show()

# 2. Age Distribution Before and After Imputation
# Plot age distribution before and after imputation
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Age distribution before imputation
sns.histplot(train_data['Age'].dropna(), bins=30, kde=True, ax=ax[0], color='#2ca02c')
ax[0].set_title("Age Distribution (After Imputation)")

# Age distribution after imputation
sns.histplot(train_data['Age'], bins=30, kde=True, ax=ax[1], color='#d62728')
ax[1].set_title("Age Distribution (After Imputation)")
plt.tight_layout()
plt.show()

# 3. Survival Count by Gender and Class
# Survival by Gender
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(data=train_data, x='Sex', hue='Survived', ax=ax[0], palette='Set1')
ax[0].set_title("Survival Count by Gender")
ax[0].set_xlabel("Gender")
ax[0].set_ylabel("Count")

# Survival by Class
sns.countplot(data=train_data, x='Pclass', hue='Survived', ax=ax[1], palette='Set2')
ax[1].set_title("Survival Count by Class")
ax[1].set_xlabel("Passenger Class")
ax[1].set_ylabel("Count")
plt.tight_layout()
plt.show()

