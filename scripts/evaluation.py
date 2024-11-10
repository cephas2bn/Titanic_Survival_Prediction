X_test_scaled = scaler.transform(X_test)  

# Make predictions using the best Logistic Regression model
y_test_pred = logreg.predict(X_test_scaled)  

#  Prepare the submission file
predictions = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': y_test_pred
})

# Save the submission file
predictions.to_csv("../data/titanic_predictions.csv", index=False)
print("Predictions saved to titanic_predictions.csv")

# Add predictions to the original test data
test_data['Survived'] = y_test_pred

# Plot survival distributions by Gender
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=train_data, x='Survived', hue='Sex', palette='viridis')
plt.title("Survival Distribution by Gender (Train Data)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Sex", labels=['Male', 'Female'])

plt.subplot(1, 2, 2)
sns.countplot(data=test_data, x='Survived', hue='Sex', palette='viridis')
plt.title("Survival Distribution by Gender (Test Data Predictions)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Sex", labels=['Male', 'Female'])
plt.tight_layout()
plt.show()

# Plot survival distributions by Passenger Class (Pclass)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=train_data, x='Survived', hue='Pclass', palette='viridis')
plt.title("Survival Distribution by Passenger Class (Train Data)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Pclass")

plt.subplot(1, 2, 2)
sns.countplot(data=test_data, x='Survived', hue='Pclass', palette='viridis')
plt.title("Survival Distribution by Passenger Class (Test Data Predictions)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.legend(title="Pclass")
plt.tight_layout()
plt.show()

# Plot survival distributions by Age
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=train_data, x='Age', hue='Survived', multiple='stack', palette='viridis', bins=20)
plt.title("Survival Distribution by Age (Train Data)")
plt.xlabel("Age")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.histplot(data=test_data, x='Age', hue='Survived', multiple='stack', palette='viridis', bins=20)
plt.title("Survival Distribution by Age (Test Data Predictions)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Define age groups 
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=[0, 12, 18, 35, 60, 80], labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
test_data['AgeGroup'] = pd.cut(test_data['Age'], bins=[0, 12, 18, 35, 60, 80], labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

# Survival count summary by Gender
gender_summary_train = train_data.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
gender_summary_train['Dataset'] = 'Train Data'
gender_summary_test = test_data.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
gender_summary_test['Dataset'] = 'Test Data'

# Survival count summary by Passenger Class (Pclass)
pclass_summary_train = train_data.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
pclass_summary_train['Dataset'] = 'Train Data'
pclass_summary_test = test_data.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
pclass_summary_test['Dataset'] = 'Test Data'

# Survival count summary by Age Group
age_summary_train = train_data.groupby(['AgeGroup', 'Survived']).size().unstack(fill_value=0)
age_summary_train['Dataset'] = 'Train Data'
age_summary_test = test_data.groupby(['AgeGroup', 'Survived']).size().unstack(fill_value=0)
age_summary_test['Dataset'] = 'Test Data'

# Concatenate summaries
gender_summary = pd.concat([gender_summary_train, gender_summary_test])
pclass_summary = pd.concat([pclass_summary_train, pclass_summary_test])
age_summary = pd.concat([age_summary_train, age_summary_test])

# Rename columns for clarity
# gender_summary.columns = ['Not Survived', 'Survived', 'Dataset']
pclass_summary.columns = ['Not Survived', 'Survived', 'Dataset']
age_summary.columns = ['Not Survived', 'Survived', 'Dataset']

# Display summaries
print("Survival Summary by Gender")
print('male: 0, female: 1')
print(gender_summary)
print("\nSurvival Summary by Passenger Class")
print(pclass_summary)
print("\nSurvival Summary by Age Group")
print(age_summary)

