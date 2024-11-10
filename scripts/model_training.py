# Encode 'Sex' column (0 for female, 1 for male)
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])

# One-hot encode 'Embarked' column
train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

# Select features and target variable
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = train_data['Survived']

X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=41)
len(X_train), len(X_val), len(y_train), len(y_val)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=41)
print(len(X_train), len(X_val), len(y_train), len(y_val))

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Initialize the logistic regression model with a higher max_iter
logreg = LogisticRegression(max_iter=500, random_state=41)
logreg.fit(X_train_scaled, y_train)

# Make predictions on the validation set
y_pred = logreg.predict(X_val_scaled)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

# Display results
print("Model Evaluation Metrics after Scaling and Increasing max_iter:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# Hyperparameter tuning for Decision Tree
param_grid_dt = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='f1')
grid_search_dt.fit(X_train_scaled, y_train)
best_dt = grid_search_dt.best_estimator_
y_pred_dt = best_dt.predict(X_val_scaled)
f1_dt = f1_score(y_val, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_val, y_pred_dt)
accuracy_dt = accuracy_score(y_val, y_pred_dt)

print(f"\nDecision Tree (Tuned) Best Parameters = {grid_search_dt.best_params_}")
print(f"Accuracy: {accuracy_dt:.2f}")
print(f"F1 Score = {f1_dt:.2f}")
print("Confusion Matrix:")
print(conf_matrix_dt)

# Hyperparameter tuning for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1')
grid_search_rf.fit(X_train_scaled, y_train)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_val_scaled)
f1_rf = f1_score(y_val, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_val, y_pred_rf)
accuracy_rf = accuracy_score(y_val, y_pred_rf)

print(f"\nRandom Forest (Tuned): Best Parameters = {grid_search_rf.best_params_}")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"F1 Score = {f1_rf:.2f}")
print("Confusion Matrix:")
print(conf_matrix_rf)

# Hyperparameter tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='f1')
grid_search_gb.fit(X_train_scaled, y_train)
best_gb = grid_search_gb.best_estimator_
y_pred_gb = best_gb.predict(X_val_scaled)
f1_gb = f1_score(y_val, y_pred_gb)
conf_matrix_gb = confusion_matrix(y_val, y_pred_gb)
accuracy_gb = accuracy_score(y_val, y_pred_gb)

print(f"\nGradient Boosting (Tuned): Best Parameters = {grid_search_gb.best_params_}")
print(f"Accuracy: {accuracy_gb:.2f}")
print(f"F1 Score = {f1_gb:.2f}")
print("Confusion Matrix:")
print(conf_matrix_gb)

# Model evaluation results
model_results = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [accuracy, accuracy_dt, accuracy_rf, accuracy_gb],
    'F1 Score': [f1, f1_dt, f1_rf, f1_gb]
}

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(model_results)

# Plotting accuracy and F1 score for each model with vertical bars
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot Accuracy (Vertical Bars)
ax[0].bar(results_df['Model'], results_df['Accuracy'], color='skyblue')
ax[0].set_title("Model Comparison - Accuracy")
ax[0].set_ylabel("Accuracy")
ax[0].set_ylim(0, 1)

# Plot F1 Score (Vertical Bars)
ax[1].bar(results_df['Model'], results_df['F1 Score'], color='salmon')
ax[1].set_title("Model Comparison - F1 Score")
ax[1].set_ylabel("F1 Score")
ax[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()


results_df

