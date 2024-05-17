from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

df = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)

# Handling the data set
for col in df.columns:
    # If the column is of a 'object' type (typically strings), factorize it
    if df[col].dtype == 'object':
        df[col], _ = pd.factorize(df[col])

X = df.drop(['attack_cat', 'Label'], axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Impute the missing values
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)

X_train_imputed = iterative_imputer.fit_transform(X_train)

X_valid_imputed = iterative_imputer.transform(X_valid)
X_test_imputed = iterative_imputer.transform(X_test)

X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_valid = pd.DataFrame(X_valid_imputed, columns=X_valid.columns)
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

# Creating the model instance
model = RandomForestClassifier(random_state=42)

# Load pre-trained model if exists, otherwise train and save new model
model_path = 'trained_model.joblib'
if os.path.exists(model_path):
    model = load(model_path)
    print("Load pre-trained model.")
else:
    model.fit(X_train, y_train)
    dump(model, model_path)
    print("Train and save new model.")

# Evaluate the model
y_valid_pred = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print("Validation set accuracy before selecting important features: ", valid_accuracy)
print(classification_report(y_valid, y_valid_pred))

# Choosing the important features
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

names = [X_train.columns[i] for i in indices]

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.xticks(range(X_train.shape[1]), names, rotation=90)
plt.show()

# Plot the cumulative feature importances
plt.figure(figsize=(10, 6))
plt.title("Cumulative Feature Importance")
plt.plot(np.cumsum(importances[indices]), marker='o', linestyle='-')
plt.xlabel("Number of features")
plt.ylabel("Cumulative importance")
plt.show()

# Plot the confusion matrix
cm = confusion_matrix(y_valid, y_valid_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix before selecting important features")
plt.show()

# Plot the correlation matrix
# corr = X_train.corr()
# plt.figure(figsize=(10, 6))
# plt.matshow(corr, fignum=1)

cumulative_importance = np.cumsum(importances[indices])
threshold = 0.95
selected_features = np.where(cumulative_importance > threshold)[0][0]

# N = 8

selected_features_indices = indices[:selected_features]
# selected_features_indices = indices[:N]

# Validating the selected features
X_important_train = X_train.iloc[:, selected_features_indices]
X_important_valid = X_valid.iloc[:, selected_features_indices]
X_important_test = X_test.iloc[:, selected_features_indices]

# Delete features with high correlation
corr_threshold = 0.90
corr_matrix = X_important_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
print("Features to drop: ", to_drop)
X_important_train = X_important_train.drop(to_drop, axis=1)
X_important_valid = X_important_valid.drop(to_drop, axis=1)
X_important_test = X_important_test.drop(to_drop, axis=1)

# Print the names of the selected features
selected_features_names = [X_important_train.columns[i] for i in range(len(X_important_train.columns))]
print("Selected features: ", selected_features_names)

model_important_features = RandomForestClassifier(random_state=42)

model_important_features_path = 'trained_model_with_important_features.joblib'
if os.path.exists(model_important_features_path):
    model_important_features = load(model_important_features_path)
    print("Load pre-trained model with important features.")
else:
    model_important_features.fit(X_important_train, y_train)
    dump(model_important_features, model_important_features_path)
    print("Train and save new model with important features.")

y_important_valid_pred = model_important_features.predict(X_important_valid)
valid_accuracy_important_features = accuracy_score(y_valid, y_important_valid_pred)

print("Validation set accuracy with important features: ", valid_accuracy_important_features)

print(classification_report(y_valid, y_important_valid_pred))

# cm = confusion_matrix(y_valid, y_important_valid_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix with Important Features")
# plt.show()

# Plot the correlation matrix of selected features
corr = X_important_train.corr()
plt.figure(figsize=(10, 6))
plt.title("Correlation Matrix of Selected Features")
plt.matshow(corr, fignum=1)
plt.xticks(range(len(X_important_train.columns)), X_important_train.columns, rotation=90)
plt.yticks(range(len(X_important_train.columns)), X_important_train.columns)
plt.colorbar()
plt.show()