from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from joblib import dump, load
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

cumulative_importance = np.cumsum(importances[indices])
threshold = 0.95
selected_features = np.where(cumulative_importance > threshold)[0][0]

# N = 8

selected_features_indices = indices[:selected_features]
# selected_features_indices = indices[:N]

# print out the name of selected features
selected_features_names = X_train.columns[selected_features_indices]
print("Selected features: ", selected_features_names)

# Validating the selected features
X_important_train = X_train.iloc[:, selected_features_indices]
X_important_valid = X_valid.iloc[:, selected_features_indices]
X_important_test = X_test.iloc[:, selected_features_indices]

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