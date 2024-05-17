# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier
import os
from joblib import dump, load
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import learning_curve
from sklearn.model_selection import LearningCurveDisplay
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import time

df = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)

# Handling the data set
for col in df.columns:
    # If the column is of a 'object' type (typically strings), factorize it
    if df[col].dtype == 'object':
        df[col], _ = pd.factorize(df[col])

# df=df.fillna(df.mean())
#print(df.isnull().any(axis=0))

x = df.drop(['attack_cat', 'Label'], axis=1)
y = df['Label']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# Impute the missing values
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)

X_train_imputed = iterative_imputer.fit_transform(X_train)

X_valid_imputed = iterative_imputer.transform(X_valid)
X_test_imputed = iterative_imputer.transform(X_test)

X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_valid = pd.DataFrame(X_valid_imputed, columns=X_valid.columns)
X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns)

print("Shape of Train Features: {}".format(X_train.shape))
print("Shape of Test Features: {}".format(X_test.shape))
print("Shape of Train Target: {}".format(y_train.shape))
print("Shape of Test Target: {}".format(y_test.shape))

# Perform univariate feature selection
selector = SelectKBest(score_func=f_regression, k=10)
X_new = selector.fit_transform(X_train, y_train)

# Get the selected feature indices
selected_indices = selector.get_support(indices=True)
selected_features = x.columns.values[selected_indices]

# Get the feature scores
scores = selector.scores_

# Plot the feature scores
plt.figure(figsize=(10,4))
plt.bar(range(len(x.columns.values)), scores, tick_label=x.columns.values)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Scores')
plt.title('Univariate Feature Selection: Feature Scores')
plt.show()

print("Selected Features:")
print(selected_features)

#validate data
X_important_train = X_train.iloc[:, selected_indices]
X_important_valid = X_valid.iloc[:, selected_indices]
X_important_test = X_test.iloc[:, selected_indices]
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