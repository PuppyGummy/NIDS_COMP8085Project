from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from collections import OrderedDict
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from joblib import dump, load
import pandas as pd
import numpy as np
import os

df = pd.read_csv('UNSW-NB15-BALANCED-TRAIN.csv', low_memory=False)

# Columns to be used for preprocessing
nominal_cols = ['srcip','dstip','proto','state','service']
t_stamp_cols = ['Stime','Ltime']

# Custom transformer to convert timestamp
class CustomTimestampConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_X = X.copy()
        return transformed_X

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('nominal', OneHotEncoder(handle_unknown='infrequent_if_exist'), nominal_cols),
        ('t_stamp', Pipeline(steps=[
            ('timestamp_to_date', CustomTimestampConverter()),
            ('date_parts_scaler', MinMaxScaler())
        ]), t_stamp_cols)
    ],
    sparse_threshold=0.0
)

X = df.drop(['attack_cat', 'Label'], axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Impute the missing values
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)

pipeline = make_pipeline(preprocessor, iterative_imputer)

X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)
X_valid_processed = pipeline.transform(X_valid)

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
    
# Evalueate the model
y_valid_pred = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print("Validation set accuracy before selecting important features: ", valid_accuracy)
print(classification_report(y_valid, y_valid_pred))

# feature names after preprocessing
def get_feature_names(column_transformer):
    output_features = []
    
    # iterate over all transformers within the ColumnTransformer
    for transformer_tuple in column_transformer.transformers_:
        # unpack the tuple
        transformer_name, transformer, original_features = transformer_tuple[:3]
        
        # if it's not "passthrough" or "drop" and is a pipeline, get the last step as the transformer
        if transformer == 'passthrough' or transformer == 'drop':
            continue
        elif isinstance(transformer, Pipeline):
            transformer = transformer.steps[-1][1]

        if hasattr(transformer, 'get_feature_names_out'):
            if transformer_name == 'remainder':
                # if the remainder is a transformer that expects a 2D input, get the feature names accordingly
                if hasattr(transformer,'get_feature_names'):
                    # If the remainder is a transformer, use it to generate feature names
                    feature_names = transformer.get_feature_names(original_features)
                else:
                    # Otherwise, use the original feature names
                    feature_names = original_features
            else:
                # use get_feature_names_out to get the feature names for the transformation
                feature_names = transformer.get_feature_names_out()
        elif hasattr(transformer, 'get_feature_names'):
            # use get_feature_names (this is usually for older versions of scikit-learn)
            feature_names = transformer.get_feature_names(original_features)
        else:
            # if no method is available to get feature names, just use the original feature names
            feature_names = original_features

        # Collect the feature names
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        output_features.extend(feature_names)
    
    return output_features

feature_names_processed = get_feature_names(preprocessor)

# Mapping the processed feature names to the original feature names
def get_transformer_feature_names(column_transformer, processed_feature_indices):
    original_features_dict = OrderedDict()
    cumulative_features_count = 0
    
    for transformer_name, transformer, cols in column_transformer.transformers_:
        if transformer == 'passthrough' or transformer == 'drop':
            continue
        
        if isinstance(transformer, Pipeline):
            transformer = transformer.steps[-1][1]

        if hasattr(transformer, 'get_feature_names_out'):
            feature_names = transformer.get_feature_names_out()
        elif hasattr(transformer, 'get_feature_names'):
            feature_names = transformer.get_feature_names()
        else:
            feature_names = cols
        
        next_cumulative_features_count = cumulative_features_count + len(feature_names)
        relevant_indices = [index - cumulative_features_count for index in processed_feature_indices if cumulative_features_count <= index < next_cumulative_features_count]
        
        for index in relevant_indices:
            if isinstance(transformer, OneHotEncoder):
                original_feature_name = feature_names[index].split('_', 1)[0]
            else:
                original_feature_name = feature_names[index]
            
            # Save the original feature name in an ordered dictionary to maintain order
            original_features_dict[original_feature_name] = None
        
        cumulative_features_count = next_cumulative_features_count
    
    # Convert the dictionary back to a list
    return list(original_features_dict.keys())

# Choosing the important features
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

cumulative_importance = np.cumsum(importances[indices])
threshold = 0.95
selected_features = np.where(cumulative_importance > threshold)[0][0]

selected_features_indices = df.columns[indices[:selected_features]]

# Get the original feature names
original_feature_names = get_transformer_feature_names(preprocessor, selected_features_indices)

print('Selected features:', original_feature_names)
# print('Selected features:', selected_features_indices)

# Validating the selected features
X_important_train = X_train_processed[:, selected_features_indices]
X_important_valid = X_valid_processed[:, selected_features_indices]
X_important_test = X_test_processed[:, selected_features_indices]

model_important_features = RandomForestClassifier(random_state=42)

model_important_features.fit(X_important_train, y_train)

y_important_valid_pred = model_important_features.predict(X_important_valid)
valid_accuracy_important_features = accuracy_score(y_valid, y_important_valid_pred)

print("Validation set accuracy with important features: ", valid_accuracy_important_features)

print(classification_report(y_valid, y_important_valid_pred))