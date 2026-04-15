import pandas as pd
import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import logging

from src.config import MASTER_DATA_FILE, MODELS_DIR

def build_pipeline(classifier):
    numeric_features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
        'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Temp9am', 'Temp3pm'
    ]
    categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

def train_models(cleaned_batch_filepath):
    logging.info(f"[MODEL TRAINING] Accumulating history. Adding file {cleaned_batch_filepath}")
    df_batch = pd.read_csv(cleaned_batch_filepath)

    if not os.path.exists(MASTER_DATA_FILE):
        df_batch.to_csv(MASTER_DATA_FILE, index=False)
        df_master = df_batch
    else:
        df_batch.to_csv(MASTER_DATA_FILE, mode='a', header=False, index=False)
        df_master = pd.read_csv(MASTER_DATA_FILE)

    logging.info(f"[MODEL TRAINING] Historical dataset size: {len(df_master)} rows")

    y = df_master['RainTomorrow'].map({'Yes': 1, 'No': 0})
    X = df_master.drop(columns=['RainTomorrow']) 

    dt_pipeline = build_pipeline(DecisionTreeClassifier(max_depth=7, random_state=69))
    mlp_pipeline = build_pipeline(MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=69))

    logging.info("[MODEL TRAINING] Training Pipeline (Decision Tree)...")
    dt_pipeline.fit(X, y)

    logging.info("[MODEL TRAINING] Training Pipeline (MLP Neural Net)...")
    mlp_pipeline.fit(X, y)

    version = len(df_master)
    dt_path = os.path.join(MODELS_DIR, f"dt_model_v{version}.pkl")
    mlp_path = os.path.join(MODELS_DIR, f"mlp_model_v{version}.pkl")

    with open(dt_path, 'wb') as f:
        pickle.dump(dt_pipeline, f)

    with open(mlp_path, 'wb') as f:
        pickle.dump(mlp_pipeline, f)

    logging.info(f"[MODEL TRAINING] Models serialized in models/ directory:\n  {dt_path}\n  {mlp_path}")

    return dt_path, mlp_path, MASTER_DATA_FILE
