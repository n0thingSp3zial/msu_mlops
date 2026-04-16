import pandas as pd
import pickle
import os
import logging
from src.config import MODELS_DIR

def predict_on_new_data(filepath):
    logging.info(f"[INFERENCE] Best model inference with file: {filepath}")

    if not os.path.exists(filepath):
        logging.info(f"[INFERENCE] Error: File {filepath} is not found")
        return None

    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(best_model_path):
        logging.info(f"[INFERENCE] Error: Model {best_model_path} is not found")
        return None

    with open(best_model_path, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(filepath)
    
    try:
        preds = model.predict(df)
    except ValueError as e:
        logging.info(f"[INFERENCE] Error occured: {e}")
        return None

    df['predict'] = preds
    output_path = filepath.replace(".csv", "_predicted.csv")
    df.to_csv(output_path, index=False)

    logging.info(f"[INFERENCE] Inference is successful! Results are here: {output_path}")
    return output_path
