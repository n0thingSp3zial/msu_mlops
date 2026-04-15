import pandas as pd
import pickle
import os
import logging
from src.config import MODELS_DIR

def predict_on_new_data(filepath):
    logging.info(f"[INFERENCE] Запуск применения лучшей модели к файлу: {filepath}")

    if not os.path.exists(filepath):
        logging.info(f"[INFERENCE] Ошибка: Файл {filepath} не найден.")
        return None

    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(best_model_path):
        logging.info(f"[INFERENCE] Ошибка: Модель {best_model_path} не найдена.")
        return None

    with open(best_model_path, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(filepath)
    
    try:
        preds = model.predict(df)
    except ValueError as e:
        logging.info(f"[INFERENCE] Ошибка применения: {e}")
        return None

    df['predict'] = preds
    output_path = filepath.replace(".csv", "_predicted.csv")
    df.to_csv(output_path, index=False)

    logging.info(f"[INFERENCE] Применение завершено успешно! Результаты сохранены в: {output_path}")
    return output_path
