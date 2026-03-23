import pandas as pd
import pickle
import os
import shutil
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

from src.config import MODELS_DIR, REPORTS_DIR

def evaluate_models(dt_path, mlp_path, master_data_path):
    print(f"[VALIDATION] Запуск валидации моделей...")
    df = pd.read_csv(master_data_path)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

    y = df['RainTomorrow'].map({'Yes': 1, 'No': 0}).values
    X = df.drop(columns=['RainTomorrow'])

    with open(dt_path, 'rb') as f:
        dt_pipe = pickle.load(f)
    with open(mlp_path, 'rb') as f:
        mlp_pipe = pickle.load(f)

    tscv = TimeSeriesSplit(n_splits=5)
    
    train_idx, test_idx = list(tscv.split(X))[-1]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    # decision tree
    dt_preds = dt_pipe.predict(X_test)
    dt_probs = dt_pipe.predict_proba(X_test)[:, 1]
    dt_acc = accuracy_score(y_test, dt_preds)
    dt_roc = roc_auc_score(y_test, dt_probs)

    # mlp
    mlp_preds = mlp_pipe.predict(X_test)
    mlp_probs = mlp_pipe.predict_proba(X_test)[:, 1]
    mlp_acc = accuracy_score(y_test, mlp_preds)
    mlp_roc = roc_auc_score(y_test, mlp_probs)

    print(f"[VALIDATION] Метрики Decision Tree: ROC-AUC={dt_roc:.3f}, Accuracy={dt_acc:.3f}")
    print(f"[VALIDATION] Метрики MLP Neural Net: ROC-AUC={mlp_roc:.3f}, Accuracy={mlp_acc:.3f}")

    if dt_roc >= mlp_roc:
        best_name = "DecisionTree"
        best_path = dt_path
    else:
        best_name = "MLP"
        best_path = mlp_path

    best_model_dest = os.path.join(MODELS_DIR, "best_model.pkl")
    shutil.copy(best_path, best_model_dest)
    print(f"[VALIDATION] Лучшая модель: {best_name}! Сохранена в {best_model_dest}")

    eval_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_name,
        "dt_accuracy": dt_acc,
        "dt_roc_auc": dt_roc,
        "mlp_accuracy": mlp_acc,
        "mlp_roc_auc": mlp_roc,
    }
    
    metrics_file = os.path.join(REPORTS_DIR, "validation_metrics.csv")
    metrics_df = pd.DataFrame([eval_info])
    
    if not os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, index=False)
    else:
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)

    return best_model_dest
