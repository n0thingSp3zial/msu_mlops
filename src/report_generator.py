import pandas as pd
import os
from datetime import datetime
from src.config import META_DATA_DIR, REPORTS_DIR, DQ_REPORT_FILE

def generate_summary():
    print("[SUMMARY] Генерация отчета о состоянии ML-системы...")

    batches_meta_path = os.path.join(META_DATA_DIR, "batches_meta.csv")
    metrics_path = os.path.join(REPORTS_DIR, "validation_metrics.csv")

    report_lines = []
    report_lines.append("# Отчет о мониторинге MLOps системы")
    report_lines.append(f"**Дата формирования отчета:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ============================================================================================================

    report_lines.append("## 1. Сбор и качество данных")
    if os.path.exists(batches_meta_path) and os.path.exists(DQ_REPORT_FILE):
        meta_df = pd.read_csv(batches_meta_path)
        dq_df = pd.read_csv(DQ_REPORT_FILE)

        total_batches = len(meta_df)
        total_rows_collected = meta_df['rows_count'].sum()
        total_duplicates = dq_df['duplicates_found'].sum()

        report_lines.append(f"- **Обраработано батчей:** {total_batches}")
        report_lines.append(f"- **Собрано сырых строк:** {total_rows_collected}")

        report_lines.append("### Мониторинг метапараметров батчей")
        report_lines.append("| Batch ID | Прочитано строк | Средняя MinTemp | Средняя MaxTemp | Пустых таргетов (удалено) |")
        report_lines.append("|:---|:---|:---|:---|:---|")
        for _, row in meta_df.iterrows():
            report_lines.append(
                f"| {row['batch_id']} | {row['rows_count']} | {row['min_temp_mean']:.1f} | "
                f"{row['max_temp_mean']:.1f} | {row['missing_target_count']} |"
            )
    else:
        report_lines.append("*Нет данных о собранных батчах.*")

    # ============================================================================================================

    report_lines.append("\n## 2. Метрики моделей и валидация")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        report_lines.append("### История обучения")
        report_lines.append("| Дата обучення | Лучшая модель | DT Accuracy | DT ROC-AUC | MLP Accuracy | MLP ROC-AUC |")
        report_lines.append("|:---|:---|:---|:---|:---|:---|")
        for _, row in metrics_df.iterrows():
            report_lines.append(
                f"| {row['timestamp']} | **{row['best_model']}** | {row['dt_accuracy']:.3f} | "
                f"{row['dt_roc_auc']:.3f} | {row['mlp_accuracy']:.3f} | {row['mlp_roc_auc']:.3f} |"
            )
    else:
        report_lines.append("*Нет метрик обученных моделей.*")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "summary_report.md")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print(f"[SUMMARY] Отчет успешно сформирован и сохранен в: {report_path}")
    return report_path
