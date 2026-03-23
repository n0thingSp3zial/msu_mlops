import argparse
import sys

from src.data_collection import get_next_batch
from src.data_quality import run_dq_checks
from src.model_training import train_models
from src.model_validation import evaluate_models
from src.model_inference import predict_on_new_data
from src.report_generator import generate_summary

def run_update():
    print(">>> Запуск режима UPDATE: Сбор нового батча и дообучение...")
    
    batch_filepath = get_next_batch()
    if batch_filepath == None:
        print("<<< UPDATE завершен: Новых данных нет.")
        return False
        
    cleaned_batch_filepath = run_dq_checks(batch_filepath)

    dt_model_path, mlp_model_path, master_data_path = train_models(cleaned_batch_filepath)
    
    best_model_path = evaluate_models(dt_model_path, mlp_model_path, master_data_path)
    
    print("<<< UPDATE успешно завершен.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Pipeline MVP")
    parser.add_argument("-mode", type=str, required=True, choices=["inference", "update", "summary"],
                        help="Режим работы ML-системы")
    parser.add_argument("-file", type=str, required=False, 
                        help="Путь к файлу (требуется только для режима inference)")
    
    args = parser.parse_args()

    if args.mode == "update":
        success = run_update()
        sys.exit(0 if success else 1)
        
    elif args.mode == "inference":
        if not args.file:
            print("Ошибка: Для режима inference необходимо указать путь к файлу -file <путь>")
            sys.exit(1)
        predict_on_new_data(args.file)
        
    elif args.mode == "summary":
        generate_summary()
