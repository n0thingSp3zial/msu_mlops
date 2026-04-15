import argparse
import sys
import logging
import os
from src.config import LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

from src.data_collection import get_next_batch
from src.data_quality import run_dq_checks
from src.model_training import train_models
from src.model_validation import evaluate_models
from src.model_inference import predict_on_new_data
from src.report_generator import generate_summary

def run_update():
    logging.info(">>> Starting UPDATE mode: Collecting new batch and retraining...")
    
    batch_filepath = get_next_batch()
    if batch_filepath == None:
        logging.warning("<<< UPDATE finished: Data stream exhausted or error occurred")
        return False
        
    cleaned_batch_filepath = run_dq_checks(batch_filepath)
    dt_path, mlp_path, master_data_path = train_models(cleaned_batch_filepath)
    evaluate_models(dt_path, mlp_path, master_data_path)
    
    logging.info("<<< UPDATE successfully finished")
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
            logging.error("Error: For inference mode, specify the file path using -file <path>")
            sys.exit(1)
        predict_on_new_data(args.file)
        
    elif args.mode == "summary":
        generate_summary()
