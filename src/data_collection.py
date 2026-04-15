import pandas as pd
import os
from datetime import datetime
import logging
from src.config import DATA_SOURCE_PATH, RAW_DATA_DIR, STATE_FILE, METADATA_FILE, BATCH_SIZE

def get_next_batch():
    if not os.path.exists(DATA_SOURCE_PATH):
        raise FileNotFoundError(f"[DATA COLLECTION] The dataset could not be located at the {DATA_SOURCE_PATH}")

    skip_rows = 0
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            skip_rows = int(f.read().strip())

    logging.info(f"[DATA COLLECTION] Reading rows from {skip_rows} to {skip_rows + BATCH_SIZE}...")
    
    try:
        header = pd.read_csv(DATA_SOURCE_PATH, nrows=0).columns
        df_batch = pd.read_csv(DATA_SOURCE_PATH, 
                               skiprows=range(1, skip_rows + 1), 
                               nrows=BATCH_SIZE,
                               names=header,
                               header=0)
    except Exception as e:
        logging.info(f"[DATA COLLECTION] Error reading data: {e}")
        return None

    if df_batch.empty:
        logging.info("[DATA COLLECTION] Data stream exhausted (end of file reached)")
        return None

    batch_idx = skip_rows // BATCH_SIZE + 1
    batch_filename = os.path.join(RAW_DATA_DIR, f"batch_{batch_idx}.csv")
    df_batch.to_csv(batch_filename, index=False)
    logging.info(f"[DATA COLLECTION] Batch {batch_idx} saved to {batch_filename}")

    meta_info = {
        "batch_id": batch_idx,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_count": len(df_batch),
        "min_temp_mean": df_batch['MinTemp'].mean(),
        "max_temp_mean": df_batch['MaxTemp'].mean(),
        "rainfall_mean": df_batch['Rainfall'].mean(),
        "missing_target_count": df_batch['RainTomorrow'].isna().sum()
    }
    
    meta_df = pd.DataFrame([meta_info])
    
    if not os.path.exists(METADATA_FILE):
        meta_df.to_csv(METADATA_FILE, index=False)
    else:
        meta_df.to_csv(METADATA_FILE, mode='a', header=False, index=False)

    with open(STATE_FILE, "w") as f:
        f.write(str(skip_rows + len(df_batch)))

    return batch_filename

if __name__ == "__main__":
    get_next_batch()
