import pandas as pd
import os
from datetime import datetime
import logging
from src.config import DQ_REPORT_FILE, CLEANED_DATA_DIR

def run_dq_checks(batch_filepath):
    logging.info(f"[DATA QUALITY] Started analysis of file: {batch_filepath}")
    df = pd.read_csv(batch_filepath)
    
    initial_rows = len(df)
    
    # missing_ratio = df.isna().mean().to_dict()
    # duplicates_count = df.duplicated().sum()
    
    rule1_mask = df['MinTemp'] > df['MaxTemp']
    rule2_mask = (df['Humidity9am'] < 0) | (df['Humidity9am'] > 100) | (df['Humidity3pm'] < 0) | (df['Humidity3pm'] > 100)
    rule3_mask = (df['Rainfall'] > 1.0) & (df['RainToday'] == 'No')
    rule4_mask = (df['WindGustSpeed'] < 0) | (df['WindSpeed9am'] < 0) | (df['WindSpeed3pm'] < 0)
    rule5_mask = (df['Pressure9am'] < 850) | (df['Pressure9am'] > 1100)
    
    violations = {
        "rule1_Min_gt_MaxTemp": rule1_mask.sum(),
        "rule2_Humidity_out_of_bounds": rule2_mask.sum(),
        "rule3_Rain_logic_error": rule3_mask.sum(),
        "rule4_Negative_Wind": rule4_mask.sum(),
        "rule5_Pressure_anomalies": rule5_mask.sum()
    }
    
    bad_data_mask = rule1_mask | rule2_mask | rule3_mask | rule4_mask | rule5_mask
    
    df_cleaned = df[~bad_data_mask].copy()
    df_cleaned = df_cleaned.drop_duplicates()
    df_cleaned = df_cleaned.dropna(subset=['RainTomorrow'])
    
    final_rows = len(df_cleaned)
    logging.info(f"[DATA QUALITY] Cleaning: removed {initial_rows - final_rows} defective/empty rows")
    
    dq_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_file": os.path.basename(batch_filepath),
        "initial_rows": initial_rows,
        "final_rows": final_rows,
    }
    dq_info.update(violations)
    
    dq_df = pd.DataFrame([dq_info])
    if not os.path.exists(DQ_REPORT_FILE):
        dq_df.to_csv(DQ_REPORT_FILE, index=False)
    else:
        dq_df.to_csv(DQ_REPORT_FILE, mode='a', header=False, index=False)
        
    cleaned_filename = os.path.join(CLEANED_DATA_DIR, f"cleaned_{os.path.basename(batch_filepath)}")
    df_cleaned.to_csv(cleaned_filename, index=False)
    logging.info(f"[DATA QUALITY] Cleaned data saved to {cleaned_filename}")
    
    return cleaned_filename
