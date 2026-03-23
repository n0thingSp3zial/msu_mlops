import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SOURCE_PATH = os.path.join(BASE_DIR, "data", "source", "weatherAUS.csv")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
META_DATA_DIR = os.path.join(BASE_DIR, "data", "meta")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

CLEANED_DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")
DQ_REPORT_FILE = os.path.join(META_DATA_DIR, "dq_metrics.csv")

MASTER_DATA_FILE = os.path.join(CLEANED_DATA_DIR, "master_data.csv")

BATCH_SIZE = 25000

STATE_FILE = os.path.join(META_DATA_DIR, "stream_state.txt")
METADATA_FILE = os.path.join(META_DATA_DIR, "batches_meta.csv")
