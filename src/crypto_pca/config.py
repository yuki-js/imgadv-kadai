"""Configuration for crypto PCA project."""
import os
from datetime import datetime, timedelta, timezone

# Use repository root (one level above 'src')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_URL = "https://api.binance.com"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

TARGET_SYMBOLS = 100

INTERVAL = "1d"
LIMIT = 1000

NOW_UTC = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
START_DATE = (NOW_UTC - timedelta(days=365*6))
END_DATE = NOW_UTC

REQUEST_TIMEOUT = (10, 30)  # connect, read
RETRY_MAX = 5
RETRY_BACKOFF = 1.5
SLEEP_BETWEEN_CALLS = 0.05

def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

# Filenames
PRICES_CSV = os.path.join(DATA_DIR, "crypto_prices_6y.csv")
SYMBOL_SELECTION_CSV = os.path.join(OUTPUT_DIR, "symbol_selection.csv")
TOP_SYMBOLS_CSV = os.path.join(OUTPUT_DIR, f"top_{TARGET_SYMBOLS}_symbols_6y.csv")
EXPLAINED_VARIANCE_CSV = os.path.join(OUTPUT_DIR, "explained_variance_6y.csv")
PC_COMPONENTS_CSV = os.path.join(OUTPUT_DIR, "pc_components_6y.csv")  # Temporal principal components
PC_SCORES_CSV = os.path.join(OUTPUT_DIR, "pc_scores.csv")  # Deprecated: kept for backward compatibility
PC_PLOT_PATH = os.path.join(OUTPUT_DIR, "principal_components_3x3_6y.png")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "pca_summary.txt")