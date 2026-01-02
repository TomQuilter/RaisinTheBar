from config.base import DATA_DIR
import json
from pathlib import Path

RAISIN_DATA_PATH = DATA_DIR / "Raisin_Dataset.csv"

# Load training configuration from JSON
CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CONFIG_DIR / "training_config.json"
 
def load_training_config():
    """Load training configuration from JSON file."""
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config['training']

def load_split_config():
    """Load split configuration from JSON file."""
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config['split']

# Load config values
TRAINING_CONFIG = load_training_config()
LEARNING_RATE = TRAINING_CONFIG['learning_rate']
NUM_ITERATIONS = TRAINING_CONFIG['num_iterations']
RANDOM_SEED = TRAINING_CONFIG['random_seed']

SPLIT_CONFIG = load_split_config()
TEST_SPLIT = SPLIT_CONFIG['test_split']
VAL_SPLIT = SPLIT_CONFIG['val_split']  