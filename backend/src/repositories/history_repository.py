import json
from src.config.app_config import LOGS_DIR

class HistoryRepository:
    @staticmethod
    def append_log(filename, record):
        filepath = LOGS_DIR / filename
        try:
            history = []
            if filepath.exists():
                with open(filepath, "r") as f:
                    history = json.load(f)
            history.append(record)
            history = history[-100:]  # cap at 100 entries
            with open(filepath, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"[HistoryRepository] Failed to write history log: {e}")
