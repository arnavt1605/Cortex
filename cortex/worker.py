import os
import json
import time
from pathlib import Path

from cortex.storage import SecureMemoryDB
from cortex.extractor import MemoryExtractor

CORTEX_DIR = Path.home() / ".cortex"
QUEUE_DIR = CORTEX_DIR / "queue"
LOCK_FILE = QUEUE_DIR / "worker.lock"


def process_queue():
    
    # Create the lock file so that worker is active
    LOCK_FILE.touch(exist_ok=True)

    db= SecureMemoryDB()
    extractor= None
    current_model= None

    try:
        while True:
            files = QUEUE_DIR.glob("transcript_*.json")
            files = sorted(files)

            if not files:
                #Queue is empty
                break

            for path in files:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        payload= json.load(f)

                    model_name= payload.get("model_name", "llama3.1:8b")
                    transcript= payload.get("transcript", [])

                    # Lazy loading the LLM for efficiency
                    if extractor is None or current_model != model_name:
                        extractor = MemoryExtractor(model_name=model_name)
                        current_model = model_name

                    extracted_facts= extractor.extract_facts(transcript)
                    if extracted_facts:
                        for fact in extracted_facts:
                            db.add_memory(fact)

                except Exception as e:
                    print(f"Error processing {path.name}: {e}")

                finally:
                    if path.exists():
                        path.unlink()

            time.sleep(0.5)

    finally:
        if LOCK_FILE.exists():
            #Delete the lock file
            LOCK_FILE.unlink()        

if __name__ == "__main__":
    # When launched by subprocess.Popen then this runs immediately
    process_queue()