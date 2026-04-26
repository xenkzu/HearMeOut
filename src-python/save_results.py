import os
import shutil
import json
import requests
from pathlib import Path

import sys

# Configuration
# Default JOB_ID, but can be overridden by first argument
JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "87023144-1965-4698-b195-633c52510b70"
API_URL = f"http://localhost:8000/job/{JOB_ID}/status"
TARGET_DIR = Path(r"c:\Projects\VisualiserMax\HearMeOutTesting\results")

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching results for job {JOB_ID}...")
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        job_data = response.json()
        
        if job_data["status"] != "completed":
            print(f"Error: Job is in state '{job_data['status']}'. Wait for it to complete.")
            return

        # 1. Save Analysis JSON
        analysis_file = TARGET_DIR / "analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(job_data.get("analysis", {}), f, indent=2)
        print(f"Saved analysis to {analysis_file}")
        
        # 2. Copy Stems
        stems = job_data.get("stems", {})
        for name, path in stems.items():
            if os.path.exists(path):
                dest = TARGET_DIR / f"{name}.wav"
                print(f"Copying {name} stem...")
                shutil.copy2(path, dest)
            else:
                print(f"Warning: Stem {name} not found at {path}")
                
        print("\nAll results stored successfully in:")
        print(TARGET_DIR)
        
    except Exception as e:
        print(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()
