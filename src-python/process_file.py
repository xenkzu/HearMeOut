import os
import sys
import time
import requests
import json
import shutil
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000"
TESTING_DIR = Path(r"c:\Projects\VisualiserMax\HearMeOutTesting")
RESULTS_DIR = TESTING_DIR / "results"

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_file.py <filename_in_testing_folder>")
        return

    filename = sys.argv[1]
    input_path = TESTING_DIR / filename

    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
        return

    print(f"--- Starting Analysis for: {filename} ---")
    
    # 1. Trigger Analysis
    print("Uploading file to backend...")
    try:
        with open(input_path, 'rb') as f:
            files = {'file': (filename, f)}
            response = requests.post(f"{API_BASE}/analyze", files=files)
            response.raise_for_status()
            job_id = response.json().get("job_id")
    except Exception as e:
        print(f"Failed to start job: {e}")
        print("Is the backend running? (python main.py)")
        return

    print(f"Job started! ID: {job_id}")

    # 2. Polling
    print("Processing (Separation + Analysis)... This may take 1-3 minutes.")
    status = "pending"
    while status not in ["completed", "failed"]:
        try:
            r = requests.get(f"{API_BASE}/job/{job_id}/status")
            job_data = r.json()
            status = job_data.get("status", "unknown")
            
            if status == "separating":
                print("s", end="", flush=True)
            elif status == "analyzing":
                print("a", end="", flush=True)
            elif status == "pending" or status == "queued":
                print(".", end="", flush=True)
            
            if status in ["completed", "failed"]:
                print(f"\nFinished with status: {status}")
                break
        except Exception as e:
            print(f"\nPolling error: {e}")
            break
        
        time.sleep(5)

    if status == "failed":
        print(f"Error detail: {job_data.get('error')}")
        return

    # 3. Save Results
    print(f"Saving results to {RESULTS_DIR}...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Analysis JSON
    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(job_data.get("analysis", {}), f, indent=2)
    
    # Copy Stems
    stems = job_data.get("stems", {})
    for name, path in stems.items():
        if os.path.exists(path):
            shutil.copy2(path, RESULTS_DIR / f"{name}.wav")
            print(f"  [v] {name} stem saved")
            
    print("\n--- Done! Check the 'results' folder ---")

if __name__ == "__main__":
    main()
