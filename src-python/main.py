from fastapi import FastAPI
import uvicorn
import sys
import torch
import os
import threading
import time
from pathlib import Path
from platformdirs import user_data_dir
import requests

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Global state
download_status = {
    "status": "idle",
    "percentage": 0,
    "current_file": "",
    "error": None
}

system_info = {
    "device": "detecting",
    "device_name": "Detecting...",
    "platform": sys.platform,
    "torch_version": torch.__version__
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def detect_hardware():
    global system_info
    print("API: Starting hardware detection...")
    try:
        device = "cpu"
        device_name = "CPU"
        
        # Check if user is using CPU-only torch
        is_cpu_only = "+cpu" in torch.__version__
        
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            print(f"API: CUDA detected: {device_name}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            device_name = "Apple MPS"
            print("API: MPS detected")
        elif is_cpu_only:
            print("API: No GPU detected. NOTE: You are using a CPU-only version of Torch (+cpu).")
            print("API: To use your GPU, install the CUDA version of Torch.")
            device_name = "CPU (Torch +cpu)"
        else:
            print("API: No GPU detected, using CPU")
        
        system_info["device"] = device
        system_info["device_name"] = device_name
    except Exception as e:
        print(f"API: Error during hardware detection: {e}")
        system_info["device"] = "error"
        system_info["device_name"] = f"Error: {str(e)}"

# Start hardware detection in a separate thread
threading.Thread(target=detect_hardware, daemon=True).start()

@app.get("/")
def read_root():
    return {"status": "alive"}

@app.get("/health")
def health_check():
    print("API: Health check requested")
    return {"status": "ok"}

@app.get("/system-info")
def get_system_info():
    print(f"API: System info requested: {system_info['device_name']}")
    return system_info

@app.get("/download-progress")
def get_download_progress():
    return download_status

def perform_download(url, dest_path):
    global download_status
    print(f"API: Starting download from {url}")
    try:
        download_status["status"] = "downloading"
        download_status["current_file"] = dest_path.name
        download_status["percentage"] = 0
        download_status["error"] = None
        
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        print(f"API: Total size to download: {total_size} bytes")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, "wb") as f:
            if total_size == 0:
                f.write(response.content)
                download_status["percentage"] = 100
            else:
                downloaded = 0
                for data in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                    downloaded += len(data)
                    f.write(data)
                    percent = int(100 * downloaded / total_size)
                    if percent != download_status["percentage"]:
                        download_status["percentage"] = percent
                        # print(f"API: Download progress: {percent}%")
        
        print("API: Download complete")
        download_status["status"] = "complete"
    except Exception as e:
        print(f"API: Download failed: {e}")
        download_status["status"] = "failed"
        download_status["error"] = str(e)

@app.post("/download-models")
async def start_download():
    global download_status
    print("API: Download models requested")
    if download_status["status"] == "downloading":
        return {"message": "Download already in progress"}
    
    model_url = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/htdemucs_6s-c8435831.th"
    
    app_data = Path(user_data_dir("HearMeOut", "yashk"))
    dest_path = app_data / "models" / "htdemucs_6s.th"
    
    print(f"API: Model destination: {dest_path}")
    
    if dest_path.exists() and dest_path.stat().st_size > 1000000: # Check if it's a real file
        print("API: Model already exists")
        download_status["status"] = "complete"
        download_status["percentage"] = 100
        return {"message": "Model already exists"}
    
    threading.Thread(target=perform_download, args=(model_url, dest_path), daemon=True).start()
    
    return {"message": "Download started"}

if __name__ == "__main__":
    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    
    print(f"API: Starting server on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=port)
