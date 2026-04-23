from fastapi import FastAPI
import uvicorn
import sys
import torch
import os
import threading
from pathlib import Path
from platformdirs import user_data_dir
import requests

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Global state for download progress
download_status = {
    "status": "idle",
    "percentage": 0,
    "current_file": "",
    "error": None
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "alive"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/system-info")
def get_system_info():
    device = "cpu"
    device_name = "CPU"
    
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple MPS"
    
    return {
        "device": device,
        "device_name": device_name,
        "platform": sys.platform,
        "torch_version": torch.__version__
    }

@app.get("/download-progress")
def get_download_progress():
    return download_status

def perform_download(url, dest_path):
    global download_status
    try:
        download_status["status"] = "downloading"
        download_status["current_file"] = dest_path.name
        download_status["percentage"] = 0
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, "wb") as f:
            if total_size == 0:
                f.write(response.content)
                download_status["percentage"] = 100
            else:
                downloaded = 0
                for data in response.iter_content(chunk_size=4096):
                    downloaded += len(data)
                    f.write(data)
                    download_status["percentage"] = int(100 * downloaded / total_size)
        
        download_status["status"] = "complete"
    except Exception as e:
        download_status["status"] = "failed"
        download_status["error"] = str(e)

@app.post("/download-models")
async def start_download():
    global download_status
    if download_status["status"] == "downloading":
        return {"message": "Download already in progress"}
    
    # htdemucs_6s model weights
    # Note: Using the official Facebook public file URL for htdemucs_6s
    model_url = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/htdemucs_6s-c8435831.th"
    
    app_data = Path(user_data_dir("HearMeOut", "yashk"))
    dest_path = app_data / "models" / "htdemucs_6s.th"
    
    if dest_path.exists():
        download_status["status"] = "complete"
        download_status["percentage"] = 100
        return {"message": "Model already exists"}
    
    thread = threading.Thread(target=perform_download, args=(model_url, dest_path))
    thread.start()
    
    return {"message": "Download started"}

if __name__ == "__main__":
    # Get port from command line or default to 8765
    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    
    uvicorn.run(app, host="127.0.0.1", port=port)
