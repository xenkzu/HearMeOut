import os
import uuid
import time
import shutil
import threading
import requests
import asyncio
import json
import numpy as np
import librosa
import soundfile as sf
import logging
logging.basicConfig(level=logging.INFO, force=True)
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from sse_starlette.sse import EventSourceResponse

# --- DLL Search Path for CUDA ---
# Must happen before `import onnxruntime` so the CUDA provider DLL can find all its deps.
site_packages = Path(os.getenv("LOCALAPPDATA", "")) / "Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages"
if site_packages.exists():
    # ORT capi dir must be on PATH so cuDNN split-libs (cudnn_ops_infer64_8.dll etc.) are found at runtime
    ort_capi = site_packages / "onnxruntime/capi"
    if ort_capi.exists():
        try:
            os.add_dll_directory(str(ort_capi))
            os.environ["PATH"] = str(ort_capi) + os.pathsep + os.environ.get("PATH", "")
        except:
            pass
    # Add all NVIDIA library bin dirs
    nvidia_base = site_packages / "nvidia"
    if nvidia_base.exists():
        for bin_dir in nvidia_base.glob("**/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            except:
                pass
    # Fallback: torch/lib if present
    torch_lib = site_packages / "torch/lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
        except:
            pass

# Import ORT after DLL paths are configured
import onnxruntime as ort

# --- Configuration ---
APP_NAME = "hearmeout"
MODEL_NAME = "demucs_6s.onnx"
HF_URL = f"https://huggingface.co/xenkzu/hearmeout-models/resolve/main/models/{MODEL_NAME}"

# Paths
LOCAL_APP_DATA = Path(os.getenv("LOCALAPPDATA", os.path.expanduser("~")))
BASE_DIR = LOCAL_APP_DATA / APP_NAME
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

for d in [MODEL_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Global State ---
class GlobalState:
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.provider = "CPUExecutionProvider"
        self.download_progress = {"status": "idle", "progress": 0, "total": 0, "error": None}
        self.jobs: Dict[str, dict] = {}
        self.lock = threading.Lock()

state = GlobalState()

app = FastAPI(title="HearMeOut ONNX Backend")

# --- Helper Functions ---

def detect_provider():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"

def load_session():
    model_path = MODEL_DIR / MODEL_NAME
    if not model_path.exists():
        return False
    
    try:
        # Let ONNX Runtime handle fallback automatically
        logging.info(f"Available ORT providers: {ort.get_available_providers()}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        state.session = ort.InferenceSession(str(model_path), providers=providers)
        state.provider = state.session.get_providers()[0] # Get what was actually used
        logging.info(f"Active provider: {state.provider}")
        return True
    except Exception as e:
        print(f"Error loading ONNX session: {e}")
        return False

async def download_model_task():
    model_path = MODEL_DIR / MODEL_NAME
    with state.lock:
        state.download_progress = {"status": "downloading", "progress": 0, "total": 0, "error": None}
    
    try:
        response = requests.get(HF_URL, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with state.lock:
            state.download_progress["total"] = total_size
        
        with open(model_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    with state.lock:
                        state.download_progress["progress"] = downloaded
        
        with state.lock:
            state.download_progress["status"] = "complete"
        
        # Load session immediately
        load_session()
    except Exception as e:
        with state.lock:
            state.download_progress = {"status": "error", "progress": 0, "total": 0, "error": str(e)}

def separate_audio_task(job_id: str, input_path: str):
    try:
        if state.session is None:
            if not load_session():
                raise Exception("Model not loaded and file missing.")

        state.jobs[job_id]["status"] = "processing"
        
        # 1. Load Audio
        audio, sr = librosa.load(input_path, sr=44100, mono=False)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        elif audio.shape[0] > 2:
            audio = audio[:2]
            
        # Model's fixed sequence length
        SEGMENT_LEN = 343980 
        total_samples = audio.shape[1]
        
        # 2. Chunk and Process
        # We'll process in segments of SEGMENT_LEN
        all_stems = [] # List to store results for each chunk
        
        for i in range(0, total_samples, SEGMENT_LEN):
            chunk = audio[:, i:i+SEGMENT_LEN]
            actual_len = chunk.shape[1]
            
            # Pad if last chunk is too short
            if actual_len < SEGMENT_LEN:
                chunk = np.pad(chunk, ((0, 0), (0, SEGMENT_LEN - actual_len)))
            
            # Add batch dim: [1, 2, SEGMENT_LEN]
            mix = chunk[np.newaxis, :].astype(np.float32)
            
            # Run ONNX Inference
            inputs = {state.session.get_inputs()[0].name: mix}
            outputs = state.session.run(None, inputs)
            stems_chunk = outputs[0] # [1, 6, 2, SEGMENT_LEN]
            
            # Remove padding from the result of the last chunk
            if actual_len < SEGMENT_LEN:
                stems_chunk = stems_chunk[:, :, :, :actual_len]
                
            all_stems.append(stems_chunk)
        
        # 3. Stitch and Save
        # Concatenate along the sample dimension (axis 3)
        # Result shape: [1, 6, 2, total_samples]
        full_stems = np.concatenate(all_stems, axis=3)
        
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)
        
        stem_names = ["drums", "bass", "other", "vocals", "guitar", "piano"]
        stems = {}
        
        for i, name in enumerate(stem_names):
            stem_path = job_output_dir / f"{name}.wav"
            # [1, 6, 2, N] -> [2, N] -> [N, 2] for soundfile
            stem_audio = full_stems[0, i].T 
            sf.write(str(stem_path), stem_audio, 44100)
            stems[name] = str(stem_path)
            
        state.jobs[job_id]["status"] = "completed"
        state.jobs[job_id]["stems"] = stems
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        state.jobs[job_id]["status"] = "failed"
        state.jobs[job_id]["error"] = str(e)

# --- Endpoints ---

# @app.on_event("startup")
# async def startup_event():
#     load_session()

@app.get("/system-info")
async def system_info():
    # Ensure model is loaded so provider reflects reality
    if state.session is None:
        load_session()
    
    gpu_name = "None"
    vram = "None"
    
    if state.provider == "CUDAExecutionProvider":
        gpu_name = "NVIDIA RTX 4060"
    
    return {
        "gpu": gpu_name,
        "provider_active": state.provider,
        "vram": vram,
        "onnx_version": ort.__version__
    }

@app.post("/setup/download")
async def start_download(background_tasks: BackgroundTasks):
    if state.download_progress["status"] == "downloading":
        return {"message": "Download already in progress"}
    
    background_tasks.add_task(download_model_task)
    return {"message": "Download started"}

@app.get("/setup/download/progress")
async def download_progress():
    async def event_generator():
        while True:
            with state.lock:
                data = json.dumps(state.download_progress)
                yield {"data": data}
                if state.download_progress["status"] in ["complete", "error"]:
                    break
            await asyncio.sleep(0.5)
            
    return EventSourceResponse(event_generator())

@app.post("/separate")
async def separate(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    input_path = str(UPLOAD_DIR / f"{job_id}_{file.filename}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    state.jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "filename": file.filename,
        "stems": {},
        "error": None,
        "timestamp": time.time()
    }
    
    background_tasks.add_task(separate_audio_task, job_id, input_path)
    return {"job_id": job_id}

@app.get("/job/{job_id}/status")
async def job_status(job_id: str):
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return state.jobs[job_id]

@app.get("/job/{job_id}/stems/{stem_name}")
async def get_stem(job_id: str, stem_name: str):
    if job_id not in state.jobs or state.jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Stem not ready")
    
    path = state.jobs[job_id]["stems"].get(stem_name)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing")
    
    return FileResponse(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
