import os
import uuid
import time
import shutil
import threading
import requests
import asyncio
import json
import numpy as np
import soundfile as sf
import aiofiles
import torch
import torchaudio
import torchaudio.transforms as T
import logging
import librosa
import numpy as np
import scipy
# Monkeypatch scipy.inf for msaf compatibility (scipy 1.11+ removed it)
if not hasattr(scipy, "inf"):
    scipy.inf = np.inf
# Monkeypatch scipy.signal.gaussian for msaf compatibility
import scipy.signal
if not hasattr(scipy.signal, "gaussian"):
    from scipy.signal import windows
    scipy.signal.gaussian = windows.gaussian
import msaf
logging.basicConfig(level=logging.INFO, force=True)
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import webbrowser
from sse_starlette.sse import EventSourceResponse

# Phase 4 Imports
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import scipy.stats
# We'll import basic_pitch and other libs later to avoid startup overhead if possible
# but for now let's just add the structure.

# Site packages path removed as we no longer use direct ONNX runtime DLL injection for this stage.

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
        self.demucs_model = None           # PyTorch demucs model
        self.demucs_sources: List[str] = []
        self.torch_device = "cpu"
        self.provider = "CPU"
        self.download_progress = {"status": "idle", "progress": 0, "total": 0, "error": None}
        self.jobs: Dict[str, dict] = {}
        self.lock = threading.Lock()
        self.chord_templates = None

state = GlobalState()

app = FastAPI(title="HearMeOut ONNX Backend")

# --- Helper Functions ---

def cleanup_job(job_id: str):
    """Removes job files to prevent disk bloat on failure."""
    job_dir = OUTPUT_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    logging.info(f"Cleaned up failed job {job_id}")

def init_chord_templates(device):
    """Initializes chord templates on the specified device."""
    if state.chord_templates is not None:
        return
    templates = []
    # Major: 0, 4, 7
    for i in range(12):
        t = torch.zeros(12).to(device)
        t[i] = 1.0
        t[(i + 4) % 12] = 1.0
        t[(i + 7) % 12] = 1.0
        templates.append(t)
    # Minor: 0, 3, 7
    for i in range(12):
        t = torch.zeros(12).to(device)
        t[i] = 1.0
        t[(i + 3) % 12] = 1.0
        t[(i + 7) % 12] = 1.0
        templates.append(t)
    state.chord_templates = torch.stack(templates)

def load_session():
    """Load the PyTorch Demucs model for separation (ONNX kept for future use)."""
    if state.demucs_model is not None:
        return True
    try:
        import torch
        import demucs.pretrained
        import demucs.apply
        logging.info("Loading htdemucs_6s PyTorch model...")
        bag = demucs.pretrained.get_model("htdemucs_6s")
        model = bag.models[0] if hasattr(bag, "models") else bag
        model.eval()

        # Use GPU if available
        if torch.cuda.is_available():
            state.torch_device = "cuda"
            state.provider = "CUDA"
            model = model.to("cuda")
            init_chord_templates("cuda")
            logging.info(f"Demucs running on {torch.cuda.get_device_name(0)}")
        else:
            state.torch_device = "cpu"
            state.provider = "CPU"
            init_chord_templates("cpu")
            logging.info("Demucs running on CPU")

        state.demucs_model = model
        state.demucs_sources = list(model.sources)
        logging.info(f"Demucs sources: {state.demucs_sources}")
        return True
    except Exception as e:
        logging.error(f"Error loading Demucs model: {e}")
        import traceback; traceback.print_exc()
        return False

def estimate_key(chroma_sum):
    """Estimates the musical key given a 12-element chroma sum vector."""
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    # Ensure chroma_sum is a numpy array for correlation calc
    if isinstance(chroma_sum, torch.Tensor):
        chroma_sum = chroma_sum.cpu().numpy()
        
    best_key = ""
    max_corr = -1
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i in range(12):
        major_rot = np.roll(major_profile, i)
        minor_rot = np.roll(minor_profile, i)
        corr_major = np.corrcoef(chroma_sum, major_rot)[0, 1]
        corr_minor = np.corrcoef(chroma_sum, minor_rot)[0, 1]
        if corr_major > max_corr:
            max_corr = corr_major
            best_key = f"{notes[i]} major"
        if corr_minor > max_corr:
            max_corr = corr_minor
            best_key = f"{notes[i]} minor"
    return best_key

def detect_chords(chroma_gpu, sr):
    """GPU-accelerated chord detection using template matching."""
    try:
        if state.chord_templates is None:
            init_chord_templates(chroma_gpu.device)
            
        templates = state.chord_templates.to(chroma_gpu.device)
        
        # 2. Calculate Similarity (Cosine Similarity)
        chroma_norm = torch.nn.functional.normalize(chroma_gpu, dim=0) # (12, time)
        templates_norm = torch.nn.functional.normalize(templates, dim=1) # (24, 12)
        scores = torch.matmul(templates_norm, chroma_norm) # (24, time)
        
        # 3. Smoothing (Simple Average Pooling over time)
        scores_smooth = torch.nn.functional.avg_pool1d(scores.unsqueeze(0), 21, stride=1, padding=10)[0]
        
        # 4. Get Best Chord
        best_chord_idx = torch.argmax(scores_smooth, dim=0).cpu().numpy()
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chord_labels = [f"{notes[i % 12]}" for i in range(12)] + [f"{notes[i % 12]}m" for i in range(12)]
        
        # 5. Group into intervals
        hop_time = 1024 / sr
        chords_data = []
        if len(best_chord_idx) == 0: return []
            
        current_chord = chord_labels[best_chord_idx[0]]
        start_time = 0.0
        
        for i in range(1, len(best_chord_idx)):
            chord = chord_labels[best_chord_idx[i]]
            if chord != current_chord:
                end_time = i * hop_time
                if end_time - start_time > 0.1:
                    chords_data.append({
                        "chord": current_chord,
                        "start": round(float(start_time), 2),
                        "end": round(float(end_time), 2)
                    })
                current_chord = chord
                start_time = end_time
                
        chords_data.append({
            "chord": current_chord,
            "start": round(float(start_time), 2),
            "end": round(float(len(best_chord_idx) * hop_time), 2)
        })
        return chords_data
    except Exception as e:
        logging.error(f"Chord detection failed: {e}")
        return []

def get_sections(file_path):
    """MSAF-based song segmentation using Foote algorithm."""
    try:
        # MSAF process returns (boundaries, labels)
        # We use Foote algorithm for boundaries
        boundaries, _ = msaf.process(str(file_path), boundaries_id="foote", labels_id=None)
        
        intervals = []
        for i in range(len(boundaries)-1):
            intervals.append((boundaries[i], boundaries[i+1]))
            
        if not intervals:
            return [{"label": "full", "start": 0, "end": 0}]

        durations = [e - s for s, e in intervals]
        median_dur = np.median(durations)
        
        sections = []
        for i, (start, end) in enumerate(intervals):
            dur = end - start
            if i == 0:
                label = "intro"
            elif i == len(intervals) - 1 and dur < median_dur:
                label = "outro"
            else:
                # Heuristic: long = chorus, short = verse
                label = "chorus" if dur >= median_dur else "verse"
                
            sections.append({
                "label": label,
                "start": round(float(start), 2),
                "end": round(float(end), 2)
            })
        return sections
    except Exception as e:
        logging.warning(f"MSAF segmentation failed: {e}. Falling back to single section.")
        # Fallback to full duration
        waveform, sr = torchaudio.load(file_path)
        duration = waveform.shape[-1] / sr
        return [{"label": "full", "start": 0, "end": round(float(duration), 2)}]

def ensure_vocals_stem(job_id, input_path):
    job_output_dir = OUTPUT_DIR / job_id
    vocals_path = job_output_dir / "vocals.wav"
    if vocals_path.exists():
        return str(vocals_path)
    logging.info(f"Vocals stem missing for job {job_id}, running separation...")
    separate_audio_task(job_id, input_path)
    if vocals_path.exists():
        return str(vocals_path)
    raise Exception("Failed to generate vocals stem for analysis.")

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
    """Heavy GPU task: Separates audio into 6 stems."""
    try:
        if state.demucs_model is None:
            if not load_session():
                raise Exception("Failed to load Demucs model.")

        import torch
        import demucs.apply

        with state.lock:
            state.jobs[job_id]["status"] = "separating"

        # 1. Load Audio
        audio_tensor, sr = torchaudio.load(input_path)
        if sr != 44100:
            audio_tensor = T.Resample(sr, 44100)(audio_tensor)
            sr = 44100
            
        if audio_tensor.shape[0] == 1:
            audio_tensor = torch.cat([audio_tensor, audio_tensor], dim=0)
        elif audio_tensor.shape[0] > 2:
            audio_tensor = audio_tensor[:2]

        mix = audio_tensor.unsqueeze(0)
        if state.torch_device == "cuda":
            mix = mix.cuda()

        logging.info(f"Running Demucs separation on {state.torch_device}...")
        with torch.no_grad():
            sources = demucs.apply.apply_model(
                state.demucs_model,
                mix,
                device=state.torch_device,
                progress=True,
                num_workers=1,
            )

        sources_np = sources.cpu().numpy()

        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)

        stems = {}
        for i, name in enumerate(state.demucs_sources):
            stem_path = job_output_dir / f"{name}.wav"
            stem_audio = sources_np[0, i].T
            stem_audio = np.clip(stem_audio, -1.0, 1.0)
            sf.write(str(stem_path), stem_audio, 44100)
            stems[name] = str(stem_path)

        with state.lock:
            state.jobs[job_id]["stems"] = stems
        return stems

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def analyze_basic_features(job_id: str, input_path: str):
    """GPU-accelerated basic feature analysis."""
    try:
        with state.lock:
            if state.jobs[job_id]["status"] == "pending":
                 state.jobs[job_id]["status"] = "analyzing"

        # 1. Load Audio to GPU
        waveform, sr = torchaudio.load(input_path)
        device = state.torch_device
        waveform = waveform.to(device)
        
        # Convert to mono for analysis if stereo
        if waveform.shape[0] > 1:
            y_gpu = torch.mean(waveform, dim=0, keepdim=True)
        else:
            y_gpu = waveform
            
        # Resample to 44.1k if needed (most analysis models expect this)
        if sr != 44100:
            y_gpu = T.Resample(sr, 44100).to(device)(y_gpu)
            sr = 44100

        # 2. RMS (Energy) on GPU
        # Window size 4410 (100ms)
        frame_len = 4410
        # Pad to ensure frames fit
        remainder = y_gpu.shape[-1] % frame_len
        pad_len = (frame_len - remainder) if remainder != 0 else 0
        y_padded = torch.nn.functional.pad(y_gpu, (0, pad_len))
        frames = y_padded.unfold(-1, frame_len, frame_len) # (1, num_frames, frame_len)
        rms_gpu = torch.sqrt(torch.mean(frames**2, dim=-1))[0] # (num_frames)
        
        # 3. Chroma on GPU (using Spectrogram + Mapping)
        # We use a 4096-bin spectrogram for better frequency resolution
        spec_transform = T.Spectrogram(n_fft=4096, hop_length=1024, power=2).to(device)
        spec = spec_transform(y_gpu)[0] # (freq, time)
        
        # Apply Chroma filters (pre-computed via librosa but applied on GPU)
        chroma_fb = librosa.filters.chroma(sr=sr, n_fft=4096)
        chroma_fb = torch.from_numpy(chroma_fb).to(device).float()
        chroma_gpu = torch.matmul(chroma_fb, spec) # (12, time)
        chroma_sum = torch.sum(chroma_gpu, dim=1) # (12)
        
        # 4. Key Estimation (using the GPU-computed chroma sum)
        key_scale = estimate_key(chroma_sum)
        
        # 5. Chord Detection (using GPU Chroma)
        chords = detect_chords(chroma_gpu, sr)
        
        # 6. Song Sections (MSAF requires file path)
        sections = get_sections(input_path)
        
        # 6. BPM and Beats (CPU librosa still best for this)
        y_cpu = y_gpu[0].cpu().numpy()
        tempo, beats = librosa.beat.beat_track(y=y_cpu, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
        
        # Format Energy Data
        rms_np = rms_gpu.cpu().numpy()
        times = np.arange(len(rms_np)) * (frame_len / sr)
        energy_data = [[round(float(t), 3), round(float(r), 4)] for t, r in zip(times, rms_np)]
        
        results = {
            "bpm": round(float(np.atleast_1d(tempo)[0]), 1),
            "key": key_scale,
            "chords": chords,
            "beats": [round(float(b), 3) for b in beat_times],
            "energy": energy_data,
            "sections": sections
        }
        
        with state.lock:
            state.jobs[job_id]["analysis"].update(results)
        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def transcribe_vocals_task(job_id: str, vocals_path: str):
    """GPU/ONNX task: Uses basic-pitch on active segments only."""
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        
        # 1. Load Vocals to GPU to find active regions
        waveform, sr = torchaudio.load(vocals_path)
        device = state.torch_device
        waveform = waveform.to(device)
        y_gpu = torch.mean(waveform, dim=0) # Mono
        
        # 2. Compute RMS in 100ms windows
        frame_len = int(sr * 0.1) # 100ms
        num_frames = y_gpu.shape[0] // frame_len
        y_trimmed = y_gpu[:num_frames * frame_len]
        frames = y_trimmed.view(num_frames, frame_len)
        rms_vals = torch.sqrt(torch.mean(frames**2, dim=1))
        
        # 3. Identify Active Segments (RMS > 0.02)
        is_active = (rms_vals > 0.02).cpu().numpy()
        
        active_segments = []
        if np.any(is_active):
            # Find continuous blocks of True
            start_idx = None
            for i, active in enumerate(is_active):
                if active and start_idx is None:
                    start_idx = i
                elif not active and start_idx is not None:
                    active_segments.append((start_idx * 0.1, i * 0.1))
                    start_idx = None
            if start_idx is not None:
                active_segments.append((start_idx * 0.1, len(is_active) * 0.1))
        
        if not active_segments:
            logging.info("No vocal activity detected above threshold.")
            with state.lock:
                state.jobs[job_id]["analysis"]["vocal_notes"] = []
            return []

        # 4. Concatenate Active Audio
        # We add a tiny 100ms silence between segments to avoid transient artifacts
        silence_gap = torch.zeros(int(sr * 0.1)).to(device)
        active_chunks = []
        mapping = [] # List of (orig_start, orig_end, concat_start)
        current_concat_time = 0.0
        
        for start_t, end_t in active_segments:
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            chunk = y_gpu[start_sample:end_sample]
            active_chunks.append(chunk)
            active_chunks.append(silence_gap)
            
            duration = end_t - start_t
            mapping.append({
                "orig_start": start_t,
                "orig_end": end_t,
                "concat_start": current_concat_time
            })
            current_concat_time += duration + 0.1 # chunk + gap

        full_active_audio = torch.cat(active_chunks)
        
        # Save temp file for basic-pitch (it needs a file path)
        temp_vocals = Path(vocals_path).parent / f"active_vocals_{job_id}.wav"
        torchaudio.save(str(temp_vocals), full_active_audio.cpu().unsqueeze(0), sr)

        # 5. Run Basic-Pitch once
        onnx_model = ICASSP_2022_MODEL_PATH.parent / "nmp.onnx"
        model_path = str(onnx_model) if onnx_model.exists() else str(ICASSP_2022_MODEL_PATH)
        
        logging.info(f"Transcribing {len(active_segments)} active vocal segments...")
        model_output, midi_data, note_events = predict(str(temp_vocals), model_or_model_path=model_path)
        
        # 6. Remap Timestamps
        vocal_notes = []
        for start, end, pitch, velocity, confidence in note_events:
            # Find which original segment this note belongs to
            orig_start = -1
            orig_end = -1
            
            for m in reversed(mapping): # Check latest segments first
                if start >= m["concat_start"]:
                    offset = start - m["concat_start"]
                    duration = m["orig_end"] - m["orig_start"]
                    if offset < duration: # Note actually in this segment
                        orig_start = m["orig_start"] + offset
                        orig_end = orig_start + (end - start)
                    break
            
            if orig_start != -1:
                vocal_notes.append({
                    "pitch": int(pitch),
                    "start": round(float(orig_start), 2),
                    "end": round(float(orig_end), 2),
                    "conf": round(float(np.atleast_1d(confidence)[0]), 2)
                })
        
        # Cleanup temp file
        if temp_vocals.exists():
            temp_vocals.unlink()
            
        with state.lock:
            state.jobs[job_id]["analysis"]["vocal_notes"] = vocal_notes
        return vocal_notes
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

async def orchestrate_parallel_job(job_id: str, input_path: str):
    """Orchestrates separation and analysis in parallel."""
    try:
        # Start both Separation (GPU) and Basic Analysis (CPU) simultaneously
        # We use run_in_executor to not block the event loop
        loop = asyncio.get_event_loop()
        
        # Start tasks
        task_sep = loop.run_in_executor(None, separate_audio_task, job_id, input_path)
        task_anal = loop.run_in_executor(None, analyze_basic_features, job_id, input_path)
        
        # Wait for both to finish (don't raise immediately to avoid race conditions with cleanup)
        results = await asyncio.gather(task_sep, task_anal, return_exceptions=True)
        
        # Check for failures in gathered tasks
        for res in results:
            if isinstance(res, Exception):
                raise res

        # Now that separation is done, we have the vocals.wav
        stems = state.jobs[job_id].get("stems", {})
        vocals_path = stems.get("vocals")
        
        if vocals_path:
            # Briefly mark as analyzing for transcription
            with state.lock:
                state.jobs[job_id]["status"] = "analyzing"
            await loop.run_in_executor(None, transcribe_vocals_task, job_id, vocals_path)
        
        with state.lock:
            state.jobs[job_id]["status"] = "completed"
            
    except Exception as e:
        with state.lock:
            state.jobs[job_id]["status"] = "failed"
            state.jobs[job_id]["error"] = str(e)
        cleanup_job(job_id)

# analyze_audio_task is no longer needed, replaced by orchestrate_parallel_job

# --- Endpoints ---

# @app.on_event("startup")
# async def startup_event():
#     load_session()

@app.get("/system-info")
async def system_info():
    import platform
    device = "cpu"
    device_name = "CPU"
    
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon"
        
    return {
        "device": device,
        "device_name": device_name,
        "cpu_info": platform.processor(),
        "platform": f"{platform.system()} {platform.release()}",
        "torch_version": torch.__version__
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
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use WAV, MP3, or FLAC.")
        
    job_id = str(uuid.uuid4())
    input_path = str(UPLOAD_DIR / f"{job_id}_{file.filename}")
    
    async with aiofiles.open(input_path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)
    
    with state.lock:
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

@app.post("/analyze")
async def analyze_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use WAV, MP3, or FLAC.")
        
    job_id = str(uuid.uuid4())
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = job_dir / f"original_{file.filename}"
    async with aiofiles.open(str(input_path), "wb") as f:
        content = await file.read()
        await f.write(content)
        
    with state.lock:
        state.jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "stems": {},
            "analysis": {},
            "created_at": time.time()
        }
        
    # Launch parallel orchestrator
    background_tasks.add_task(orchestrate_parallel_job, job_id, str(input_path))
    
    return {"job_id": job_id, "message": "Parallel separation and analysis started."}

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
    
    return FileResponse(path, media_type="audio/wav")

# Serve React build
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/dist/index.html")

# Catch-all for React Router
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # Don't catch API routes
    if full_path.startswith("api/") or full_path.startswith("job/") or full_path.startswith("setup/"):
        raise HTTPException(status_code=404, detail="Not found")
        
    index = Path("frontend/dist/index.html")
    if index.exists():
        return FileResponse(index)
    return {"error": "Frontend not built"}

# Auto-open browser once on startup
def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")

threading.Thread(target=open_browser, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
