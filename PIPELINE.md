# HearMeOut Backend Pipeline Documentation

This document outlines the high-fidelity audio separation and music intelligence pipeline implemented in the HearMeOut backend.

## 1. System Architecture & Hardware
- **Core Engine**: PyTorch-native implementation.
- **Hardware Acceleration**: 
    - **GPU (NVIDIA RTX 4060)**: Used for Audio Separation (Demucs) and Vocal Transcription (Basic-Pitch ONNX).
    - **CPU**: Used for Audio Pre-processing, Librosa analysis (BPM/Key), and Filesystem I/O.
- **Framework**: FastAPI (Asynchronous Job Queue).

---

## 2. Component Flow

### Step A: Entry Point (`/analyze` endpoint)
1. **User Action**: Uploads `song.mp3`.
2. **FastAPI**: Receives file, generates a `job_id`, and saves `original.mp3`.
3. **Parallel Orchestration**: Spawns `orchestrate_parallel_job` which launches two simultaneous background threads.

### Step B: Simultaneous Processing (GPU & CPU Parallel)
This stage uses `asyncio.gather` to run separation and basic analysis at the same time.

#### B1: Audio Separation (GPU)
- **Function**: `separate_audio_task()`
- **Hardware**: Heavy GPU utilization (Demucs v4).
- **Status**: Job shows `separating`.

#### B2: Basic Analysis (GPU & CPU)
- **Function**: `analyze_basic_features()`
- **Hardware**: 
    - **GPU (CUDA)**: `torchaudio` (Audio Loading), `torch` (RMS Energy, Chroma Spectrogram), and **Native GPU Chord Detection** (Template Matching).
    - **CPU**: `msaf` (Structural Segmentation via Foote), `librosa` (Beat Tracking).
- **Tasks**: BPM, Key Estimation, **Chord Progression**, Energy Map, and Structural Sections.
- **Status**: Job shows `analyzing`.

### Step C: Sequential Post-Processing
1. **Vocal Activity Detection (VAD)**: Before transcription, the system scans the `vocals.wav` stem using a GPU-accelerated RMS gate (100ms windows, 0.02 threshold).
2. **Dynamic Concatenation**: Only "active" vocal regions are concatenated into a temporary buffer. This skips long instrumental silences, significantly reducing inference time.
3. **Vocal Transcription**: `basic_pitch` runs on the concatenated audio.
4. **Timestamp Remapping**: The system mathematically remaps note timestamps from the concatenated buffer back to their original positions in the song.
5. **Final Merge**: All JSON results are merged into the final `analysis.json`.

### Step D: Data Aggregation & Storage
1. **Final JSON**: Combines all features into a single `analysis.json`.
2. **Status Update**: Job status switches from `processing` -> `completed`.
3. **Client Polling**: User calls `GET /job/{id}/status` to retrieve the full map and download links.

---

## 3. Directory Structure (Data Flow)
```text
C:\Users\yashk\AppData\Local\Packages\... (Sandbox Redirect)
└── LocalCache\Local\hearmeout\
    ├── models\             (Optional ONNX cache)
    └── output\
        └── {job_id}\
            ├── original.mp3     (Input)
            ├── drums.wav        (Stem)
            ├── bass.wav         (Stem)
            ├── ...
            └── analysis.json    (The "Musical Map")
```

---

## 4. Summary of Models
| Task | Model / Library | Hardware |
| :--- | :--- | :--- |
| **Separation** | HTDemucs v4 (6-Stem) | GPU (CUDA) |
| **BPM / Key** | Librosa (Signal Processing) | CPU |
| **Vocal Pitch** | Spotify Basic-Pitch | GPU (ONNX) |
| **API / Queue** | FastAPI + Uvicorn | CPU |
