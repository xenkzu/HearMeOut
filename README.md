# HearMeOut: A High-Performance GPU-Accelerated Audio Separation and Music Intelligence System

## Abstract
HearMeOut is a high-fidelity audio separation and music intelligence pipeline designed to extract stems and comprehensive musical features from raw audio files. By leveraging CPU and GPU parallelization, the system efficiently handles heavy computational tasks such as source separation (using HTDemucs), chord detection, structural segmentation, beat tracking, and vocal pitch transcription. This document details the architectural decisions, technology stack, and analytical formulations that constitute the HearMeOut engine.

## 1. Introduction
The analysis of digital audio involves significant computational overhead, especially when employing deep learning models for source separation and feature extraction. The primary objective of HearMeOut is to provide a unified, fast, and scalable pipeline that transforms an ordinary audio file into a rich "Musical Map." This map includes separated audio stems (vocals, drums, bass, etc.), BPM, musical key, chord progressions, structural sections, and vocal note transcriptions.

## 2. Methodology and Pipeline Architecture
The pipeline is designed to minimize I/O bottlenecks and latency through asynchronous execution and parallel hardware utilization.

### 2.1 Parallel Orchestration
Upon receiving an audio file via the FastAPI endpoint, the system generates a unique job ID and launches two simultaneous background threads:
- **Branch 1: Audio Separation (GPU-bound):** Utilizes the HTDemucs v4 (6-Stem) model. This branch is heavily GPU-dependent and outputs individual audio stems.
- **Branch 2: Basic Feature Analysis (GPU/CPU-bound):** Computes BPM, Key, Chords, Energy, and Structural sections. It intelligently routes tensor operations (like RMS and Chroma) to the GPU while relying on the CPU for complex heuristic algorithms (like Beat Tracking).

### 2.2 Sequential Post-Processing
Once Branch 1 completes and the vocal stem is isolated, the pipeline enters a sequential phase for vocal transcription:
- **Vocal Activity Detection (VAD):** Rather than passing the entire vocal stem to the transcription model, the system scans the stem using a GPU-accelerated RMS gate. 
- **Dynamic Concatenation:** Only regions exceeding the VAD threshold are concatenated (with a 100ms silence gap) and sent to the Spotify Basic-Pitch model. This design choice skips long instrumental silences, exponentially reducing inference time. The output notes are then mathematically remapped to their original timestamps.

## 3. Technology Stack and Language Choices
The project is divided into a robust backend and a highly responsive frontend.

### 3.1 Backend: Python
Python was selected as the backend language due to its unparalleled ecosystem for deep learning and audio digital signal processing (DSP). 
- **Framework:** FastAPI with Uvicorn. Chosen for its native asynchronous capabilities, making it ideal for managing long-running background tasks and Server-Sent Events (SSE) for progress polling.
- **Deep Learning Framework:** PyTorch & TorchAudio. Enables seamless tensor operations on the GPU (CUDA), drastically speeding up spectrogram generation and matrix multiplications.
- **DSP Libraries:** `librosa` (for beat tracking and chroma filters) and `msaf` (for structural segmentation).

### 3.2 Frontend: TypeScript & React (Vite)
- **Languages:** TypeScript provides type safety, which is crucial when handling complex, nested JSON responses representing the "Musical Map."
- **Framework:** React, bundled with Vite, offers a lightweight, component-driven architecture capable of dynamically rendering real-time job status updates and interactive audio visualizations.

## 4. Analytical Methods and Formulations
The backend employs several distinct mathematical and algorithmic approaches to extract musical intelligence:

### 4.1 Energy (RMS) Calculation
Energy is computed on the GPU using the Root Mean Square (RMS) of the audio signal over 100ms windows.
$$ \text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2} $$
where $N = 4410$ (for a 44.1kHz sample rate).

### 4.2 Chroma Spectrogram and Key Estimation
- **Spectrogram:** Computed via a Short-Time Fourier Transform (STFT) with $N_{fft} = 4096$ and a hop length of 1024, yielding high frequency resolution.
- **Chroma Filter Bank:** Transforms the power spectrogram into a 12-bin representation of pitch classes (C, C#, D, etc.).
- **Key Estimation:** The chroma vector is summed over time. The system calculates the Pearson correlation coefficient between this sum and predefined major/minor profiles (Krumhansl-Schmuckler style profiles). The key with the maximum correlation is selected.

### 4.3 GPU-Accelerated Chord Detection
Chord detection utilizes Template Matching via Cosine Similarity on the GPU.
- **Templates:** Binary vectors are created for all 12 major (intervals: 0, 4, 7) and 12 minor (intervals: 0, 3, 7) chords.
- **Similarity:** Computes the dot product between the $L2$-normalized chroma matrix and the normalized chord templates.
- **Smoothing:** A 1D Average Pooling filter (kernel size = 21, stride = 1) is applied over time to smooth transitions, followed by an $\text{argmax}$ operation to find the most probable chord per timeframe.

### 4.4 Structural Segmentation
Utilizes the Foote algorithm (via `msaf`), which computes a self-similarity matrix of the audio features. It applies a checkerboard kernel along the diagonal to detect points of high novelty, which correspond to section boundaries (e.g., Verse, Chorus).

### 4.5 Vocal Activity Detection (VAD) Thresholding
The vocal stem is segmented into 100ms frames. A boolean mask is generated where $\text{RMS}_{frame} > 0.02$. Only contiguous blocks of `True` are processed for pitch transcription, filtering out background noise and bleed.

## 5. System Requirements and Installation
The engine is heavily optimized for NVIDIA GPUs but includes CPU fallbacks.

### Installation
```bash
git clone https://github.com/xenkzu/hearmeout
cd hearmeout
pip install -r requirements.txt
python run.py
```
*Note: The frontend is built automatically on the first run, and the application opens in your default browser.*

## 6. Conclusion
HearMeOut demonstrates a highly optimized approach to audio intelligence. By meticulously routing tasks between the CPU and GPU and employing dynamic techniques like VAD-based concatenation, the pipeline minimizes computational waste while delivering state-of-the-art source separation and musical feature extraction.
