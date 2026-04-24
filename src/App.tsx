import { useState, useEffect } from "react";
import "./App.css";

interface SystemInfo {
  device: string;
  device_name: string;
  platform: string;
  torch_version: string;
}

interface DownloadStatus {
  status: "idle" | "downloading" | "complete" | "failed";
  percentage: number;
  current_file: string;
  error: string | null;
}

function App() {
  const [status, setStatus] = useState<"connecting" | "connected" | "failed">("connecting");
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus>({
    status: "idle",
    percentage: 0,
    current_file: "",
    error: null
  });
  const [pingCount, setPingCount] = useState(0);

  const fetchSystemInfo = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8765/system-info");
      if (response.ok) {
        const data = await response.json();
        setSystemInfo(data);
      }
    } catch (err) {
      console.error("Failed to fetch system info:", err);
    }
  };

  const fetchDownloadProgress = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8765/download-progress");
      if (response.ok) {
        const data = await response.json();
        setDownloadStatus(data);
      }
    } catch (err) {
      console.error("Failed to fetch download progress:", err);
    }
  };

  const startDownload = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8765/download-models", { method: "POST" });
      if (response.ok) {
        setDownloadStatus(prev => ({ ...prev, status: "downloading" }));
      }
    } catch (err) {
      console.error("Failed to start download:", err);
    }
  };

  const checkHealth = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8765/health");
      if (response.ok) {
        setStatus("connected");
        if (!systemInfo) fetchSystemInfo();
      } else {
        setStatus("failed");
      }
    } catch (err) {
      setStatus("failed");
    }
  };

  useEffect(() => {
    const healthInterval = setInterval(() => {
      checkHealth();
      setPingCount(prev => prev + 1);
    }, 2000);
    
    // Poll system info if not yet detected
    let systemInfoInterval: number | undefined;
    if (status === "connected" && (!systemInfo || systemInfo.device === "detecting")) {
      systemInfoInterval = window.setInterval(fetchSystemInfo, 3000);
    }
    
    checkHealth();
    
    return () => {
      clearInterval(healthInterval);
      if (systemInfoInterval) clearInterval(systemInfoInterval);
    };
  }, [status, systemInfo]);

  useEffect(() => {
    let downloadInterval: number | undefined;
    if (downloadStatus.status === "downloading") {
      downloadInterval = window.setInterval(fetchDownloadProgress, 1000);
    }
    return () => {
      if (downloadInterval) clearInterval(downloadInterval);
    };
  }, [downloadStatus.status]);

  return (
    <main className="container">
      <div className="glass-card">
        <div className="header">
          <div className="logo-container">
            <div className="pulse-ring"></div>
            <img src="/tauri.svg" className="logo" alt="Tauri logo" />
          </div>
          <h1 className="title">Hear Me Out</h1>
          <p className="subtitle">AI-Powered Audio Visualizer</p>
        </div>

        <div className="status-section">
          <div className={`status-badge ${status}`}>
            <span className="dot"></span>
            {status === "connecting" && "Connecting to Backend..."}
            {status === "connected" && "System Ready"}
            {status === "failed" && "Backend Offline"}
          </div>
          
          {systemInfo && (
            <div className="hardware-info">
              <div className="hw-badge">
                <span className="hw-icon">{systemInfo.device === "cuda" ? "⚡" : systemInfo.device === "mps" ? "🍎" : "💻"}</span>
                <span className="hw-label">{systemInfo.device_name} detected</span>
              </div>
            </div>
          )}
        </div>

        {downloadStatus.status !== "complete" && status === "connected" && (
          <div className="download-section">
            <div className="download-header">
              <h3>Model Weights Required</h3>
              <p>Demucs htdemucs_6s model is needed for source separation.</p>
            </div>
            
            {downloadStatus.status === "idle" ? (
              <button className="primary-btn" onClick={startDownload}>
                Download Model Weights
              </button>
            ) : (
              <div className="progress-container">
                <div className="progress-bar-bg">
                  <div 
                    className="progress-bar-fill" 
                    style={{ width: `${downloadStatus.percentage}%` }}
                  ></div>
                </div>
                <div className="progress-text">
                  <span>{downloadStatus.status === "downloading" ? "Downloading..." : "Error"}</span>
                  <span>{downloadStatus.percentage}%</span>
                </div>
              </div>
            )}
          </div>
        )}

        {downloadStatus.status === "complete" && (
          <div className="ready-section">
            <div className="ready-badge">✨ Model Ready for Inference</div>
          </div>
        )}

        <div className="stats">
          <div className="stat-item">
            <span className="label">Backend Status</span>
            <span className="value">{status === "connected" ? "OK" : "Error"}</span>
          </div>
          <div className="stat-item">
            <span className="label">Platform</span>
            <span className="value">{systemInfo?.platform || "Detecting..."}</span>
          </div>
        </div>

        <div className="info-section">
          <h3>Advanced Architecture</h3>
          <ul>
            <li>Sidecar: FastAPI on port 8765</li>
            <li>Compute: {systemInfo?.device_name || "Detecting..."}</li>
            <li>Torch Version: {systemInfo?.torch_version || "..."}</li>
          </ul>
        </div>
      </div>
    </main>
  );
}

export default App;
