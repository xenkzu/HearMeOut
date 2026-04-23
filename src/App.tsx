import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [status, setStatus] = useState<"connecting" | "connected" | "failed">("connecting");
  const [pingCount, setPingCount] = useState(0);

  const checkHealth = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8765/health");
      if (response.ok) {
        console.log("Health check success");
        setStatus("connected");
      } else {
        console.warn("Health check returned non-OK status:", response.status);
        setStatus("failed");
      }
    } catch (err) {
      console.error("Health check failed:", err);
      setStatus("failed");
    }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      checkHealth();
      setPingCount(prev => prev + 1);
    }, 2000);
    
    checkHealth(); // Initial check
    
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="container">
      <div className="glass-card">
        <div className="header">
          <div className="logo-container">
            <div className="pulse-ring"></div>
            <img src="/tauri.svg" className="logo" alt="Tauri logo" />
          </div>
          <h1>Tauri Python Bridge</h1>
          <p className="subtitle">React + Vite + FastAPI Sidecar</p>
        </div>

        <div className="status-section">
          <div className={`status-badge ${status}`}>
            <span className="dot"></span>
            {status === "connecting" && "Connecting to API..."}
            {status === "connected" && "API Connected"}
            {status === "failed" && "API Offline"}
          </div>
          
          <div className="stats">
            <div className="stat-item">
              <span className="label">Endpoint</span>
              <span className="value">localhost:8765</span>
            </div>
            <div className="stat-item">
              <span className="label">Health Checks</span>
              <span className="value">{pingCount}</span>
            </div>
          </div>
        </div>

        <div className="action-section">
          <button 
            className="refresh-btn" 
            onClick={() => {
              setStatus("connecting");
              checkHealth();
            }}
          >
            Manual Reconnect
          </button>
        </div>

        <div className="info-section">
          <h3>How it works</h3>
          <ul>
            <li>Tauri spawns <code>api.exe</code> as a sidecar process</li>
            <li>Python FastAPI runs on port 8765</li>
            <li>Frontend polls <code>/health</code> endpoint</li>
            <li>Process is killed automatically on exit</li>
          </ul>
        </div>
      </div>
    </main>
  );
}

export default App;
