import { useState, useEffect, useRef } from "react";
import "./App.css";
import { Visualizer } from "./components/Visualizer.tsx";
import { AnalysisData, StemPaths } from "./types/analysis";

function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle"); // idle, uploading, pending, separating, analyzing, completed, failed
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [stemPaths, setStemPaths] = useState<StemPaths | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let interval: number;

    if (jobId && !["completed", "failed"].includes(status)) {
      interval = window.setInterval(async () => {
        try {
          const res = await fetch(`/job/${jobId}/status`);
          if (res.ok) {
            const data = await res.json();
            setStatus(data.status);
            if (data.error) setError(data.error);

            if (data.status === "completed") {
              setAnalysisData(data.analysis);
              setStemPaths(data.stems);
              clearInterval(interval);
            }
          }
        } catch (err) {
          console.error("Failed to poll status", err);
        }
      }, 2000);
    }

    return () => clearInterval(interval);
  }, [jobId, status]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setStatus("uploading");
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      setJobId(data.job_id);
      setStatus("pending"); // FastAPI says pending initially
    } catch (err: any) {
      setError(err.message);
      setStatus("failed");
    }
  };

  if (status === "completed" && analysisData && stemPaths && jobId) {
    return (
      <Visualizer 
        jobId={jobId} 
        analysisData={analysisData} 
        stemPaths={stemPaths} 
      />
    );
  }

  // Upload Screen
  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center', 
      height: '100vh', 
      backgroundColor: 'black',
      color: 'white'
    }}>
      <h1 style={{ fontSize: '3rem', marginBottom: '1rem', fontWeight: 600 }}>Hear Me Out</h1>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '3rem' }}>Upload an audio file to separate stems and analyze.</p>
      
      {status === "idle" || status === "failed" ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <input 
            type="file" 
            accept="audio/wav, audio/mp3, audio/flac, audio/mpeg" 
            style={{ display: 'none' }} 
            ref={fileInputRef}
            onChange={handleFileUpload}
          />
          <button 
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: '16px 32px',
              backgroundColor: 'var(--accent)',
              color: 'black',
              borderRadius: '30px',
              fontSize: '1.2rem',
              fontWeight: 600
            }}
          >
            Select Audio File
          </button>
          {error && <p style={{ color: 'var(--red)', marginTop: '20px' }}>{error}</p>}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div style={{ 
            width: '60px', height: '60px', borderRadius: '50%', 
            border: '4px solid var(--surface)', borderTopColor: 'var(--accent)', 
            animation: 'spin 1s linear infinite', marginBottom: '20px'
          }} />
          <style>{"@keyframes spin { 100% { transform: rotate(360deg); } }"}</style>
          <h2 style={{ textTransform: 'capitalize' }}>
            {status === "uploading" ? "Uploading..." : `Processing: ${status}...`}
          </h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: '10px' }}>
            This will take a few minutes on the GPU.
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
