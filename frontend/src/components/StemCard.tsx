import React, { useRef, useEffect } from 'react';
import { StemAudioData } from '../hooks/useAudioEngine';

interface StemCardProps {
  name: string;
  data: StemAudioData;
  isAnySolo: boolean;
  onVolumeChange: (vol: number) => void;
  onMuteToggle: () => void;
  onSoloToggle: () => void;
}

export const StemCard: React.FC<StemCardProps> = ({ 
  name, data, isAnySolo, onVolumeChange, onMuteToggle, onSoloToggle 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  const isMuted = data.controls.muted;
  const isSolo = data.controls.solo;
  const isActive = isSolo || (!isAnySolo && !isMuted);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const analyser = data.analyserNode;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      const barWidth = (canvas.width / bufferLength) * 2.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = dataArray[i] / 2;
        
        ctx.fillStyle = isActive ? `rgba(255, 255, 255, ${barHeight/150 + 0.1})` : 'rgba(255, 255, 255, 0.1)';
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
      }
    };

    draw();

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [data.analyserNode, isActive]);

  return (
    <div className={`stem-card ${isSolo ? 'solo-active' : ''}`} style={{ opacity: isActive ? 1 : 0.4 }}>
      <div className="stem-header">
        <h3 className="syne-font uppercase">{name}</h3>
        <div className="stem-actions">
          <button 
            className={`control-pill ${isMuted ? 'active-red' : ''}`}
            onClick={onMuteToggle}
          >
            M
          </button>
          <button 
            className={`control-pill ${isSolo ? 'active-white' : ''}`}
            onClick={onSoloToggle}
          >
            S
          </button>
        </div>
      </div>
      
      <div className="stem-body">
        <canvas ref={canvasRef} width={200} height={80} className="waveform-canvas" />
        <input 
          type="range" 
          min="0" max="1" step="0.01" 
          value={data.controls.volume}
          onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
          className="vol-slider vertical"
        />
      </div>
    </div>
  );
};
