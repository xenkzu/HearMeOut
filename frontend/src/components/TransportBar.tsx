import React from 'react';
import { Play, Pause, SkipBack, Rewind, Square, Repeat, Plus, Download, ChevronUp } from 'lucide-react';
import { AnalysisData } from '../types/analysis';

interface TransportBarProps {
  engine: any; // Using any for brevity here, should type properly
  analysisData: AnalysisData;
}

const formatTime = (time: number) => {
  const mins = Math.floor(time / 60);
  const secs = Math.floor(time % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

export const TransportBar: React.FC<TransportBarProps> = ({ engine, analysisData: _ }) => {
  const { isPlaying, currentTime, duration, play, pause, seek } = engine;

  return (
    <div className="transport-bar" style={{ padding: '24px 32px', background: '#0a0a0b', borderTop: '1px solid var(--border)' }}>
      {/* File Info and Expand (like ref image) */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <span className="mono" style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Minimize View</span>
          <span className="mono" style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Player View</span>
          <span className="mono" style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Comments View</span>
        </div>
        <span className="syne-font" style={{ fontSize: '14px', fontWeight: 600 }}>Glass Skin (Mix-01).wav</span>
        <button><ChevronUp size={20} color="var(--text-secondary)" /></button>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {/* Time */}
        <div className="mono" style={{ fontSize: '20px', width: '200px' }}>
          <span style={{ color: 'var(--text-primary)' }}>{formatTime(currentTime)}</span>
          <span style={{ color: 'var(--text-secondary)' }}> / {formatTime(duration)}</span>
        </div>

        {/* Core Controls */}
        <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
          <button style={{ color: 'var(--text-secondary)' }}><SkipBack size={20} fill="currentColor" /></button>
          <button style={{ color: 'var(--text-secondary)' }}><Rewind size={20} fill="currentColor" /></button>
          <button style={{ color: 'var(--text-secondary)' }}><SkipBack size={20} fill="currentColor" /></button>
          
          <button onClick={isPlaying ? pause : play} style={{ color: 'var(--text-primary)' }}>
            {isPlaying ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" />}
          </button>
          
          <button style={{ color: 'var(--text-secondary)' }}><Square size={20} fill="currentColor" /></button>
          <button style={{ color: 'var(--text-secondary)' }}><Repeat size={20} /></button>
        </div>

        {/* Right Tools */}
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center', width: '200px', justifyContent: 'flex-end' }}>
          <button style={{ color: 'var(--text-primary)' }}><Plus size={20} /></button>
          <button style={{ color: 'var(--text-primary)' }}><div style={{ width: '20px', height: '20px', border: '2px solid currentColor', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><div style={{ width: '8px', height: '8px', background: 'currentColor', borderRadius: '50%' }} /></div></button>
          <button style={{ color: 'var(--text-primary)' }}><Download size={20} /></button>
          
          <div style={{ display: 'flex', border: '1px solid var(--border)', borderRadius: '8px', overflow: 'hidden' }}>
            <button style={{ padding: '8px 16px', fontSize: '12px', borderRight: '1px solid var(--border)' }}>A</button>
            <button style={{ padding: '8px 16px', fontSize: '12px', background: 'var(--surface)' }}>B</button>
          </div>
        </div>
      </div>

      {/* Progress Bar (Full width underneath) */}
      <div style={{ marginTop: '24px', position: 'relative' }}>
        <input 
          type="range" 
          min="0" max={duration || 100} 
          value={currentTime}
          onChange={(e) => seek(parseFloat(e.target.value))}
          style={{ 
            width: '100%', height: '4px', background: 'var(--surface-hover)', 
            borderRadius: '2px', outline: 'none' 
          }}
        />
        <div style={{ 
          position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)', 
          height: '4px', background: 'var(--text-primary)', 
          width: `${(currentTime / duration) * 100}%`, pointerEvents: 'none',
          borderRadius: '2px'
        }} />
      </div>
    </div>
  );
};
