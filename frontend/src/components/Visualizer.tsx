import React, { useState } from 'react';
import { AnalysisData, StemPaths } from '../types/analysis';
import { useAudioEngine } from '../hooks/useAudioEngine';
import { StemCard } from './StemCard.tsx';
import { Topbar } from './Topbar.tsx';
import { SideRail } from './SideRail.tsx';
import { HeroSection } from './HeroSection.tsx';
import { TransportBar } from './TransportBar.tsx';
import { TabBar } from './TabBar.tsx';

interface VisualizerProps {
  jobId: string;
  analysisData: AnalysisData;
  stemPaths: StemPaths;
}

export const Visualizer: React.FC<VisualizerProps> = ({ jobId, analysisData, stemPaths }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'tracks' | 'lyrics'>('tracks');
  // Ensure vocals is first or whatever order we want. Let's just use all provided keys.
  const allStems = Object.keys(stemPaths).filter(k => stemPaths[k]);

  const engine = useAudioEngine(jobId, allStems);

  const isAnySolo = Object.values(engine.stemsData).some(s => s.controls.solo);

  return (
    <div className="visualizer-layout">
      <Topbar />
      
      <HeroSection analysisData={analysisData} />
      
      <SideRail />
      
      <div className="tab-bar-container">
        <TabBar activeTab={activeTab} onTabChange={setActiveTab} />
        
        <div className="tab-content" style={{ padding: '20px', height: '300px', overflowY: 'auto' }}>
          {activeTab === 'tracks' && (
            <div className="tracks-grid" style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
              gap: '16px' 
            }}>
              {allStems.map((stemName, i) => (
                <div key={stemName} style={{ animation: `fadeUp 0.4s ease forwards ${i * 0.05}s`, opacity: 0 }}>
                  {engine.stemsData[stemName] && (
                    <StemCard 
                      name={stemName}
                      data={engine.stemsData[stemName]}
                      isAnySolo={isAnySolo}
                      onVolumeChange={(v) => engine.setVolume(stemName, v)}
                      onMuteToggle={() => engine.toggleMute(stemName)}
                      onSoloToggle={() => engine.toggleSolo(stemName)}
                    />
                  )}
                </div>
              ))}
            </div>
          )}

          {activeTab === 'overview' && (
            <div className="overview-tab fade-up">
              <h2 className="syne-font">Song Overview</h2>
              <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
                <div className="data-pill">BPM: <span className="mono">{analysisData.bpm}</span></div>
                <div className="data-pill">Key: <span className="mono">{analysisData.key}</span></div>
              </div>
              <div className="chords-scroll" style={{ display: 'flex', overflowX: 'auto', gap: '8px', paddingBottom: '10px' }}>
                {analysisData.chords.map((chord, i) => (
                  <div key={i} className="chord-pill mono" style={{
                    background: 'var(--surface)', padding: '4px 12px', borderRadius: '12px',
                    border: '1px solid var(--border)', flexShrink: 0
                  }}>
                    {chord.chord}
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'lyrics' && (
            <div className="lyrics-tab flex-center" style={{ height: '100%', color: 'var(--text-secondary)' }}>
              <h2>Lyrics coming soon</h2>
            </div>
          )}
        </div>
      </div>

      <TransportBar engine={engine} analysisData={analysisData} />
    </div>
  );
};
