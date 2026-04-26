import React from 'react';
import { AnalysisData } from '../types/analysis';

interface HeroProps {
  analysisData: AnalysisData;
}

export const HeroSection: React.FC<HeroProps> = ({ analysisData: _ }) => {
  return (
    <div className="hero-section" style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden', backgroundColor: 'black' }}>
      {/* Bottom Gradient Overlay to fade smoothly into the UI background */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0, height: '60%',
        background: 'linear-gradient(to bottom, transparent, var(--bg))',
        pointerEvents: 'none'
      }} />
    </div>
  );
};
