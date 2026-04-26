import React from 'react';
import { Star, BarChart2, Settings2, User, Sliders } from 'lucide-react';

export const SideRail: React.FC = () => {
  return (
    <div className="side-rail" style={{ 
      display: 'flex', flexDirection: 'column', gap: '24px', 
      padding: '24px', justifyContent: 'center', background: 'rgba(255,255,255,0.02)',
      borderRadius: '30px 0 0 30px', marginRight: '16px'
    }}>
      <button style={{ color: 'var(--text-secondary)' }}><Star size={20} /></button>
      <button style={{ background: 'var(--accent)', color: '#000', padding: '12px', borderRadius: '50%' }}>
        <BarChart2 size={20} />
      </button>
      <button style={{ color: 'var(--text-secondary)' }}><Settings2 size={20} /></button>
      <button style={{ color: 'var(--text-secondary)' }}><User size={20} /></button>
      <button style={{ color: 'var(--text-secondary)' }}><Sliders size={20} /></button>
    </div>
  );
};
