import React from 'react';
import { ChevronLeft, User } from 'lucide-react';

export const Topbar: React.FC = () => {
  return (
    <div className="topbar" style={{ 
      display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
      padding: '24px 32px', background: 'linear-gradient(rgba(10,10,11,0.8), transparent)' 
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <button className="icon-btn"><ChevronLeft size={20} /></button>
        <span className="syne-font" style={{ fontSize: '16px', fontWeight: 600 }}>Echo Mirage</span>
      </div>
      
      <div className="phase-pills" style={{ 
        display: 'flex', background: 'var(--surface)', borderRadius: '20px', padding: '4px' 
      }}>
        <div style={{ padding: '6px 16px', fontSize: '12px', color: 'var(--text-secondary)' }}>Pre Production</div>
        <div style={{ padding: '6px 16px', fontSize: '12px', background: 'var(--accent)', color: '#000', borderRadius: '16px', fontWeight: 500 }}>Production</div>
        <div style={{ padding: '6px 16px', fontSize: '12px', color: 'var(--text-secondary)' }}>Post Production</div>
      </div>
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span style={{ fontSize: '14px' }}>Mason Cole</span>
        <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'var(--surface)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <User size={16} />
        </div>
      </div>
    </div>
  );
};
