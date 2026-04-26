import React from 'react';
import { Maximize2, LayoutGrid, ListMusic, Mic2 } from 'lucide-react';

interface TabBarProps {
  activeTab: 'overview' | 'tracks' | 'lyrics';
  onTabChange: (tab: 'overview' | 'tracks' | 'lyrics') => void;
}

export const TabBar: React.FC<TabBarProps> = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'overview', label: 'Overview', icon: <LayoutGrid size={16} /> },
    { id: 'tracks', label: 'Tracks', icon: <ListMusic size={16} /> },
    { id: 'lyrics', label: 'Lyrics', icon: <Mic2 size={16} /> }
  ] as const;

  return (
    <div style={{ display: 'flex', gap: '16px', padding: '0 32px' }}>
      {tabs.map(tab => (
        <div 
          key={tab.id}
          onClick={() => onTabChange(tab.id)}
          style={{ 
            flex: 1, 
            display: 'flex', 
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '16px 24px',
            background: activeTab === tab.id ? 'var(--surface-hover)' : 'var(--surface)',
            borderRadius: '16px',
            cursor: 'pointer',
            border: activeTab === tab.id ? '1px solid var(--border)' : '1px solid transparent',
            transition: 'all 0.3s ease'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: activeTab === tab.id ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
            {tab.icon}
            <span className="syne-font" style={{ fontSize: '14px', fontWeight: 500 }}>{tab.label}</span>
          </div>
          <Maximize2 size={14} color="var(--text-secondary)" />
        </div>
      ))}
    </div>
  );
};
