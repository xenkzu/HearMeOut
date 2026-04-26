import { useState, useEffect, useRef, useCallback } from 'react';

export interface StemControls {
  volume: number;
  muted: boolean;
  solo: boolean;
}

export interface StemAudioData {
  buffer: AudioBuffer;
  sourceNode: AudioBufferSourceNode | null;
  gainNode: GainNode;
  analyserNode: AnalyserNode;
  controls: StemControls;
}

interface AudioEngineReturn {
  isLoaded: boolean;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  stemsData: Record<string, StemAudioData>;
  play: () => void;
  pause: () => void;
  seek: (time: number) => void;
  setVolume: (stem: string, vol: number) => void;
  toggleMute: (stem: string) => void;
  toggleSolo: (stem: string) => void;
}

export const useAudioEngine = (jobId: string, stemNames: string[]): AudioEngineReturn => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [stemsData, setStemsData] = useState<Record<string, StemAudioData>>({});

  const ctxRef = useRef<AudioContext | null>(null);
  const startTimeRef = useRef<number>(0);
  const pauseTimeRef = useRef<number>(0);
  const animFrameRef = useRef<number>(0);

  // Initialize Audio Context and load buffers
  useEffect(() => {
    if (!jobId || stemNames.length === 0) return;

    const ctx = new window.AudioContext();
    ctxRef.current = ctx;

    const loadStems = async () => {
      const data: Record<string, StemAudioData> = {};
      let maxDuration = 0;

      await Promise.all(
        stemNames.map(async (name) => {
          try {
            const res = await fetch(`/job/${jobId}/stems/${name}`);
            if (!res.ok) throw new Error(`Failed to load ${name}`);
            const arrayBuffer = await res.arrayBuffer();
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
            
            if (audioBuffer.duration > maxDuration) {
              maxDuration = audioBuffer.duration;
            }

            const gainNode = ctx.createGain();
            const analyserNode = ctx.createAnalyser();
            analyserNode.fftSize = 256;
            
            gainNode.connect(analyserNode);
            analyserNode.connect(ctx.destination);

            data[name] = {
              buffer: audioBuffer,
              sourceNode: null,
              gainNode,
              analyserNode,
              controls: { volume: 1, muted: false, solo: false }
            };
          } catch (e) {
            console.error(e);
          }
        })
      );

      setStemsData(data);
      setDuration(maxDuration);
      setIsLoaded(true);
    };

    loadStems();

    return () => {
      ctx.close();
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [jobId]);

  // Sync controls with gain values
  useEffect(() => {
    const isAnySolo = Object.values(stemsData).some(s => s.controls.solo);
    
    Object.values(stemsData).forEach(stem => {
      if (stem.controls.muted || (isAnySolo && !stem.controls.solo)) {
        stem.gainNode.gain.value = 0;
      } else {
        stem.gainNode.gain.value = stem.controls.volume;
      }
    });
  }, [stemsData]);

  const updateTime = useCallback(() => {
    if (!ctxRef.current || !isPlaying) return;
    setCurrentTime(ctxRef.current.currentTime - startTimeRef.current);
    animFrameRef.current = requestAnimationFrame(updateTime);
  }, [isPlaying]);

  const play = useCallback(() => {
    if (!ctxRef.current || !isLoaded) return;
    
    // Stop existing sources
    Object.values(stemsData).forEach(stem => {
      if (stem.sourceNode) {
        try { stem.sourceNode.stop(); } catch(e){}
      }
    });

    const newData = { ...stemsData };
    
    Object.keys(newData).forEach(name => {
      const source = ctxRef.current!.createBufferSource();
      source.buffer = newData[name].buffer;
      source.connect(newData[name].gainNode);
      source.start(0, pauseTimeRef.current);
      newData[name].sourceNode = source;
    });

    setStemsData(newData);
    startTimeRef.current = ctxRef.current.currentTime - pauseTimeRef.current;
    
    if (ctxRef.current.state === 'suspended') {
      ctxRef.current.resume();
    }
    
    setIsPlaying(true);
    animFrameRef.current = requestAnimationFrame(updateTime);
  }, [isLoaded, stemsData, updateTime]);

  const pause = useCallback(() => {
    if (!ctxRef.current || !isPlaying) return;
    
    Object.values(stemsData).forEach(stem => {
      if (stem.sourceNode) {
        try { stem.sourceNode.stop(); } catch(e){}
      }
    });

    pauseTimeRef.current = ctxRef.current.currentTime - startTimeRef.current;
    setIsPlaying(false);
    cancelAnimationFrame(animFrameRef.current);
  }, [isPlaying, stemsData]);

  const seek = useCallback((time: number) => {
    pauseTimeRef.current = time;
    setCurrentTime(time);
    if (isPlaying) {
      pause();
      play(); // Restart from new time
    }
  }, [isPlaying, play, pause]);

  const setVolume = (stem: string, vol: number) => {
    setStemsData(prev => ({
      ...prev,
      [stem]: { ...prev[stem], controls: { ...prev[stem].controls, volume: vol } }
    }));
  };

  const toggleMute = (stem: string) => {
    setStemsData(prev => ({
      ...prev,
      [stem]: { ...prev[stem], controls: { ...prev[stem].controls, muted: !prev[stem].controls.muted } }
    }));
  };

  const toggleSolo = (stem: string) => {
    setStemsData(prev => ({
      ...prev,
      [stem]: { ...prev[stem], controls: { ...prev[stem].controls, solo: !prev[stem].controls.solo } }
    }));
  };

  return {
    isLoaded, isPlaying, currentTime, duration, stemsData,
    play, pause, seek, setVolume, toggleMute, toggleSolo
  };
};
