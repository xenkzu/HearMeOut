export interface Section {
  label: string;
  start: number;
  end: number;
}

export interface Chord {
  chord: string;
  start: number;
  end: number;
}

export interface VocalNote {
  pitch: number;
  start: number;
  end: number;
  conf: number;
}

export interface AnalysisData {
  bpm: number;
  key: string;
  beats: number[];
  energy: number[][]; // [time, rms][]
  sections: Section[];
  chords: Chord[];
  vocal_notes: VocalNote[];
}

export interface StemPaths {
  drums?: string;
  bass?: string;
  vocals?: string;
  guitar?: string;
  piano?: string;
  other?: string;
  [key: string]: string | undefined;
}

export interface JobData {
  id: string;
  status: string;
  stems: StemPaths;
  analysis: AnalysisData;
  created_at: number;
}
