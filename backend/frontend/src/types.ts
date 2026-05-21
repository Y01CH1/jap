export type Verdict = 'likely_ai' | 'likely_real' | 'uncertain';

export interface DetectionResponse {
  id: string;
  score: number;
  verdict: Verdict;
  model_used: string;
  inference_time_ms: number;
}

export interface ErrorResponse {
  detail: string;
  code: string;
}

export type DetectionState = 'idle' | 'uploading' | 'detecting' | 'done' | 'error';

export interface HistoryEntry {
  id: string;
  fileName: string;
  thumbnailUrl: string;
  score: number;
  verdict: Verdict;
  timestamp: number;
}
