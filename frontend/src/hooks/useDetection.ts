import { useCallback, useState, useRef, useEffect } from 'react';
import { DetectionResponse, DetectionState, HistoryEntry } from '../types';
import { detectImage } from '../services/api';

interface UseDetectionReturn {
  state: DetectionState;
  progress: number;
  result: DetectionResponse | null;
  error: string | null;
  history: HistoryEntry[];
  submit: (file: File) => Promise<void>;
  reset: () => void;
}

export function useDetection(): UseDetectionReturn {
  const [state, setState] = useState<DetectionState>('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const objectUrls = useRef<string[]>([]);

  const submit = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setState('uploading');
    setProgress(0);

    const thumbnailUrl = URL.createObjectURL(file);
    objectUrls.current.push(thumbnailUrl);

    try {
      const data = await detectImage(file, (pct) => {
        setProgress(pct);
        if (pct >= 100) setState('detecting');
      });
      setResult(data);

      setHistory((prev) => [
        {
          id: data.id,
          fileName: file.name,
          thumbnailUrl,
          score: data.score,
          verdict: data.verdict,
          timestamp: Date.now(),
        },
        ...prev,
      ]);

      setState('done');
    } catch (err: unknown) {
      URL.revokeObjectURL(thumbnailUrl);
      objectUrls.current = objectUrls.current.filter((u) => u !== thumbnailUrl);

      const msg =
        err && typeof err === 'object' && 'message' in err
          ? (err as { message: string }).message
          : 'Unknown error';
      setError(msg);
      setState('error');
    }
  }, []);

  const reset = useCallback(() => {
    setState('idle');
    setProgress(0);
    setResult(null);
    setError(null);
  }, []);

  useEffect(() => {
    const urls = objectUrls.current;
    return () => {
      urls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  return { state, progress, result, error, history, submit, reset };
}
