import { DetectionResponse } from '../types';
import styles from './ResultCard.module.css';

interface ResultCardProps {
  result: DetectionResponse;
  onRetry: () => void;
}

function RingChart({ score }: { score: number }) {
  const r = 42;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - score);

  let color = '#eab308'; // uncertain
  if (score > 0.7) color = '#ef4444';
  else if (score < 0.3) color = '#22c55e';

  return (
    <svg width="120" height="120" viewBox="0 0 120 120">
      <circle cx="60" cy="60" r={r} fill="none" stroke="#e2e8f0" strokeWidth="8" />
      <circle
        cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="8"
        strokeLinecap="round" strokeDasharray={circumference}
        strokeDashoffset={offset} transform="rotate(-90 60 60)"
        style={{ transition: 'stroke-dashoffset 0.6s ease' }}
      />
      <text x="60" y="56" textAnchor="middle" fontSize="22" fontWeight="700" fill="#1e293b">
        {Math.round(score * 100)}%
      </text>
      <text x="60" y="76" textAnchor="middle" fontSize="11" fill="#64748b">
        AI probability
      </text>
    </svg>
  );
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const labels: Record<string, { text: string; className: string }> = {
    likely_ai: { text: 'Likely AI-Generated', className: 'ai' },
    likely_real: { text: 'Likely Real', className: 'real' },
    uncertain: { text: 'Uncertain', className: 'uncertain' },
  };
  const { text, className } = labels[verdict] ?? labels.uncertain;

  return <span className={`${styles.badge} ${styles[className]}`}>{text}</span>;
}

export function ResultCard({ result, onRetry }: ResultCardProps) {
  return (
    <div className={styles.card}>
      <RingChart score={result.score} />
      <VerdictBadge verdict={result.verdict} />
      <p className={styles.meta}>inference: {result.inference_time_ms}ms</p>
      <button className={styles.retry} onClick={onRetry}>
        Test another image
      </button>
    </div>
  );
}
