import { HistoryEntry } from '../types';
import styles from './HistoryList.module.css';

interface HistoryListProps {
  history: HistoryEntry[];
}

const verdictLabel: Record<string, string> = {
  likely_ai: 'AI',
  likely_real: 'Real',
  uncertain: '?',
};

export function HistoryList({ history }: HistoryListProps) {
  if (history.length === 0) return null;

  return (
    <div className={styles.section}>
      <h3 className={styles.heading}>This session</h3>
      <div className={styles.list}>
        {history.map((entry) => (
          <div key={entry.id} className={styles.item}>
            <img src={entry.thumbnailUrl} alt={entry.fileName} className={styles.thumb} />
            <div className={styles.info}>
              <span className={styles.name}>{entry.fileName}</span>
              <span className={styles.score}>{Math.round(entry.score * 100)}% AI</span>
            </div>
            <span className={`${styles.tag} ${styles[entry.verdict]}`}>
              {verdictLabel[entry.verdict]}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
