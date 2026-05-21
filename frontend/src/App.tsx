import { useDetection } from './hooks/useDetection';
import { UploadZone } from './components/UploadZone';
import { ResultCard } from './components/ResultCard';
import { HistoryList } from './components/HistoryList';
import styles from './App.module.css';

export default function App() {
  const { state, progress, result, error, history, submit, reset } = useDetection();

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <h1 className={styles.logo}>JAP</h1>
        <p className={styles.subtitle}>Judge AI Pictures — Spot synthetic media, instantly.</p>
      </header>

      <main className={styles.main}>
        {(state === 'idle' || state === 'uploading' || state === 'detecting') && (
          <UploadZone state={state} progress={progress} onUpload={submit} />
        )}

        {state === 'done' && result && (
          <ResultCard result={result} onRetry={reset} />
        )}

        {state === 'error' && (
          <div className={styles.error}>
            <p>{error || 'Something went wrong'}</p>
            <button onClick={reset}>Try again</button>
          </div>
        )}

        <HistoryList history={history} />
      </main>
    </div>
  );
}
