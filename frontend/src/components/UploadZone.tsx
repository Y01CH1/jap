import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { DetectionState } from '../types';
import styles from './UploadZone.module.css';

interface UploadZoneProps {
  state: DetectionState;
  progress: number;
  onUpload: (file: File) => void;
}

export function UploadZone({ state, progress, onUpload }: UploadZoneProps) {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length === 0) return;
      const file = accepted[0];
      setPreview(URL.createObjectURL(file));
      onUpload(file);
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    maxFiles: 1,
    maxSize: 16 * 1024 * 1024,
    disabled: state === 'uploading' || state === 'detecting',
  });

  const isBusy = state === 'uploading' || state === 'detecting';

  return (
    <div className={styles.wrapper}>
      <div
        {...getRootProps()}
        className={`${styles.dropzone} ${isDragActive ? styles.active : ''} ${isBusy ? styles.busy : ''}`}
      >
        <input {...getInputProps()} />
        {isBusy ? (
          <div className={styles.progress}>
            <div className={styles.spinner} />
            <p>{state === 'uploading' ? `Uploading... ${progress}%` : 'Analyzing...'}</p>
            {state === 'uploading' && (
              <div className={styles.bar}>
                <div className={styles.fill} style={{ width: `${progress}%` }} />
              </div>
            )}
          </div>
        ) : isDragActive ? (
          <p>Drop image here</p>
        ) : (
          <div>
            <p className={styles.cta}>Drag & drop an image, or click to select</p>
            <p className={styles.hint}>JPG, PNG, WebP up to 16MB</p>
          </div>
        )}
      </div>
      {preview && !isBusy && (
        <div className={styles.preview}>
          <img src={preview} alt="Preview" />
        </div>
      )}
    </div>
  );
}
