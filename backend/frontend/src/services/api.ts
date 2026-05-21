import axios, { AxiosProgressEvent } from 'axios';
import { DetectionResponse } from '../types';

const client = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});

export async function detectImage(
  file: File,
  onProgress?: (percent: number) => void
): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await client.post<DetectionResponse>('/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (event: AxiosProgressEvent) => {
      if (event.total && onProgress) {
        onProgress(Math.round((event.loaded * 100) / event.total));
      }
    },
  });

  return data;
}

export async function healthCheck(): Promise<boolean> {
  try {
    await client.get('/health');
    return true;
  } catch {
    return false;
  }
}
