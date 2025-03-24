export interface TranscriptionConfig {
  chunk_size_mb?: number;
  parallel_mode?: boolean;
  use_multi_provider?: boolean;
  provider_preference?: string[];
}

export interface TranscriptionJob {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  result_path?: string;
  error?: string;
}

export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const transcriptionService = {
  /**
   * Upload an audio file for transcription
   */
  async transcribeAudio(file: File, config?: TranscriptionConfig): Promise<{ job_id: string }> {
    const formData = new FormData();
    formData.append('file', file);

    if (config) {
      formData.append('config', JSON.stringify(config));
    }

    const response = await fetch(`${API_URL}/transcribe/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Transcription failed: ${response.statusText}`);
    }

    return response.json();
  },

  /**
   * Check the status of a transcription job
   */
  async getJobStatus(jobId: string): Promise<TranscriptionJob> {
    const response = await fetch(`${API_URL}/status/${jobId}`);

    if (!response.ok) {
      throw new Error(`Failed to get job status: ${response.statusText}`);
    }

    return response.json();
  },

  /**
   * Fetch the transcription result
   */
  async getTranscriptionResult(resultPath: string): Promise<string> {
    const response = await fetch(`${API_URL}${resultPath}`);

    if (!response.ok) {
      throw new Error(`Failed to get transcription result: ${response.statusText}`);
    }

    return response.text();
  }
}; 