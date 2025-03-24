import React, { useState, useEffect } from 'react';
import { TranscriptionJob, transcriptionService } from '../../api/transcriptionService';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Progress } from '../ui/progress';

interface JobProgressProps {
  jobId: string;
  onJobCompleted: (result: string) => void;
}

export function JobProgress({ jobId, onJobCompleted }: JobProgressProps) {
  const [job, setJob] = useState<TranscriptionJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  useEffect(() => {
    const pollJobStatus = async () => {
      try {
        const jobStatus = await transcriptionService.getJobStatus(jobId);
        setJob(jobStatus);

        if (jobStatus.status === 'completed' && jobStatus.result_path) {
          clearInterval(intervalId);
          const transcriptionResult = await transcriptionService.getTranscriptionResult(jobStatus.result_path);
          setResult(transcriptionResult);
          onJobCompleted(transcriptionResult);
        } else if (jobStatus.status === 'failed') {
          clearInterval(intervalId);
          setError(jobStatus.error || 'Transcription failed');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to get job status');
        clearInterval(intervalId);
      }
    };

    // Poll every 3 seconds
    const intervalId = setInterval(pollJobStatus, 3000);

    // Initial poll
    pollJobStatus();

    return () => clearInterval(intervalId);
  }, [jobId, onJobCompleted]);

  // Format progress for better display
  const formatProgress = (progress?: number) => {
    if (progress === undefined) return '0';
    return progress.toFixed(1);
  };

  // Get status text
  const getStatusText = (status: string) => {
    switch (status) {
      case 'pending':
        return 'Waiting to be processed...';
      case 'processing':
        return `Processing: ${formatProgress(job?.progress)}%`;
      case 'completed':
        return 'Transcription completed!';
      case 'failed':
        return 'Transcription failed';
      default:
        return 'Unknown status';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Transcription Job</CardTitle>
        <CardDescription>
          Job ID: {jobId}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {error ? (
          <div className="text-red-500">{error}</div>
        ) : job ? (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Status:</span>
              <span className="text-sm">{getStatusText(job.status)}</span>
            </div>

            {job.status === 'processing' && job.progress !== undefined && (
              <Progress value={job.progress} />
            )}

            {job.status === 'completed' && result && (
              <div className="mt-4">
                <p className="text-sm text-gray-500 mb-2">Transcription Result:</p>
                <div className="bg-gray-50 p-4 rounded-md text-sm overflow-auto max-h-[300px]">
                  {result}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-6">
            <p className="text-sm text-gray-500">Loading job status...</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
} 