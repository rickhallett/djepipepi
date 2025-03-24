import React, { useState } from 'react';
import { FileUpload } from './FileUpload';
import { JobProgress } from './JobProgress';
import { TranscriptViewer } from './TranscriptViewer';

export function Dashboard() {
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [completedTranscript, setCompletedTranscript] = useState<string | null>(null);

  const handleJobCreated = (jobId: string) => {
    setCurrentJobId(jobId);
    setCompletedTranscript(null);
  };

  const handleJobCompleted = (result: string) => {
    setCompletedTranscript(result);
  };

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <FileUpload onJobCreated={handleJobCreated} />
        </div>
        <div>
          {currentJobId && (
            <JobProgress
              jobId={currentJobId}
              onJobCompleted={handleJobCompleted}
            />
          )}
        </div>
      </div>

      {completedTranscript && (
        <div className="mt-10">
          <TranscriptViewer content={completedTranscript} />
        </div>
      )}
    </div>
  );
} 