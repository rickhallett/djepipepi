import React, { useState, useRef } from 'react';
import { TranscriptionConfig, transcriptionService } from '../../api/transcriptionService';
import { Button } from '../../components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../ui/card';

interface FileUploadProps {
  onJobCreated: (jobId: string) => void;
}

export function FileUpload({ onJobCreated }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    try {
      setIsUploading(true);
      setError(null);

      // Default configuration
      const config: TranscriptionConfig = {
        parallel_mode: true
      };

      const result = await transcriptionService.transcribeAudio(file, config);
      onJobCreated(result.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during upload');
    } finally {
      setIsUploading(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upload Audio</CardTitle>
        <CardDescription>
          Upload an audio file for transcription
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div
          className="border-2 border-dashed border-gray-300 rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
            accept="audio/*"
          />

          {file ? (
            <div className="text-center">
              <p className="text-sm font-medium">{file.name}</p>
              <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  clearFile();
                }}
                className="mt-2 text-xs text-red-500 hover:text-red-700"
              >
                Remove
              </button>
            </div>
          ) : (
            <div className="text-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-12 w-12 text-gray-400 mb-3"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <p className="text-sm text-gray-500">Drag and drop an audio file, or click to browse</p>
            </div>
          )}
        </div>

        {error && (
          <div className="mt-3 text-sm text-red-500">
            {error}
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-end">
        <Button
          onClick={handleUpload}
          disabled={!file || isUploading}
        >
          {isUploading ? 'Uploading...' : 'Transcribe Audio'}
        </Button>
      </CardFooter>
    </Card>
  );
} 