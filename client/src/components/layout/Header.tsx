import React from 'react';

export function Header() {
  return (
    <header className="bg-primary text-primary-foreground py-4 px-6 shadow-md">
      <div className="container mx-auto flex items-center justify-between">
        <h1 className="text-xl font-bold">Audio Transcription</h1>
        <div className="flex items-center space-x-4">
          <a
            href="https://github.com/yourusername/audio-transcription"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-primary-foreground/80"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
              />
            </svg>
          </a>
        </div>
      </div>
    </header>
  );
} 