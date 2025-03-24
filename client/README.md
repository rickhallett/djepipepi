# Audio Transcription UI

A modern web application for transcribing audio files using a FastAPI backend. Built with React, TypeScript, TailwindCSS, and shadcn UI components.

## Features

- 🎤 Upload audio files for transcription
- 📊 Monitor transcription process in real time
- 📝 View transcription results in beautiful markdown format
- 🚀 Modern, responsive UI built with shadcn components
- 🔄 Real-time progress updates via polling

## Getting Started

### Prerequisites

- Bun (or npm/yarn)
- FastAPI backend running (see server configuration)

### Installation

1. Clone the repository
2. Navigate to the client directory
3. Install dependencies:

```bash
bun install
```

4. Create a `.env` file with the backend URL:

```
VITE_API_URL=http://localhost:8000
```

5. Start the development server:

```bash
bun dev
```

### Server Configuration

This UI connects to a FastAPI backend for audio transcription. Ensure the backend is running and configured properly before using the UI.

The default FastAPI url is `http://localhost:8000`, but you can change it in the `.env` file.

## Usage

1. Open the application in your browser
2. Upload an audio file using the drop zone
3. Monitor the transcription progress in real time
4. View the completed transcription when processing is complete

## Technology Stack

- React 19
- TypeScript
- TailwindCSS 4
- shadcn UI Components
- Vite

## Why FastAPI (vs Convex)?

For this application, we chose to interface directly with FastAPI rather than using Convex for the following reasons:

1. **Direct File Handling**: FastAPI handles file uploads efficiently, especially important for audio files that can be large.
2. **Existing Process Monitoring**: The backend already includes a polling mechanism for job status and progress.
3. **Simplified Architecture**: For this use case, introducing Convex would add an unnecessary layer between the UI and the processing backend.
4. **Focused Requirements**: Our app primarily needs file upload, process monitoring, and result display - not complex real-time collaborative features.

Convex would have been advantageous if we needed:
- Real-time collaboration among multiple users
- Complex data relationships and queries
- Automatic state synchronization across clients
- Serverless database functionality
