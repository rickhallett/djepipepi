# DJE Pipeline Project

A modular toolkit for building AI-powered content generation and publishing pipelines.

## Project Structure

This project consists of three main components:

### [Client](/client/)
The frontend application for user interaction and content management.

### [Convex](/convex/)
Backend services built with Convex for data storage, processing, and API endpoints.
- Real-time database with automatic synchronization
- Serverless functions for data processing
- Secure authentication and access control
- File storage for media assets

### [Single File Architecture](/single-file-architecture/)
Standalone utility agents that can be run independently:
- [Blog Management Agent](/single-file-architecture/#blog-management-agent) - Create, manage, and publish blog posts to Ghost CMS
- [Transcription Agent](/single-file-architecture/#transcription-agent) - Process audio files into structured transcripts

## Getting Started

Each component has its own setup instructions and documentation. Follow the links above to get started with the specific component you need.

## Development

This project uses:
- Python 3.8+ for standalone agents
- Node.js and modern frontend frameworks for the client
- Convex for backend services and real-time database

### Integration Flow

1. Audio files are processed by the Transcription Agent to create structured transcripts
2. Processed transcripts are used by the Blog Management Agent to generate blog content
3. Content is classified and published to Ghost CMS 
4. The client application provides a UI for managing the entire pipeline
5. Convex provides the real-time database layer connecting these components

## License

MIT License

Copyright (c) 2025 DJE Pipeline Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 