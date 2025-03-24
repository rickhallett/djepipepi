# Single File Architecture Tools

A collection of powerful standalone agents built with a single-file architecture approach.

## Tools Overview

- [Blog Management Agent](#blog-management-agent) - Blog creation and management with Ghost CMS integration
- [Transcription Agent](#transcription-agent) - Audio transcription pipeline with OpenAI Whisper

> wait a sec
- [I thought huge files were an anti-pattern?](https://www.oceanheart.blog/best-codebase-architecture-for-ai-coding-and-ai/)

# Blog Management Agent

A powerful single-file blog management system with Ghost CMS integration.

## Features

- **Single File Architecture**: Complete blog management system in a single Python file
- **Claude 3.7 Integration**: Generate AI-written blog posts from transcripts using Anthropic's Claude
- **Ghost CMS Integration**: Publish directly to Ghost blogs via their API
- **Multiple Input Formats**: Process JSON transcripts or markdown files
- **Command Line Interface**: Easy to use commands for all operations
- **Database Tracking**: SQLite database to track published content
- **Automated Categorization**: LLM-powered tagging system that classifies content into one of three categories

## Content Guidelines

- **Author Attribution**: All blog posts are automatically attributed to 'Richard Hallett'
- **Content Categories**: Each post must be tagged with exactly one of the following categories:
  - `science` - For content related to scientific topics, research, or technology
  - `story` - For narrative content, case studies, or personal experiences
  - `spirit` - For content related to philosophy, ethics, or inspirational topics
- **Category Selection**: Categories are automatically assigned by an LLM that analyzes the blog content
- **Publishing Requirements**: Posts must have exactly one of these categories assigned to be published

## Installation

### Requirements

- Python 3.8+
- Pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd single-file-architecture
```

2. Install required dependencies:
```bash
pip install anthropic python-dotenv requests markdown python-frontmatter pyjwt rich
```

3. Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your-anthropic-api-key
GHOST_API_URL=https://your-ghost-blog.com
GHOST_ADMIN_API_KEY=your-admin-api-key-here
GHOST_API_VERSION=v5.0
```

## Usage

### List Existing Blog Posts

```bash
python sfa_blog_agent_v2.py --list
```

### Create a Blog Post from a File

```bash
python sfa_blog_agent_v2.py --file /path/to/transcript.json
```

### Process Multiple Files in a Directory

```bash
python sfa_blog_agent_v2.py --dir /path/to/transcripts/
```

### Publish a Blog Post to Ghost

```bash
python sfa_blog_agent_v2.py --publish <blog-post-id>
```

### Interactive Mode

```bash
python sfa_blog_agent_v2.py
```

Then enter your request, for example:
- "Create a new blog post about AI tools"
- "Search for posts about coding"
- "Update the blog post with ID 12345"

## Ghost CMS Integration

The agent supports publishing to Ghost CMS using their Admin API. It:

1. Authenticates using JWT tokens
2. Converts markdown to HTML
3. Uploads tags, author information, and content
4. Verifies the post has exactly one of the required tags ('science', 'story', or 'spirit')
5. Tracks Ghost post IDs and URLs

To publish to Ghost, you'll need:
- Ghost Admin API key in format `id:secret`
- Your Ghost blog URL
- API version (defaults to v5.0)
- Post must have exactly one of the required category tags

## AI Blog Generation

The agent can generate complete blog posts from transcript data using Claude 3.7:

1. Extracts key information from JSON transcripts
2. Sends a structured prompt to Claude
3. Formats the response as a well-structured blog post
4. Automatically attributes all content to 'Richard Hallett'
5. Uses LLM to categorize content with one required tag: 'science', 'story', or 'spirit'
6. Saves the generated content with appropriate metadata

## Data Structure

Blog posts are stored as JSON files in `data/blogs/` with the following format:

```json
{
  "id": "uuid-string",
  "title": "Blog Post Title",
  "content": "Markdown content...",
  "author": "Richard Hallett",
  "tags": ["science"],
  "published": true,
  "created_at": "ISO timestamp",
  "updated_at": "ISO timestamp",
  "ghost_id": "ghost-post-id",
  "ghost_url": "https://your-blog.com/post-url"
}
```

## Database Tracking

The agent uses SQLite to track which source files have been processed into blog posts. The database is stored at `data/db/blog.sqlite`.

# Transcription Agent

A complete audio transcription pipeline in a single Python file.

## Features

- **Automated Workflow**: Handles the entire transcription process from audio file to JSON transcript
- **File Normalization**: Standardizes filenames to machine-readable format
- **Large File Handling**: Splits audio files into manageable chunks for API limits
- **Progress Tracking**: Uses SQLite database to track transcription status
- **Resume Capability**: Can resume interrupted transcriptions
- **Rich Console Output**: Visual progress indicators and status updates

## Installation

### Requirements

- Python 3.8+
- Pip package manager

### Setup

1. Install required dependencies:
```bash
pip install pydub openai rich
```

2. Set your OpenAI API key:
```
export OPENAI_API_KEY=your-openai-api-key
```

## Usage

### Transcribe a Single File

```bash
python sfa_transcription_agent.py --file path/to/audio/file.mp3
```

### Process All Audio Files in a Directory

```bash
python sfa_transcription_agent.py --dir --audio-dir path/to/audio/files
```

### Specify Custom Output

```bash
python sfa_transcription_agent.py --dir --output-dir path/to/output
```

### Additional Options

```bash
python sfa_transcription_agent.py --verbose  # Enable detailed logging
python sfa_transcription_agent.py --chunks-dir custom/chunks/path  # Custom chunk storage location
```

## Workflow Process

The agent performs these steps in sequence:

1. Normalizes filenames to a consistent format with timestamps
2. Checks if the file has already been transcribed (resume capability)
3. Splits large audio files into smaller chunks (default 5MB)
4. Transcribes each chunk using OpenAI's Whisper API
5. Assembles the complete transcript with adjusted timings
6. Saves the final transcript as a structured JSON file
7. Cleans up temporary chunk files

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- FLAC (.flac)
- M4A (.m4a)
- OGG (.ogg)
- AAC (.aac)
- WMA (.wma)

## Data Structure

Transcripts are stored as JSON files with the following format:

```json
{
  "original_file": "path/to/original.mp3",
  "text": "The complete transcript text...",
  "segments": [
    {
      "start": 0.0,
      "end": 4.5,
      "text": "First segment of speech",
      "chunk_source": "file_chunk_001.mp3"
    },
    ...
  ],
  "num_chunks": 3,
  "language": "en",
  "created_at": "timestamp"
}
```

## License

[Your license information] 