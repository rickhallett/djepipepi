#!/usr/bin/env python3

# /// script
# dependencies = [
#   "openai>=1.3.0",
#   "pydub>=0.25.1",
#   "rich>=13.6.0",
# ]
# ///

"""
Transcription Agent - A complete audio transcription pipeline in a single file.

This agent handles the following steps:
1. Normalizing filenames to a machine-readable format
2. Checking if files have already been transcribed
3. Splitting large audio files into manageable chunks
4. Transcribing audio chunks
5. Assembling complete transcripts
6. Tracking progress in a database

Usage:
    # Transcribe a single file
    python sfa_transcription_agent.py --file path/to/audio/file.mp3

    # Process all audio files in a directory
    python sfa_transcription_agent.py --dir --audio-dir path/to/audio/files

    # Specify custom output
    python sfa_transcription_agent.py --dir --output-dir path/to/output

    # Enable verbose logging
    python sfa_transcription_agent.py --dir --verbose
"""

import os
import sys
import re
import argparse
import json
import time
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Dependencies
# pip install pydub openai rich
from openai import OpenAI
from pydub import AudioSegment
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich import print as rprint

# Initialize rich console for formatted output
console = Console()

# Constants
DEFAULT_CHUNK_SIZE_MB = 5
AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"]

# ============== Database Functions ==============


def init_database(db_path: str):
    """Initialize the SQLite database and create tables if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS audio_transcriptions (
            file_path TEXT PRIMARY KEY,
            filename_normalized BOOLEAN NOT NULL DEFAULT 0,
            normalized_path TEXT,
            audio_split BOOLEAN NOT NULL DEFAULT 0,
            manifest_path TEXT,
            transcribed BOOLEAN NOT NULL DEFAULT 0,
            transcript_path TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )
        conn.commit()

    return conn


def get_transcript_path(conn, file_path: str) -> Optional[str]:
    """Check if a file has already been transcribed."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT transcript_path FROM audio_transcriptions WHERE file_path = ? AND transcribed = 1",
        (file_path,),
    )
    result = cursor.fetchone()
    return result[0] if result else None


def get_normalized_path(conn, file_path: str) -> Optional[str]:
    """Get the normalized path for a file if it has already been normalized."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT normalized_path FROM audio_transcriptions WHERE file_path = ? AND filename_normalized = 1",
        (file_path,),
    )
    result = cursor.fetchone()
    return result[0] if result else None


def get_manifest_path(conn, file_path: str) -> Optional[str]:
    """Get the manifest path for a file if it has already been split."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT manifest_path FROM audio_transcriptions WHERE file_path = ? AND audio_split = 1",
        (file_path,),
    )
    result = cursor.fetchone()
    return result[0] if result else None


def is_transcribed(conn, file_path: str) -> bool:
    """Check if a file has already been transcribed."""
    return get_transcript_path(conn, file_path) is not None


def update_normalized_status(conn, file_path: str, normalized_path: str):
    """Update the database to reflect that a file's filename has been normalized."""
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO audio_transcriptions (file_path, filename_normalized, normalized_path, last_updated) "
        "VALUES (?, 1, ?, CURRENT_TIMESTAMP)",
        (file_path, normalized_path),
    )
    conn.commit()


def update_split_status(conn, file_path: str, manifest_path: str):
    """Update the database to reflect that a file has been split into chunks."""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE audio_transcriptions SET audio_split = 1, manifest_path = ?, last_updated = CURRENT_TIMESTAMP "
        "WHERE file_path = ?",
        (manifest_path, file_path),
    )
    conn.commit()


def update_transcription_status(conn, file_path: str, transcript_path: str):
    """Update the database to reflect that a file has been fully transcribed."""
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE audio_transcriptions SET transcribed = 1, transcript_path = ?, last_updated = CURRENT_TIMESTAMP "
        "WHERE file_path = ?",
        (transcript_path, file_path),
    )
    conn.commit()


# ============== Audio Processing Functions ==============


def normalize_filename(file_path: str) -> str:
    """
    Normalize an audio filename to a machine-readable format:
    - Convert to lowercase
    - Replace spaces with underscores
    - Add timestamp if not present
    """
    # Get directory, filename, and extension
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)

    # Normalize the filename
    # Convert to lowercase
    normalized_name = name.lower()

    # Replace spaces and non-alphanumeric chars with underscores
    normalized_name = re.sub(r"[^a-z0-9]", "_", normalized_name)

    # Remove consecutive underscores
    normalized_name = re.sub(r"_+", "_", normalized_name)

    # Add timestamp if not present
    if not re.search(r"\d{8}_\d{6}", normalized_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        normalized_name = f"{normalized_name}_{timestamp}"

    # Create the normalized path
    normalized_path = os.path.join(directory, f"{normalized_name}{ext}")

    # Only rename if the normalized path is different
    if normalized_path != file_path:
        os.rename(file_path, normalized_path)

    return normalized_path


def split_audio(
    file_path: str, chunks_dir: str, chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB
) -> str:
    """
    Split a large audio file into smaller chunks.

    Args:
        file_path: Path to the audio file
        chunks_dir: Directory to store audio chunks
        chunk_size_mb: Target size for audio chunks in MB

    Returns:
        Path to the manifest file containing chunk information
    """
    os.makedirs(chunks_dir, exist_ok=True)

    # Define chunk size in bytes
    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Calculate chunk duration in milliseconds
    # Approximate bytes per second based on file size and duration
    file_size_bytes = os.path.getsize(file_path)
    bytes_per_ms = file_size_bytes / len(audio)
    chunk_duration_ms = chunk_size_bytes / bytes_per_ms

    # Ensure chunk duration is at least 1 second
    chunk_duration_ms = max(chunk_duration_ms, 1000)

    # Split the audio into chunks
    chunks = []
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # If file is small enough, don't split
    if file_size_bytes <= chunk_size_bytes:
        chunk_path = os.path.join(
            chunks_dir, f"{filename}_chunk_001{os.path.splitext(file_path)[1]}"
        )
        chunks.append(
            {"path": chunk_path, "start_time": 0, "end_time": len(audio), "index": 1}
        )
        audio.export(chunk_path, format=os.path.splitext(file_path)[1][1:])
    else:
        # Split into multiple chunks
        for i, start_time in enumerate(range(0, len(audio), int(chunk_duration_ms))):
            chunk_index = i + 1
            end_time = min(start_time + int(chunk_duration_ms), len(audio))

            chunk = audio[start_time:end_time]
            chunk_path = os.path.join(
                chunks_dir,
                f"{filename}_chunk_{chunk_index:03d}{os.path.splitext(file_path)[1]}",
            )

            chunk.export(chunk_path, format=os.path.splitext(file_path)[1][1:])

            chunks.append(
                {
                    "path": chunk_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "index": chunk_index,
                }
            )

    # Create a manifest file
    manifest = {
        "original_file": file_path,
        "total_duration": len(audio),
        "chunks": chunks,
        "created_at": datetime.now().isoformat(),
    }

    manifest_path = os.path.join(chunks_dir, f"{filename}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def get_chunks_from_manifest(manifest_path: str) -> List[str]:
    """Get the list of chunk paths from a manifest file."""
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    return [chunk["path"] for chunk in manifest["chunks"]]


def cleanup_chunks(chunk_paths: List[str]):
    """Clean up temporary audio chunks after successful transcription."""
    for chunk_path in chunk_paths:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)


# ============== Transcription Functions ==============


def transcribe_audio(
    audio_path: str, api_key: str, transcriptions_dir: str
) -> Dict[str, Any]:
    """
    Transcribe an audio file using the OpenAI Whisper API.

    Args:
        audio_path: Path to the audio file to transcribe
        api_key: OpenAI API key
        transcriptions_dir: Directory to store individual chunk transcriptions

    Returns:
        Dictionary containing the transcription data
    """
    os.makedirs(transcriptions_dir, exist_ok=True)

    # Check if transcription already exists
    transcription_path = os.path.join(
        transcriptions_dir,
        f"{os.path.splitext(os.path.basename(audio_path))[0]}_transcription.json",
    )

    if os.path.exists(transcription_path):
        with open(transcription_path, "r") as f:
            return json.load(f)

    # Set up OpenAI client
    client = OpenAI(api_key=api_key)

    # Transcribe the audio
    with open(audio_path, "rb") as audio_file:
        try:
            # Create file object for the audio.transcriptions.create method
            audio_file_obj = audio_file.read()

            # Use the correct API call method for the current OpenAI Python client
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.mp3", audio_file_obj, "audio/mpeg"),
                response_format="verbose_json",
            )

            # Parse the response (structure might depend on the exact API version)
            if hasattr(response, "model_dump"):
                # For newer OpenAI client versions
                response_data = response.model_dump()
                text = response_data.get("text", "")
                segments = response_data.get("segments", [])
                language = response_data.get("language", "")
                duration = response_data.get("duration", 0)
            else:
                # For older OpenAI client versions
                text = getattr(response, "text", "")
                segments = getattr(response, "segments", [])
                language = getattr(response, "language", "")
                duration = getattr(response, "duration", 0)

            # Create the transcription data structure
            transcription = {
                "audio_path": audio_path,
                "text": text,
                "segments": segments,
                "language": language,
                "duration": duration,
                "timestamp": time.time(),
            }

            # Save the transcription
            with open(transcription_path, "w") as f:
                json.dump(transcription, f, indent=2)

            return transcription

        except Exception as e:
            # If transcription fails, save the error and return a minimal response
            console.print(
                f"[bold red]Error transcribing {audio_path}: {str(e)}[/bold red]"
            )
            error_data = {
                "audio_path": audio_path,
                "error": str(e),
                "text": "",
                "segments": [],
                "language": "unknown",
                "duration": 0,
                "timestamp": time.time(),
            }

            with open(transcription_path, "w") as f:
                json.dump(error_data, f, indent=2)

            return error_data


def assemble_transcript(
    original_file: str,
    chunk_transcriptions: List[Tuple[str, Dict[str, Any]]],
    output_path: str,
) -> str:
    """
    Assemble individual chunk transcriptions into a complete transcript.

    Args:
        original_file: Path to the original audio file
        chunk_transcriptions: List of tuples (chunk_path, transcription_data)
        output_path: Path to write the assembled transcript

    Returns:
        Path to the assembled transcript
    """
    # Sort chunks by index (extracted from filename)
    sorted_chunks = sorted(
        chunk_transcriptions,
        key=lambda x: int(os.path.basename(x[0]).split("_chunk_")[1].split(".")[0]),
    )

    # Initialize the complete transcript
    complete_transcript = {
        "original_file": original_file,
        "text": "",
        "segments": [],
        "created_at": time.time(),
    }

    # Gather all text and adjust segment timings
    time_offset = 0
    for chunk_path, transcription in sorted_chunks:
        # Add text with a space
        if complete_transcript["text"]:
            complete_transcript["text"] += " "
        complete_transcript["text"] += transcription["text"]

        # Adjust segment timings and add to complete transcript
        for segment in transcription.get("segments", []):
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += time_offset
            adjusted_segment["end"] += time_offset
            adjusted_segment["chunk_source"] = os.path.basename(chunk_path)
            complete_transcript["segments"].append(adjusted_segment)

        # Update time offset for the next chunk
        if transcription.get("duration"):
            time_offset += transcription["duration"]

    # Add additional metadata
    complete_transcript["num_chunks"] = len(sorted_chunks)
    complete_transcript["language"] = (
        sorted_chunks[0][1].get("language", "unknown") if sorted_chunks else "unknown"
    )

    # Save the complete transcript
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(complete_transcript, f, indent=2)

    return output_path


# ============== Main Functionality ==============


def process_file(
    file_path: str,
    db_conn,
    output_dir: str = "data/output",
    chunks_dir: str = "data/chunks",
    transcriptions_dir: str = "data/transcriptions",
    api_key: Optional[str] = None,
    verbose: bool = False,
    parent_progress: Optional[Progress] = None,
) -> Optional[str]:
    """
    Process a single audio file through the full transcription pipeline.

    Args:
        file_path: Path to the audio file to transcribe
        db_conn: SQLite database connection
        output_dir: Directory to store the final transcripts
        chunks_dir: Directory to store audio chunks
        transcriptions_dir: Directory to store individual chunk transcriptions
        api_key: OpenAI API key
        verbose: Whether to output verbose logging
        parent_progress: Optional parent progress bar to use instead of creating a new one

    Returns:
        Path to the output transcript file, or None if transcription failed
    """
    if verbose:
        console.print(f"Processing file: [bold cyan]{file_path}[/bold cyan]")

    # Check if file already transcribed
    transcript_path = get_transcript_path(db_conn, file_path)
    if transcript_path and os.path.exists(transcript_path):
        if verbose:
            console.print(
                f"✅ File already transcribed: [green]{transcript_path}[/green]"
            )
        return transcript_path

    # Step 1: Normalize filename if needed
    normalized_path = get_normalized_path(db_conn, file_path)
    if not normalized_path:
        if verbose:
            console.print(f"🔄 Normalizing filename: [yellow]{file_path}[/yellow]")
        normalized_path = normalize_filename(file_path)
        update_normalized_status(db_conn, file_path, normalized_path)
        console.print(f"  ✓ Normalized to: [green]{normalized_path}[/green]")

    # Step 2: Split audio into chunks if needed
    manifest_path = get_manifest_path(db_conn, file_path)
    if not manifest_path:
        if verbose:
            console.print(
                f"✂️  Splitting audio into chunks: [yellow]{normalized_path}[/yellow]"
            )
        manifest_path = split_audio(normalized_path, chunks_dir)
        update_split_status(db_conn, file_path, manifest_path)
        console.print(f"  ✓ Created manifest: [green]{manifest_path}[/green]")

    # Step 3: Transcribe audio chunks
    if not is_transcribed(db_conn, file_path):
        if verbose:
            console.print("[bold magenta]🎙️ Transcribing audio chunks...[/bold magenta]")
        chunks = get_chunks_from_manifest(manifest_path)
        chunk_transcriptions = []

        # If we have a parent progress bar, use that instead of creating a new one
        if parent_progress:
            # Create a subtask in the parent progress bar
            subtask = parent_progress.add_task(
                f"[cyan]Transcribing chunks for {os.path.basename(file_path)}",
                total=len(chunks),
            )

            for chunk_path in chunks:
                chunk_name = os.path.basename(chunk_path)
                parent_progress.update(
                    subtask, description=f"[cyan]Transcribing: {chunk_name}"
                )
                transcript = transcribe_audio(chunk_path, api_key, transcriptions_dir)
                chunk_transcriptions.append((chunk_path, transcript))
                parent_progress.advance(subtask)

        else:
            # Create a new progress bar if no parent was provided
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Transcribing {len(chunks)} chunks", total=len(chunks)
                )

                for chunk_path in chunks:
                    chunk_name = os.path.basename(chunk_path)
                    progress.update(
                        task, description=f"[cyan]Transcribing: {chunk_name}"
                    )
                    transcript = transcribe_audio(
                        chunk_path, api_key, transcriptions_dir
                    )
                    chunk_transcriptions.append((chunk_path, transcript))
                    progress.advance(task)

        # Step 4: Assemble transcription
        output_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(normalized_path))[0] + "_transcript.json",
        )

        if verbose:
            console.print(
                f"📝 Assembling complete transcript: [yellow]{output_path}[/yellow]"
            )
        transcript_path = assemble_transcript(
            normalized_path, chunk_transcriptions, output_path
        )
        console.print(f"  ✓ Transcript assembled: [green]{transcript_path}[/green]")

        # Step 5: Update database with transcription complete
        update_transcription_status(db_conn, file_path, transcript_path)

        # Step 6: Cleanup temporary files
        if verbose:
            console.print(f"🧹 Cleaning up temporary files for [dim]{file_path}[/dim]")
        cleanup_chunks(chunks)
        console.print(f"  ✓ Cleanup complete")

    return get_transcript_path(db_conn, file_path)


def process_directory(
    audio_dir: str,
    db_conn,
    output_dir: str = "data/output",
    chunks_dir: str = "data/chunks",
    transcriptions_dir: str = "data/transcriptions",
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> List[str]:
    """
    Process all audio files in the specified directory.

    Args:
        audio_dir: Directory containing the audio files to transcribe
        db_conn: SQLite database connection
        output_dir: Directory to store the final transcripts
        chunks_dir: Directory to store audio chunks
        transcriptions_dir: Directory to store individual chunk transcriptions
        api_key: OpenAI API key
        verbose: Whether to output verbose logging

    Returns:
        List of paths to the output transcript files
    """
    if verbose:
        console.print(
            f"📂 Processing all audio files in [bold blue]{audio_dir}[/bold blue]"
        )

    audio_files = [
        f
        for f in os.listdir(audio_dir)
        if os.path.isfile(os.path.join(audio_dir, f))
        and any(f.lower().endswith(ext) for ext in AUDIO_EXTENSIONS)
    ]

    if not audio_files:
        console.print("[yellow]⚠️ No audio files found in directory[/yellow]")
        return []

    console.print(f"[green]🔍 Found {len(audio_files)} audio files to process[/green]")

    transcript_paths = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task(f"[cyan]Processing files", total=len(audio_files))

        for filename in audio_files:
            file_path = os.path.join(audio_dir, filename)
            progress.update(main_task, description=f"[cyan]Processing: {filename}")

            transcript_path = process_file(
                file_path,
                db_conn,
                output_dir,
                chunks_dir,
                transcriptions_dir,
                api_key,
                verbose,
                parent_progress=progress,  # Pass the progress bar to process_file
            )

            if transcript_path:
                transcript_paths.append(transcript_path)

            progress.advance(main_task)

    return transcript_paths


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Audio Transcription Pipeline")

    # Create a mutually exclusive group for file vs. directory selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file", type=str, help="Path to a single audio file to transcribe"
    )
    group.add_argument(
        "--dir",
        action="store_true",
        help="Process all audio files in the audio directory",
    )

    # Optional arguments
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="data/audio",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Directory to store transcripts",
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="data/chunks",
        help="Directory to store audio chunks",
    )
    parser.add_argument(
        "--transcriptions-dir",
        type=str,
        default="data/transcriptions",
        help="Directory to store chunk transcriptions",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/db/transcriptions.sqlite",
        help="Path to the SQLite database",
    )
    parser.add_argument("--api-key", type=str, help="OpenAI API key for transcription")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    """Main entry point for the transcription agent."""
    args = parse_args()

    # Extract API key from environment if not provided
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print(
            "[bold red]❌ Error:[/bold red] OpenAI API key is required either via --api-key or OPENAI_API_KEY environment variable"
        )
        return 1

    # Create required directories
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.chunks_dir, exist_ok=True)
    os.makedirs(args.transcriptions_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)

    # Initialize database
    db_conn = init_database(args.db_path)

    try:
        if args.file:
            if not os.path.exists(args.file):
                console.print(
                    f"[bold red]❌ Error:[/bold red] File not found: {args.file}"
                )
                return 1

            console.print(
                Panel.fit(
                    "🎧 [bold blue]Audio Transcription Pipeline - Single File Mode[/bold blue] 🎧"
                )
            )
            transcript_path = process_file(
                args.file,
                db_conn,
                args.output_dir,
                args.chunks_dir,
                args.transcriptions_dir,
                api_key,
                args.verbose,
            )

            if transcript_path:
                console.print(
                    f"\n[bold green]✅ Transcription complete:[/bold green] {transcript_path}"
                )
            else:
                console.print(
                    f"\n[bold red]❌ Failed to transcribe:[/bold red] {args.file}"
                )

        elif args.dir:
            console.print(
                Panel.fit(
                    "🎧 [bold blue]Audio Transcription Pipeline - Directory Mode[/bold blue] 🎧"
                )
            )
            transcript_paths = process_directory(
                args.audio_dir,
                db_conn,
                args.output_dir,
                args.chunks_dir,
                args.transcriptions_dir,
                api_key,
                args.verbose,
            )

            if transcript_paths:
                console.print(
                    f"\n[bold green]✅ Transcribed {len(transcript_paths)} files:[/bold green]"
                )
                for path in transcript_paths:
                    console.print(f"  - {path}")
            else:
                console.print("\n[yellow]⚠️ No files were transcribed[/yellow]")

    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {str(e)}")
        import traceback

        console.print(f"[red]{traceback.format_exc()}[/red]")
        return 1
    finally:
        db_conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
