#!/usr/bin/env python3

# /// script
# dependencies = [
#   "anthropic>=0.15.0",
#   "python-dotenv>=1.0.0",
#   "requests>=2.31.0",
#   "markdown>=3.5.0",
#   "python-frontmatter>=1.0.0",
#   "pyjwt>=2.8.0",
#   "rich>=13.6.0",
#   "openai>=1.3.0",
#   "pydub>=0.25.1",
# ]
# ///

"""
Single File Architecture Blog Agent

This module contains a complete blog management agent in a single file.
It provides capabilities to create, read, update, delete, and search blog posts.
"""

import os
import sys
import json
import uuid
import glob
import re
import time
import argparse
import sqlite3
import requests
import jwt
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from urllib.parse import urljoin

# Try to import rich for pretty console output, but handle gracefully if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Try to import frontmatter for markdown parsing
try:
    import frontmatter

    FRONTMATTER_AVAILABLE = True
except ImportError:
    FRONTMATTER_AVAILABLE = False

# Try to import Anthropic for Claude integration, but handle gracefully if not available
try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Try to import dotenv for environment variable loading
try:
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import markdown for HTML conversion
try:
    import markdown

    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# Initialize console based on rich availability
if RICH_AVAILABLE:
    console = Console()
else:
    # Simple console class as a fallback
    class SimpleConsole:
        def print(self, *args, **kwargs):
            # Remove rich-specific keywords
            for key in ["style", "panel", "markdown", "border_style", "title"]:
                kwargs.pop(key, None)
            print(*args, **kwargs)

        def rule(self, title=""):
            width = 80
            print(
                f"\n{'-' * (width // 2 - len(title) // 2)} {title} {'-' * (width // 2 - len(title) // 2)}\n"
            )

    console = SimpleConsole()

# Constants
MODEL = "claude-3-7-sonnet-20250219"
DEFAULT_THINKING_TOKENS = 3000
VALID_TAGS = ["science", "story", "spirit"]
DEFAULT_AUTHOR = "Richard Hallett"

# Create a data directory for blog posts
BLOG_POSTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "blogs"
)
os.makedirs(BLOG_POSTS_DIR, exist_ok=True)


# Utility Functions
def log_info(module: str, message: str) -> None:
    """Log an informational message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {timestamp} - {module}: {message}")


def log_error(module: str, message: str) -> None:
    """Log an error message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ERROR] {timestamp} - {module}: {message}")


def log_warning(module: str, message: str) -> None:
    """Log a warning message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[WARNING] {timestamp} - {module}: {message}")


def display_token_usage(input_tokens: int, output_tokens: int) -> None:
    """Display token usage information."""
    if RICH_AVAILABLE:
        console.print(
            f"[dim]Token usage: Input={input_tokens}, Output={output_tokens}[/dim]"
        )
    else:
        print(f"Token usage: Input={input_tokens}, Output={output_tokens}")


# Data Models
@dataclass
class BlogPost:
    """Model representing a blog post."""

    title: str
    content: str
    author: str
    tags: List[str]
    published: bool = False
    id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    ghost_id: Optional[str] = None
    ghost_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the blog post to a dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "author": self.author,
            "tags": self.tags,
            "published": self.published,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ghost_id": self.ghost_id,
            "ghost_url": self.ghost_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlogPost":
        """Create a blog post from a dictionary."""
        return cls(
            id=data.get("id"),
            title=data.get("title", ""),
            content=data.get("content", ""),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            published=data.get("published", False),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            ghost_id=data.get("ghost_id"),
            ghost_url=data.get("ghost_url"),
        )


class BlogOperationResult:
    """
    Model representing the result of a blog operation.
    """

    def __init__(self, success: bool, message: str, data: Any = None):
        """
        Initialize a blog operation result.

        Args:
            success: Whether the operation was successful
            message: A message describing the result
            data: Optional data returned by the operation
        """
        self.success = success
        self.message = message
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {"success": self.success, "message": self.message, "data": self.data}


class ToolUseRequest:
    """
    Model representing a tool use request from Claude.
    """

    def __init__(self, command: str, **kwargs):
        """
        Initialize a tool use request.

        Args:
            command: The command to execute
            **kwargs: Additional arguments for the command
        """
        self.command = command
        self.kwargs = kwargs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolUseRequest":
        """
        Create a tool use request from a dictionary.

        Args:
            data: Dictionary containing the tool use request

        Returns:
            A ToolUseRequest instance
        """
        command = data.get("command")

        # Extract all other keys as kwargs
        kwargs = {k: v for k, v in data.items() if k != "command"}

        return cls(command, **kwargs)


def setup_database():
    """
    Set up the blog database to track published files.
    Uses blog.sqlite in the data/db directory if it exists, otherwise creates it.
    """
    # Define the path to the blog.sqlite database in the data/db directory
    db_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "db")
    db_path = os.path.join(db_dir, "blog.sqlite")

    # Ensure the db directory exists
    os.makedirs(db_dir, exist_ok=True)

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the published_blogs table if it doesn't exist
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS published_blogs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        blog_post_id TEXT NOT NULL,
        source_file TEXT NOT NULL,
        source_file_id INTEGER,
        published_date TEXT NOT NULL,
        UNIQUE(source_file, source_file_id)
    )
    """
    )

    conn.commit()
    return conn, cursor


def get_transcription_id(conn, file_path):
    """
    Get the ID of a file in the transcriptions.sqlite database.

    Args:
        conn: SQLite connection
        file_path: Path to the file

    Returns:
        The ID of the file in the transcriptions database, or None if not found
    """
    try:
        cursor = conn.cursor()
        # Get the filename from the path
        file_name = os.path.basename(file_path)

        # Try to find the file in the transcriptions table
        cursor.execute("SELECT id FROM transcriptions WHERE filename = ?", (file_name,))
        result = cursor.fetchone()

        if result:
            return result[0]
        return None
    except Exception as e:
        log_error("database", f"Error getting transcription ID: {str(e)}")
        return None


def mark_as_published(conn, blog_post_id, source_file, source_file_id=None):
    """
    Mark a file as published in the blog database.

    Args:
        conn: SQLite connection
        blog_post_id: ID of the created blog post
        source_file: Path to the source file
        source_file_id: ID of the file in the transcriptions database
    """
    try:
        cursor = conn.cursor()
        published_date = datetime.now().isoformat()

        # Insert or replace the record
        cursor.execute(
            """
        INSERT OR REPLACE INTO published_blogs 
        (blog_post_id, source_file, source_file_id, published_date) 
        VALUES (?, ?, ?, ?)
        """,
            (blog_post_id, source_file, source_file_id, published_date),
        )

        conn.commit()
        log_info(
            "database", f"Marked {source_file} as published with blog ID {blog_post_id}"
        )
    except Exception as e:
        log_error("database", f"Error marking file as published: {str(e)}")


def process_markdown_file(file_path):
    """
    Process a markdown file to extract blog content.

    Args:
        file_path: Path to the markdown file

    Returns:
        Dictionary with blog post data
    """
    try:
        if not FRONTMATTER_AVAILABLE:
            log_error(
                "file_processor",
                "Frontmatter package is not installed. Please install it with 'pip install python-frontmatter'",
            )
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            # Parse front matter and content
            post = frontmatter.load(f)

            # Extract blog post data
            title = post.get("title", os.path.basename(file_path))
            author = post.get("author", "Unknown")
            tags = post.get("tags", [])
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(",")]

            content = post.content

            return {"title": title, "content": content, "author": author, "tags": tags}
    except Exception as e:
        log_error(
            "file_processor", f"Error processing markdown file {file_path}: {str(e)}"
        )
        return None


def process_json_file(file_path):
    """
    Process a JSON file to extract blog post data.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary with blog post data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract blog post data, with fallbacks

        # Try to get a better title from the filename by converting it to a readable format
        filename = os.path.basename(file_path)
        filename_without_ext = os.path.splitext(filename)[0]

        # Remove timestamp pattern if present (typically at the end like _20250324_151153)
        clean_filename = re.sub(r"_\d{8}_\d{6}", "", filename_without_ext)

        # Replace underscores with spaces and remove transcript suffix if present
        clean_title = clean_filename.replace("_", " ").replace("transcript", "").strip()

        # Convert to title case
        formatted_title = " ".join(word.capitalize() for word in clean_title.split())

        # Use formatted filename as title if available, otherwise use default from data or fallback
        title = (
            formatted_title
            if formatted_title
            else data.get("title", os.path.basename(file_path))
        )

        content = data.get("content", data.get("text", ""))
        author = data.get("author", "AI Blog Generator")

        # Try to extract meaningful tags
        default_tags = ["transcript", "ai-generated"]
        if (
            "segments" in data
            and isinstance(data["segments"], list)
            and len(data["segments"]) > 0
        ):
            default_tags.append("podcast")

        tags = data.get("tags", default_tags)
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",")]

        return {"title": title, "content": content, "author": author, "tags": tags}
    except Exception as e:
        log_error("file_processor", f"Error processing JSON file {file_path}: {str(e)}")
        return None


def create_blog_from_file(file_path, db_conn=None):
    """
    Create a blog post from a file.

    Args:
        file_path: Path to the file
        db_conn: SQLite connection

    Returns:
        The created blog post ID or None if failed
    """
    file_path = os.path.abspath(file_path)
    log_info("file_processor", f"Creating blog post from file: {file_path}")

    # Determine file type
    if file_path.endswith(".md") or file_path.endswith(".markdown"):
        blog_data = process_markdown_file(file_path)
    elif file_path.endswith(".json"):
        blog_data = process_json_file(file_path)
    else:
        log_error("file_processor", f"Unsupported file type: {file_path}")
        return None

    if not blog_data:
        return None

    # Load the raw transcript data for AI-generated content if it's a JSON file
    transcript_data = None
    should_generate_with_ai = False

    if file_path.endswith(".json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                transcript_data = json.load(f)
            should_generate_with_ai = True
            log_info("file_processor", "Using AI to generate blog post from transcript")
        except Exception as e:
            log_error("file_processor", f"Error loading transcript data: {str(e)}")
            # Continue with regular blog creation

    # Create the blog post
    result = create_blog_post(
        title=blog_data["title"],
        content=blog_data["content"],
        author=blog_data["author"],
        tags=blog_data["tags"],
        generate_with_ai=should_generate_with_ai,
        transcript_data=transcript_data,
    )

    if result.success:
        blog_post_id = result.data["id"]

        # Mark as published in the database if a connection is provided
        if db_conn:
            source_file_id = get_transcription_id(db_conn, file_path)
            mark_as_published(db_conn, blog_post_id, file_path, source_file_id)

        log_info(
            "file_processor",
            f"Successfully created blog post with ID {blog_post_id} from file {file_path}",
        )
        return blog_post_id
    else:
        log_error(
            "file_processor",
            f"Failed to create blog post from file {file_path}: {result.message}",
        )
        return None


def process_directory(dir_path, db_conn=None):
    """
    Process all markdown and JSON files in a directory to create blog posts.

    Args:
        dir_path: Path to the directory
        db_conn: SQLite connection

    Returns:
        List of created blog post IDs
    """
    dir_path = os.path.abspath(dir_path)
    log_info("file_processor", f"Processing directory: {dir_path}")

    created_posts = []

    # Get all markdown and JSON files
    md_files = glob.glob(os.path.join(dir_path, "*.md")) + glob.glob(
        os.path.join(dir_path, "*.markdown")
    )
    json_files = glob.glob(os.path.join(dir_path, "*.json"))

    # Process all files
    for file_path in md_files + json_files:
        blog_post_id = create_blog_from_file(file_path, db_conn)
        if blog_post_id:
            created_posts.append(blog_post_id)

    log_info(
        "file_processor",
        f"Created {len(created_posts)} blog posts from directory {dir_path}",
    )
    return created_posts


# Blog CRUD Operations
def classify_content_with_llm(client: Anthropic, content: str, title: str) -> str:
    """
    Use Claude to classify blog content into one of the valid categories.

    Args:
        client: The Anthropic client
        content: The blog post content
        title: The blog post title

    Returns:
        One of the valid tags: 'science', 'story', or 'spirit'
    """
    log_info("content_classifier", f"Classifying content for blog: {title}")

    # Create the prompt for classification
    system_prompt = """You are an expert content classifier. Your task is to categorize blog posts into exactly one of these three categories:
1. 'science' - For content related to scientific topics, research, technology, or factual analysis
2. 'story' - For narrative content, case studies, personal experiences, or storytelling
3. 'spirit' - For content related to philosophy, ethics, inspiration, well-being, or personal growth

You must select ONLY ONE category that best represents the content. Your answer should be just the single category name with no additional text."""

    prompt = f"""Please classify the following blog post into exactly one category (science, story, or spirit):

Title: {title}

Content:
{content[:5000]}  # Only use the first 5000 chars to avoid token limits

Respond with just one word - either 'science', 'story', or 'spirit'. Choose the category that best fits the overall content."""

    try:
        # Call Claude to classify the content
        response = client.messages.create(
            model=MODEL,
            max_tokens=10,  # Keep response short since we just need one word
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the classification
        classification = response.content[0].text.strip().lower()

        # Validate and normalize the response
        if classification in VALID_TAGS:
            log_info("content_classifier", f"Content classified as: {classification}")
            return classification
        else:
            # If the model returns something unexpected, log a warning and default to science
            log_warning(
                "content_classifier",
                f"Unexpected classification: {classification}. Defaulting to 'science'",
            )
            return "science"

    except Exception as e:
        log_error("content_classifier", f"Error classifying content: {str(e)}")
        # Default to 'science' if classification fails
        return "science"


def create_blog_post(
    title: str,
    content: str,
    author: str = DEFAULT_AUTHOR,  # Set default to Richard Hallett
    tags: list = None,
    generate_with_ai: bool = False,
    transcript_data: Dict[str, Any] = None,
) -> BlogOperationResult:
    """
    Create a new blog post.

    Args:
        title: Title of the blog post
        content: Content of the blog post
        author: Author of the blog post (defaults to Richard Hallett)
        tags: Optional list of tags (will be overridden with LLM classification)
        generate_with_ai: Whether to generate content using AI
        transcript_data: Optional transcript data to use for AI generation

    Returns:
        BlogOperationResult with result or error message
    """
    log_info("create_tool", f"Creating blog post: {title}")

    try:
        # Create directory if it doesn't exist
        os.makedirs(BLOG_POSTS_DIR, exist_ok=True)

        # Always set author to Richard Hallett
        author = DEFAULT_AUTHOR

        # Check for empty content
        if not content or content.strip() == "":
            if not generate_with_ai and not transcript_data:
                error_msg = "Blog post content cannot be empty"
                log_error("create_tool", error_msg)
                return BlogOperationResult(success=False, message=error_msg)
            log_warning(
                "create_tool",
                "Empty content provided, will attempt to generate with AI",
            )

        # Generate content with AI if requested
        if generate_with_ai and ANTHROPIC_AVAILABLE:
            try:
                # Check for API key
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    log_error(
                        "create_tool",
                        "ANTHROPIC_API_KEY environment variable is not set.",
                    )
                    return BlogOperationResult(
                        success=False,
                        message="ANTHROPIC_API_KEY environment variable is not set.",
                    )

                # Create Anthropic client
                client = Anthropic(api_key=api_key)

                # Get blog agent system prompt
                system_prompt = BlogAgent.get_system_prompt()

                # Extract key content from transcript for large files
                transcript_summary = ""
                if transcript_data:
                    # Extract transcript text or format appropriately based on structure
                    if isinstance(transcript_data, dict):
                        # Handle transcript with segments/text fields
                        if "segments" in transcript_data:
                            # For segmented transcripts, extract key segments (limit to ~10k tokens)
                            segments = transcript_data.get("segments", [])
                            sampled_segments = segments
                            transcript_summary = "\n\n".join(
                                [
                                    f"Time: {seg.get('start', 'N/A')} - {seg.get('end', 'N/A')}\n"
                                    f"Speaker: {seg.get('speaker', 'Unknown')}\n"
                                    f"Text: {seg.get('text', '')}"
                                    for seg in sampled_segments
                                ]
                            )
                        elif "text" in transcript_data:
                            # If there's a text field, use that directly
                            transcript_summary = transcript_data["text"][
                                :30000
                            ]  # Limit to ~30k chars
                        else:
                            # Use a summarized version of the whole transcript
                            transcript_summary = json.dumps(transcript_data)[:10000]
                    else:
                        # If not a dict, convert to string and limit size
                        transcript_summary = str(transcript_data)[:10000]
                else:
                    # Use the provided content
                    transcript_summary = (
                        content[:30000]
                        if content
                        else "Please generate a blog post based on the title."
                    )

                # Create specialized user prompt for blog writing
                user_prompt = f"""
                I need to write a detailed, informative, and engaging blog post based on the following transcript data.
                
                Title: {title}
                
                Here's a summary of the transcript data to use as the source material:
                
                {transcript_summary}
                
                Please write a well-structured blog post in markdown format that:
                1. Has a compelling introduction that sets the context
                2. Organizes the information into clear sections with headings
                3. Includes meaningful insights and takeaways
                4. Has a strong conclusion
                5. Is in a professional but engaging tone
                6. Uses markdown formatting for headings, emphasis, lists, etc.
                
                The blog should be detailed and thorough, covering the key points from the transcript.
                If the transcript appears to be from a video or podcast, please structure the blog 
                to capture the main ideas discussed rather than a direct transcription.

                Do not include any other text than the blog post content.
                """

                # Call Anthropic API to generate the blog post
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )

                # Extract generated content from response
                generated_content = response.content[0].text

                # Track token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                display_token_usage(input_tokens, output_tokens)

                # Use the AI-generated content instead
                content = generated_content
                log_info("create_tool", "Successfully generated blog content with AI")

                # Validate the generated content
                if not content or content.strip() == "":
                    error_msg = "AI failed to generate valid content"
                    log_error("create_tool", error_msg)
                    return BlogOperationResult(success=False, message=error_msg)

                # Log content length for debugging
                log_info(
                    "create_tool",
                    f"Generated content length: {len(content)} characters",
                )

            except Exception as e:
                log_error(
                    "create_tool", f"Error generating blog content with AI: {str(e)}"
                )
                # Continue with original content if AI generation fails
                log_info("create_tool", "Falling back to original content")

                # Check if we have valid content to fall back to
                if not content or content.strip() == "":
                    error_msg = "Failed to generate content with AI and no valid original content provided"
                    log_error("create_tool", error_msg)
                    return BlogOperationResult(success=False, message=error_msg)

        # Final check for content
        if not content or content.strip() == "":
            error_msg = "Blog post content cannot be empty"
            log_error("create_tool", error_msg)
            return BlogOperationResult(success=False, message=error_msg)

        # Generate a unique ID and timestamps
        post_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        # Check for API key for classification
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        chosen_tag = None

        if api_key and ANTHROPIC_AVAILABLE:
            try:
                # Create Anthropic client
                client = Anthropic(api_key=api_key)

                # Classify the content
                chosen_tag = classify_content_with_llm(client, content, title)
                log_info("create_tool", f"Content classified as: {chosen_tag}")
            except Exception as e:
                log_error("create_tool", f"Error classifying content: {str(e)}")
                # Default tag if classification fails
                chosen_tag = "science"
                log_info("create_tool", f"Defaulting to tag: {chosen_tag}")
        else:
            # Default tag if Anthropic is not available
            chosen_tag = "science"
            log_info(
                "create_tool",
                f"Anthropic not available. Defaulting to tag: {chosen_tag}",
            )

        # Set tags to just the single classification
        tags = [chosen_tag]

        # Create the blog post
        blog_post = BlogPost(
            id=post_id,
            title=title,
            content=content,
            author=author,
            tags=tags,
            published=False,
            created_at=current_time,
            updated_at=current_time,
            ghost_id=None,
            ghost_url=None,
        )

        # Log the content length before saving
        log_info(
            "create_tool",
            f"Final content length before saving: {len(blog_post.content)} characters",
        )

        # Save the blog post to a JSON file
        file_path = os.path.join(BLOG_POSTS_DIR, f"{post_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(blog_post.to_dict(), f, indent=2)

        log_info("create_tool", f"Created blog post: {title} with ID: {post_id}")
        return BlogOperationResult(
            success=True,
            message=f"Successfully created blog post: {title}",
            data=blog_post.to_dict(),
        )
    except Exception as e:
        error_msg = f"Failed to create blog post: {str(e)}"
        log_error("create_tool", error_msg)
        return BlogOperationResult(success=False, message=error_msg)


def read_blog_post(post_id: str) -> BlogOperationResult:
    """
    Read a blog post by ID.

    Args:
        post_id: The ID of the blog post to read

    Returns:
        BlogOperationResult with the blog post or error message
    """
    log_info("read_tool", f"Reading blog post with ID: {post_id}")

    try:
        # Read the blog post from the JSON file
        file_path = os.path.join(BLOG_POSTS_DIR, f"{post_id}.json")

        if not os.path.exists(file_path):
            error_msg = f"Blog post with ID {post_id} not found"
            log_error("read_tool", error_msg)
            return BlogOperationResult(success=False, message=error_msg)

        with open(file_path, "r", encoding="utf-8") as f:
            blog_post_data = json.load(f)

        # Validate that the blog post has content
        content = blog_post_data.get("content", "")
        if not content or content.strip() == "":
            log_warning("read_tool", f"Blog post with ID {post_id} has empty content")

        # Create a BlogPost object from the data
        blog_post = BlogPost.from_dict(blog_post_data)

        log_info("read_tool", f"Successfully read blog post: {blog_post.title}")
        return BlogOperationResult(
            success=True,
            message=f"Successfully read blog post: {blog_post.title}",
            data=blog_post.to_dict(),
        )
    except Exception as e:
        error_msg = f"Failed to read blog post: {str(e)}"
        log_error("read_tool", error_msg)
        return BlogOperationResult(success=False, message=error_msg)


def list_blog_posts(
    tag: Optional[str] = None,
    author: Optional[str] = None,
    published_only: bool = False,
) -> BlogOperationResult:
    """
    List all blog posts, optionally filtered by tag, author, or publication status.

    Args:
        tag: Optional tag to filter by
        author: Optional author to filter by
        published_only: Whether to only return published posts

    Returns:
        BlogOperationResult with a list of blog posts or error message
    """
    log_info("read_tool", "Listing blog posts")

    try:
        # Create directory if it doesn't exist
        os.makedirs(BLOG_POSTS_DIR, exist_ok=True)

        # Get all JSON files in the blog posts directory
        file_paths = glob.glob(os.path.join(BLOG_POSTS_DIR, "*.json"))

        blog_posts = []

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    blog_post_data = json.load(f)

                # Apply filters
                if published_only and not blog_post_data.get("published", False):
                    continue

                if author and blog_post_data.get("author") != author:
                    continue

                if tag and tag not in blog_post_data.get("tags", []):
                    continue

                blog_posts.append(blog_post_data)
            except Exception as e:
                log_error("read_tool", f"Error reading file {file_path}: {str(e)}")
                continue

        log_info("read_tool", f"Listed {len(blog_posts)} blog posts")
        return BlogOperationResult(
            success=True,
            message=f"Successfully listed {len(blog_posts)} blog posts",
            data=blog_posts,
        )
    except Exception as e:
        error_msg = f"Failed to list blog posts: {str(e)}"
        log_error("read_tool", error_msg)
        return BlogOperationResult(success=False, message=error_msg)


def update_blog_post(
    post_id: str,
    title: Optional[str] = None,
    content: Optional[str] = None,
    tags: Optional[List[str]] = None,
    published: Optional[bool] = None,
    ghost_id: Optional[str] = None,
    ghost_url: Optional[str] = None,
) -> BlogOperationResult:
    """
    Update a blog post by ID.

    Args:
        post_id: The ID of the blog post to update
        title: Optional new title
        content: Optional new content
        tags: Optional new tags
        published: Optional new publication status
        ghost_id: Optional Ghost post ID
        ghost_url: Optional Ghost post URL

    Returns:
        BlogOperationResult with the updated blog post or error message
    """
    log_info("update_tool", f"Updating blog post with ID: {post_id}")

    try:
        # Read the existing blog post
        read_result = read_blog_post(post_id)

        if not read_result.success:
            return read_result

        # Get the existing blog post data
        blog_post_data = read_result.data

        # Update the fields
        if title is not None:
            blog_post_data["title"] = title

        if content is not None:
            blog_post_data["content"] = content

        # Validate tags if provided
        if tags is not None:
            if len(tags) != 1 or tags[0] not in VALID_TAGS:
                error_msg = f"Blog post must have exactly one tag from {VALID_TAGS}. Provided tags: {tags}"
                log_error("update_tool", error_msg)
                return BlogOperationResult(success=False, message=error_msg)
            blog_post_data["tags"] = tags

        # Re-classify content if content or title changed and no tags provided
        if (title is not None or content is not None) and tags is None:
            updated_title = title if title is not None else blog_post_data["title"]
            updated_content = (
                content if content is not None else blog_post_data["content"]
            )

            # Check for API key for classification
            api_key = os.environ.get("ANTHROPIC_API_KEY")

            if api_key and ANTHROPIC_AVAILABLE:
                try:
                    # Create Anthropic client
                    client = Anthropic(api_key=api_key)

                    # Classify the content
                    chosen_tag = classify_content_with_llm(
                        client, updated_content, updated_title
                    )
                    log_info("update_tool", f"Content reclassified as: {chosen_tag}")
                    blog_post_data["tags"] = [chosen_tag]
                except Exception as e:
                    log_error("update_tool", f"Error reclassifying content: {str(e)}")
                    # Keep existing tag if reclassification fails

        # Ensure author is always Richard Hallett
        blog_post_data["author"] = DEFAULT_AUTHOR

        # Update publication status if provided
        if published is not None:
            blog_post_data["published"] = published

        # Update Ghost information if provided
        if ghost_id is not None:
            blog_post_data["ghost_id"] = ghost_id

        if ghost_url is not None:
            blog_post_data["ghost_url"] = ghost_url

        # Update the updated_at timestamp
        blog_post_data["updated_at"] = datetime.now().isoformat()

        # Save the updated blog post
        file_path = os.path.join(BLOG_POSTS_DIR, f"{post_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(blog_post_data, f, indent=2)

        log_info("update_tool", f"Updated blog post: {blog_post_data['title']}")
        return BlogOperationResult(
            success=True,
            message=f"Successfully updated blog post: {blog_post_data['title']}",
            data=blog_post_data,
        )
    except Exception as e:
        error_msg = f"Failed to update blog post: {str(e)}"
        log_error("update_tool", error_msg)
        return BlogOperationResult(success=False, message=error_msg)


def delete_blog_post(post_id: str) -> BlogOperationResult:
    """
    Delete a blog post by ID.

    Args:
        post_id: The ID of the blog post to delete

    Returns:
        BlogOperationResult with result or error message
    """
    log_info("delete_tool", f"Deleting blog post with ID: {post_id}")

    try:
        # Verify the blog post exists
        read_result = read_blog_post(post_id)

        if not read_result.success:
            return read_result

        # Get the blog post title for the response message
        blog_post_title = read_result.data["title"]

        # Delete the blog post file
        file_path = os.path.join(BLOG_POSTS_DIR, f"{post_id}.json")
        os.remove(file_path)

        log_info("delete_tool", f"Deleted blog post: {blog_post_title}")
        return BlogOperationResult(
            success=True, message=f"Successfully deleted blog post: {blog_post_title}"
        )
    except Exception as e:
        error_msg = f"Failed to delete blog post: {str(e)}"
        log_error("delete_tool", error_msg)
        return BlogOperationResult(success=False, message=error_msg)


def publish_blog_post(post_id: str) -> BlogOperationResult:
    """
    Publish a blog post by ID and to Ghost platform if configured.

    Args:
        post_id: The ID of the blog post to publish

    Returns:
        BlogOperationResult with the published blog post or error message
    """
    log_info("update_tool", f"Publishing blog post with ID: {post_id}")

    # First, read the blog post
    read_result = read_blog_post(post_id)
    if not read_result.success:
        return read_result

    blog_data = read_result.data

    # Validate that the blog post has content before attempting to publish
    content = blog_data.get("content", "")
    if not content or content.strip() == "":
        error_msg = f"Cannot publish blog post with ID {post_id}: Content is empty"
        log_error("update_tool", error_msg)
        return BlogOperationResult(success=False, message=error_msg)

    # If the blog post is already published, check if it's on Ghost
    if blog_data.get("published") and blog_data.get("ghost_id"):
        return BlogOperationResult(
            success=True,
            message=f"Blog post already published on Ghost: {blog_data.get('ghost_url')}",
            data=blog_data,
        )

    # Create BlogPost object from data
    blog_post = BlogPost.from_dict(blog_data)

    # Log content length for debugging
    log_info(
        "update_tool",
        f"Blog post content length before publishing: {len(blog_post.content)} characters",
    )

    # Try to publish to Ghost
    ghost_success, ghost_message, ghost_id = publish_to_ghost(blog_post)

    if ghost_success:
        log_info(
            "ghost_api", f"Published blog post {post_id} to Ghost with ID {ghost_id}"
        )

        # Extract Ghost URL from message
        ghost_url = ghost_message.split(": ")[-1] if ": " in ghost_message else None

        # Update the blog post locally with Ghost info and published status
        result = update_blog_post(
            post_id=post_id, published=True, ghost_id=ghost_id, ghost_url=ghost_url
        )

        if result.success:
            return BlogOperationResult(
                success=True,
                message=f"Blog post published successfully to Ghost: {ghost_url}",
                data=result.data,
            )
        else:
            # If local update failed but Ghost publish succeeded
            return BlogOperationResult(
                success=True,
                message=f"Blog post published to Ghost ({ghost_url}) but failed to update local data: {result.message}",
                data=blog_data,
            )
    else:
        # If Ghost publishing failed, try to just mark as published locally
        log_error("ghost_api", f"Failed to publish to Ghost: {ghost_message}")

        if not DOTENV_AVAILABLE:
            log_error(
                "ghost_api",
                "dotenv package not installed. Install with 'pip install python-dotenv'",
            )

        # Check if Ghost credentials are configured
        api_url, api_key, admin_api_key, api_version = get_ghost_credentials()
        if not api_url or not api_key or not admin_api_key:
            # Just mark as published locally if Ghost isn't configured
            local_result = update_blog_post(post_id, published=True)
            if local_result.success:
                return BlogOperationResult(
                    success=True,
                    message="Blog post marked as published locally. Ghost API not configured.",
                    data=local_result.data,
                )
            else:
                return local_result
        else:
            # Ghost was configured but publishing failed
            return BlogOperationResult(
                success=False,
                message=f"Failed to publish to Ghost: {ghost_message}",
                data=blog_data,
            )


def unpublish_blog_post(post_id: str) -> BlogOperationResult:
    """
    Unpublish a blog post by ID.

    Args:
        post_id: The ID of the blog post to unpublish

    Returns:
        BlogOperationResult with the unpublished blog post or error message
    """
    log_info("update_tool", f"Unpublishing blog post with ID: {post_id}")
    return update_blog_post(post_id, published=False)


def search_blog_posts(
    query: str,
    search_content: bool = True,
    tag: Optional[str] = None,
    author: Optional[str] = None,
) -> BlogOperationResult:
    """
    Search blog posts by query string, optionally filtered by tag or author.

    Args:
        query: The search query
        search_content: Whether to search in the content (otherwise just title and tags)
        tag: Optional tag to filter by
        author: Optional author to filter by

    Returns:
        BlogOperationResult with a list of matching blog posts or error message
    """
    log_info("search_tool", f"Searching blog posts for: {query}")

    try:
        # Create directory if it doesn't exist
        os.makedirs(BLOG_POSTS_DIR, exist_ok=True)

        # Get all JSON files in the blog posts directory
        file_paths = glob.glob(os.path.join(BLOG_POSTS_DIR, "*.json"))

        # Compile the search regex for case-insensitive search
        search_regex = re.compile(query, re.IGNORECASE)

        matching_posts = []

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    blog_post_data = json.load(f)

                # Apply filters
                if author and blog_post_data.get("author") != author:
                    continue

                if tag and tag not in blog_post_data.get("tags", []):
                    continue

                # Check for match in title
                if search_regex.search(blog_post_data.get("title", "")):
                    matching_posts.append(blog_post_data)
                    continue

                # Check for match in tags
                if any(search_regex.search(t) for t in blog_post_data.get("tags", [])):
                    matching_posts.append(blog_post_data)
                    continue

                # Check for match in content if requested
                if search_content and search_regex.search(
                    blog_post_data.get("content", "")
                ):
                    matching_posts.append(blog_post_data)
                    continue

            except Exception as e:
                log_error("search_tool", f"Error processing file {file_path}: {str(e)}")
                continue

        log_info("search_tool", f"Found {len(matching_posts)} matching blog posts")
        return BlogOperationResult(
            success=True,
            message=f"Found {len(matching_posts)} matching blog posts",
            data=matching_posts,
        )
    except Exception as e:
        error_msg = f"Failed to search blog posts: {str(e)}"
        log_error("search_tool", error_msg)
        return BlogOperationResult(success=False, message=error_msg)


# Tool Handler
def handle_tool_use(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle tool use requests from the Claude agent.

    Args:
        input_data: The tool use request data from Claude

    Returns:
        Dictionary with the result or error message
    """
    log_info("tool_handler", f"Received tool use request: {input_data}")

    try:
        # Parse the tool use request
        request = ToolUseRequest.from_dict(input_data)

        # Handle the command
        if request.command == "create_post":
            title = request.kwargs.get("title", "")
            content = request.kwargs.get("content", "")
            author = request.kwargs.get("author", "")
            tags = request.kwargs.get("tags", [])

            result = create_blog_post(title, content, author, tags)

        elif request.command == "get_post":
            post_id = request.kwargs.get("post_id", "")

            result = read_blog_post(post_id)

        elif request.command == "update_post":
            post_id = request.kwargs.get("post_id", "")
            title = request.kwargs.get("title")
            content = request.kwargs.get("content")
            tags = request.kwargs.get("tags")
            published = request.kwargs.get("published")

            result = update_blog_post(post_id, title, content, tags, published)

        elif request.command == "delete_post":
            post_id = request.kwargs.get("post_id", "")

            result = delete_blog_post(post_id)

        elif request.command == "list_posts":
            tag = request.kwargs.get("tag")
            author = request.kwargs.get("author")
            published_only = request.kwargs.get("published_only", False)

            result = list_blog_posts(tag, author, published_only)

        elif request.command == "search_posts":
            query = request.kwargs.get("query", "")
            search_content = request.kwargs.get("search_content", True)
            tag = request.kwargs.get("tag")
            author = request.kwargs.get("author")

            result = search_blog_posts(query, search_content, tag, author)

        elif request.command == "publish_post":
            post_id = request.kwargs.get("post_id", "")

            result = publish_blog_post(post_id)

        elif request.command == "unpublish_post":
            post_id = request.kwargs.get("post_id", "")

            result = unpublish_blog_post(post_id)

        else:
            log_error("tool_handler", f"Unknown command: {request.command}")
            return {"error": f"Unknown command: {request.command}"}

        # Return the result
        if result.success:
            # Convert complex objects to JSON serializable format
            if isinstance(result.data, dict) or isinstance(result.data, list):
                # Convert to JSON string and back to ensure serializability
                clean_data = json.loads(json.dumps(result.data))
                return {"result": result.message, "data": clean_data}
            else:
                return {"result": result.message}
        else:
            return {"error": result.message}

    except Exception as e:
        error_msg = f"Error handling tool use: {str(e)}"
        log_error("tool_handler", error_msg)
        return {"error": error_msg}


# Blog Agent Class
class BlogAgent:
    """
    Agent for blog management using Claude.
    """

    @staticmethod
    def get_system_prompt() -> str:
        """
        Returns the system prompt used for the blog agent.

        Returns:
            The system prompt as a string
        """
        return """You are a helpful AI assistant with blog management capabilities.
You have access to tools that can create, read, update, delete, and search blog posts.
Always think step by step about what you need to do before taking any action.
Be helpful in suggesting blog post ideas and improvements when asked.

When writing blog posts:
- Use a professional yet engaging tone
- Structure content with clear headings and sections
- Include meaningful insights and analysis
- Make the content detailed and thorough
- Use proper markdown formatting
- Ensure the content flows logically
- Add examples and evidence to support points
- Write compelling introductions and strong conclusions
"""

    @staticmethod
    def run_agent(
        client: Anthropic,
        prompt: str,
        max_thinking_tokens: int = DEFAULT_THINKING_TOKENS,
        max_loops: int = 10,
        use_token_efficiency: bool = False,
    ) -> Tuple[str, int, int]:
        """
        Run the Claude agent with blog management capabilities.

        Args:
            client: The Anthropic client
            prompt: The user's prompt
            max_thinking_tokens: Maximum tokens for thinking
            max_loops: Maximum number of tool use loops
            use_token_efficiency: Whether to use token-efficient tool use beta feature

        Returns:
            Tuple containing:
            - Final response from Claude (str)
            - Total input tokens used (int)
            - Total output tokens used (int)
        """
        if not ANTHROPIC_AVAILABLE:
            error_msg = "The Anthropic package is not installed. Please install it with 'pip install anthropic'."
            log_error("blog_agent", error_msg)
            return error_msg, 0, 0

        # Track token usage
        input_tokens_total = 0
        output_tokens_total = 0
        system_prompt = BlogAgent.get_system_prompt()

        # Define blog management tool
        blog_management_tool = {
            "name": "blog_management",
            "description": "Manage blog posts including creation, editing, searching, and publishing",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "create_post",
                            "get_post",
                            "update_post",
                            "delete_post",
                            "list_posts",
                            "search_posts",
                            "publish_post",
                            "unpublish_post",
                        ],
                        "description": "The blog management command to execute",
                    }
                },
                "required": ["command"],
            },
        }

        messages = [
            {
                "role": "user",
                "content": f"""I need help managing my blog. Here's what I want to do:

{prompt}

Please use the blog management tools to help me with this. First, think through what you need to do, then use the appropriate tools.
""",
            }
        ]

        loop_count = 0
        tool_use_count = 0
        thinking_start_time = time.time()

        while loop_count < max_loops:
            loop_count += 1

            console.rule(f"[yellow]Agent Loop {loop_count}/{max_loops}[/yellow]")
            log_info("blog_agent", f"Starting agent loop {loop_count}/{max_loops}")

            # Create message with blog management tool
            message_args = {
                "model": MODEL,
                "max_tokens": 4096,
                "tools": [blog_management_tool],
                "messages": messages,
                "system": system_prompt,
                "thinking": {"type": "enabled", "budget_tokens": max_thinking_tokens},
            }

            # Use the beta.messages with betas parameter if token efficiency is enabled
            if use_token_efficiency:
                # Using token-efficient tools beta feature
                message_args["betas"] = ["token-efficient-tools-2025-02-19"]
                response = client.beta.messages.create(**message_args)
            else:
                # Standard approach
                response = client.messages.create(**message_args)

            # Track token usage
            if hasattr(response, "usage"):
                input_tokens = getattr(response.usage, "input_tokens", 0)
                output_tokens = getattr(response.usage, "output_tokens", 0)

                input_tokens_total += input_tokens
                output_tokens_total += output_tokens

                console.print(
                    f"[dim]Loop {loop_count} tokens: Input={input_tokens}, Output={output_tokens}[/dim]"
                )
                log_info(
                    "blog_agent",
                    f"Loop {loop_count} tokens: Input={input_tokens}, Output={output_tokens}",
                )

            # Process response content
            thinking_block = None
            tool_use_block = None
            text_block = None

            for content_block in response.content:
                if content_block.type == "thinking":
                    thinking_block = content_block
                    # Access the thinking attribute which contains the actual thinking text
                    if hasattr(thinking_block, "thinking"):
                        console.print(
                            Panel(
                                thinking_block.thinking,
                                title=f"Claude's Thinking (Loop {loop_count})",
                                border_style="blue",
                            )
                        )
                    else:
                        console.print(
                            Panel(
                                "Claude is thinking...",
                                title=f"Claude's Thinking (Loop {loop_count})",
                                border_style="blue",
                            )
                        )
                elif content_block.type == "tool_use":
                    tool_use_block = content_block
                    tool_use_count += 1
                elif content_block.type == "text":
                    text_block = content_block

            # If we got a final text response with no tool use, we're done
            if text_block and not tool_use_block:
                thinking_end_time = time.time()
                thinking_duration = thinking_end_time - thinking_start_time

                console.print(
                    f"\n[bold green]Completed in {thinking_duration:.2f} seconds after {loop_count} loops and {tool_use_count} tool uses[/bold green]"
                )
                log_info(
                    "blog_agent",
                    f"Completed in {thinking_duration:.2f} seconds after {loop_count} loops and {tool_use_count} tool uses",
                )

                # Add the response to messages
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            *([thinking_block] if thinking_block else []),
                            {"type": "text", "text": text_block.text},
                        ],
                    }
                )

                return text_block.text, input_tokens_total, output_tokens_total

            # Handle tool use
            if tool_use_block:
                # Add the assistant's response to messages before handling tool calls
                messages.append({"role": "assistant", "content": response.content})

                console.print(
                    f"\n[bold blue]Tool Call:[/bold blue] {tool_use_block.name}"
                )
                log_info("blog_agent", f"Tool Call: {tool_use_block.name}")

                # Handle the tool use
                tool_result = handle_tool_use(tool_use_block.input)

                # Format tool result for Claude
                tool_result_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_block.id,
                            "content": tool_result.get("error")
                            or tool_result.get("result", ""),
                        }
                    ],
                }

                # If we have data in the result, include it as formatted markdown
                if "data" in tool_result and tool_result["data"]:
                    data_json = json.dumps(tool_result["data"], indent=2)
                    tool_result_message["content"][0][
                        "content"
                    ] += f"\n\n```json\n{data_json}\n```"

                messages.append(tool_result_message)

        # If we reach here, we hit the max loops
        console.print(
            f"\n[bold red]Warning: Reached maximum loops ({max_loops}) without completing the task[/bold red]"
        )
        log_error(
            "blog_agent",
            f"Reached maximum loops ({max_loops}) without completing the task",
        )

        return (
            f"I'm sorry, but I was unable to complete your request within the maximum number of {max_loops} interactions. "
            "Please try simplifying your request or breaking it down into smaller tasks.",
            input_tokens_total,
            output_tokens_total,
        )


# Function to run the agent
def run_agent(
    client: Anthropic,
    prompt: str,
    max_tool_use_loops: int = 15,
    token_efficient_tool_use: bool = True,
) -> Tuple[int, int]:
    """
    Run the blog agent with the given prompt.

    Args:
        client: The Anthropic client
        prompt: The user's prompt
        max_tool_use_loops: Maximum number of tool use loops
        token_efficient_tool_use: Whether to use token-efficient tool use beta feature

    Returns:
        A tuple containing the total input and output tokens used
    """
    agent = BlogAgent()

    # Run the agent with the maximum allowed thinking tokens
    final_response, input_tokens, output_tokens = agent.run_agent(
        client=client,
        prompt=prompt,
        max_thinking_tokens=3000,
        max_loops=max_tool_use_loops,
        use_token_efficiency=token_efficient_tool_use,
    )

    # Print the response in a markdown format if rich is available
    if RICH_AVAILABLE:
        console.print("\n[bold]Final Response:[/bold]")
        console.print(Markdown(final_response))
    else:
        print("\nFinal Response:")
        print(final_response)

    # Display token usage
    display_token_usage(input_tokens, output_tokens)

    return input_tokens, output_tokens


# Ghost blog integration functions
def get_ghost_credentials() -> (
    Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
):
    """
    Get Ghost API credentials from environment variables.

    Returns:
        Tuple containing Ghost API URL, API key, Admin API key, and API version
    """
    api_url = os.environ.get("GHOST_API_URL")
    api_key = os.environ.get("GHOST_API_KEY")
    admin_api_key = os.environ.get("GHOST_ADMIN_API_KEY")
    api_version = os.environ.get("GHOST_API_VERSION", "v5.0")

    if not api_url or not admin_api_key:
        log_error(
            "ghost_api", "Ghost API credentials not found in environment variables"
        )
        return None, None, None, None

    return api_url, api_key, admin_api_key, api_version


def publish_to_ghost(blog_post: BlogPost) -> Tuple[bool, str, Optional[str]]:
    """
    Publish a blog post to Ghost platform.

    Args:
        blog_post: The BlogPost object to publish

    Returns:
        Tuple containing:
        - Success status (bool)
        - Message (str)
        - Ghost post ID if successful (Optional[str])
    """
    # Get Ghost credentials
    api_url, api_key, admin_api_key, api_version = get_ghost_credentials()

    if not api_url or not admin_api_key:
        return False, "Ghost API credentials not configured", None

    # Validate tags before publishing
    if len(blog_post.tags) != 1 or blog_post.tags[0] not in VALID_TAGS:
        error_msg = f"Blog post must have exactly one tag from {VALID_TAGS}. Current tags: {blog_post.tags}"
        log_error("ghost_api", error_msg)
        return False, error_msg, None

    # Validate author
    if blog_post.author != DEFAULT_AUTHOR:
        error_msg = f"Blog post author must be '{DEFAULT_AUTHOR}'. Current author: {blog_post.author}"
        log_error("ghost_api", error_msg)
        return False, error_msg, None

    # Debug logging for content
    content_length = len(blog_post.content) if blog_post.content else 0
    content_snippet = (
        blog_post.content[:150] + "..." if content_length > 0 else "EMPTY CONTENT"
    )
    log_info("ghost_api", f"Blog content length: {content_length} characters")
    log_info("ghost_api", f"Content snippet: {content_snippet}")

    if content_length == 0:
        log_error(
            "ghost_api", "Blog post content is empty! Cannot publish empty content."
        )
        return False, "Blog post content is empty", None

    # Generate JWT token for authentication
    token = generate_ghost_jwt_token(admin_api_key)
    if not token:
        return False, "Failed to generate Ghost authentication token", None

    # Format tags for Ghost
    ghost_tags = [{"name": tag} for tag in blog_post.tags]

    # Convert markdown content to HTML for Ghost
    html_content = convert_markdown_to_html(blog_post.content)
    html_length = len(html_content) if html_content else 0
    log_info("ghost_api", f"HTML content length: {html_length} characters")

    if html_length == 0:
        log_error(
            "ghost_api", "Converted HTML content is empty! Check markdown conversion."
        )
        return False, "Converted HTML content is empty", None

    # Format published date in ISO format with 'Z' suffix for UTC
    published_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # Construct the API endpoint with source=html parameter for proper HTML processing
    admin_api_url = urljoin(api_url, "ghost/api/admin/posts/?source=html")

    # Wrap HTML content in a Lexical HTML card for lossless conversion
    wrapped_html = f"""<!--kg-card-begin: html-->
{html_content}
<!--kg-card-end: html-->"""

    try:
        # Make the API request with JWT authentication
        headers = {
            "Authorization": f"Ghost {token}",
            "Content-Type": "application/json",
            "Accept-Version": api_version,
        }

        log_info("ghost_api", f"Sending request to Ghost API: {admin_api_url}")
        log_info("ghost_api", f"Post title: {blog_post.title}")
        log_info("ghost_api", f"Using source=html for lossless HTML conversion")

        # Update post_data to use wrapped HTML
        post_data = {
            "posts": [
                {
                    "title": blog_post.title,
                    "html": wrapped_html,
                    "status": "published",
                    "tags": ghost_tags,
                    "published_at": published_date,
                    "authors": [{"slug": "ghost"}],
                    "visibility": "public",
                    "featured": False,
                    "meta_title": blog_post.title,
                    "excerpt": (
                        blog_post.content[:150].replace("\n", " ") + "..."
                        if len(blog_post.content) > 150
                        else blog_post.content
                    ),
                }
            ]
        }

        response = requests.post(admin_api_url, json=post_data, headers=headers)

        # Debug logging for response
        log_info("ghost_api", f"Response status code: {response.status_code}")
        response_text_sample = (
            response.text[:150] + "..."
            if response.text and len(response.text) > 150
            else response.text
        )
        log_info("ghost_api", f"Response sample: {response_text_sample}")

        # Check if the request was successful
        if response.status_code in (200, 201):
            try:
                response_data = response.json()
                ghost_post_id = response_data.get("posts", [{}])[0].get("id")
                ghost_post_url = response_data.get("posts", [{}])[0].get("url")
                return (
                    True,
                    f"Successfully published to Ghost: {ghost_post_url}",
                    ghost_post_id,
                )
            except Exception as e:
                log_error("ghost_api", f"Error parsing successful response: {str(e)}")
                # Even if we can't parse the response, consider it a success if status code is good
                return (
                    True,
                    f"Successfully published to Ghost (status {response.status_code})",
                    None,
                )
        else:
            error_msg = f"Failed to publish to Ghost. Status code: {response.status_code}, Response: {response.text}"
            log_error("ghost_api", error_msg)
            return False, error_msg, None

    except Exception as e:
        error_msg = f"Error publishing to Ghost: {str(e)}"
        log_error("ghost_api", error_msg)
        return False, error_msg, None


def create_lexical_content(markdown_content: str) -> str:
    """
    Create a basic Lexical format representation for Ghost from markdown content.

    Args:
        markdown_content: The markdown content to convert

    Returns:
        Lexical format JSON string
    """
    if not markdown_content:
        return json.dumps(
            {
                "root": {
                    "children": [
                        {
                            "children": [
                                {
                                    "detail": 0,
                                    "format": 0,
                                    "mode": "normal",
                                    "style": "",
                                    "text": "No content available",
                                    "type": "extended-text",
                                    "version": 1,
                                }
                            ],
                            "direction": "ltr",
                            "format": "",
                            "indent": 0,
                            "type": "paragraph",
                            "version": 1,
                        }
                    ],
                    "direction": "ltr",
                    "format": "",
                    "indent": 0,
                    "type": "root",
                    "version": 1,
                }
            }
        )

    # Split content into paragraphs
    paragraphs = markdown_content.split("\n\n")

    # Create children array for Lexical format
    children = []

    for paragraph in paragraphs:
        if paragraph.strip():
            # For simplicity, we're treating each paragraph as plain text
            # In a full implementation, you would parse markdown syntax
            children.append(
                {
                    "children": [
                        {
                            "detail": 0,
                            "format": 0,
                            "mode": "normal",
                            "style": "",
                            "text": paragraph.strip(),
                            "type": "extended-text",
                            "version": 1,
                        }
                    ],
                    "direction": "ltr",
                    "format": "",
                    "indent": 0,
                    "type": "paragraph",
                    "version": 1,
                }
            )

    # If no children were created, add a default paragraph
    if not children:
        children.append(
            {
                "children": [
                    {
                        "detail": 0,
                        "format": 0,
                        "mode": "normal",
                        "style": "",
                        "text": markdown_content.strip(),
                        "type": "extended-text",
                        "version": 1,
                    }
                ],
                "direction": "ltr",
                "format": "",
                "indent": 0,
                "type": "paragraph",
                "version": 1,
            }
        )

    # Create the full Lexical structure
    lexical_structure = {
        "root": {
            "children": children,
            "direction": "ltr",
            "format": "",
            "indent": 0,
            "type": "root",
            "version": 1,
        }
    }

    return json.dumps(lexical_structure)


def convert_markdown_to_html(content: str) -> str:
    """
    Convert markdown content to HTML for Ghost.

    Args:
        content: Markdown content

    Returns:
        HTML content
    """
    # Check for empty content
    if not content or content.strip() == "":
        log_error("ghost_api", "Cannot convert empty content to HTML")
        return "<p>No content available</p>"  # Return minimal valid HTML

    log_info(
        "ghost_api",
        f"Converting markdown to HTML, content length: {len(content)} characters",
    )

    if MARKDOWN_AVAILABLE:
        try:
            # Convert markdown to HTML using the markdown library
            html_content = markdown.markdown(
                content,
                extensions=[
                    "markdown.extensions.fenced_code",
                    "markdown.extensions.tables",
                    "markdown.extensions.nl2br",
                    "markdown.extensions.toc",
                ],
            )

            # Check if conversion produced valid HTML
            if not html_content or html_content.strip() == "":
                log_warning(
                    "ghost_api",
                    "Markdown conversion produced empty HTML, using fallback",
                )
                return simple_text_to_html(content)

            return html_content
        except Exception as e:
            log_error("ghost_api", f"Error converting markdown to HTML: {str(e)}")
            return simple_text_to_html(content)
    else:
        # Basic conversion if markdown package is not available
        log_warning(
            "ghost_api",
            "Markdown package not installed. Install with 'pip install markdown' for better HTML conversion",
        )
        return simple_text_to_html(content)


def simple_text_to_html(content: str) -> str:
    """
    Convert plain text to HTML with basic formatting.

    Args:
        content: Plain text content

    Returns:
        Basic HTML representation of the text
    """
    if not content:
        return "<p>No content available</p>"

    # Very simple conversion for code blocks and headings
    lines = content.split("\n")
    in_code_block = False
    html_lines = []

    for line in lines:
        # Code blocks
        if line.startswith("```"):
            if in_code_block:
                html_lines.append("</pre></code>")
                in_code_block = False
            else:
                html_lines.append("<code><pre>")
                in_code_block = True
            continue

        # Headings
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        # Paragraphs
        elif line.strip() == "":
            html_lines.append("<br>")
        # Normal text
        else:
            if in_code_block:
                html_lines.append(line)
            else:
                html_lines.append(f"<p>{line}</p>")

    # If we're still in a code block at the end, close it
    if in_code_block:
        html_lines.append("</pre></code>")

    # If no HTML was generated, wrap the whole content in a paragraph
    if not html_lines:
        return f"<p>{content}</p>"

    return "\n".join(html_lines)


def generate_ghost_jwt_token(admin_api_key: str) -> Optional[str]:
    """
    Generate a JWT token for Ghost Admin API authentication.

    Args:
        admin_api_key: The Ghost Admin API key in format 'id:secret'

    Returns:
        JWT token string or None if failed
    """
    try:
        # Split the key into ID and SECRET
        key_parts = admin_api_key.split(":")
        if len(key_parts) != 2:
            log_error(
                "ghost_api", "Invalid Ghost Admin API key format. Should be 'id:secret'"
            )
            return None

        id_val, secret = key_parts

        # Prepare header and payload
        iat = int(datetime.now().timestamp())

        header = {"alg": "HS256", "typ": "JWT", "kid": id_val}
        payload = {
            "iat": iat,
            "exp": iat + 5 * 60,  # Token expires in 5 minutes
            "aud": "/admin/",
        }

        # Create the token (including decoding secret)
        token = jwt.encode(
            payload, bytes.fromhex(secret), algorithm="HS256", headers=header
        )

        # jwt.encode might return bytes in older versions of PyJWT
        if isinstance(token, bytes):
            token = token.decode("utf-8")

        return token

    except Exception as e:
        log_error("ghost_api", f"Error generating Ghost JWT token: {str(e)}")
        return None


def list_all_blog_posts():
    """
    List all available blog posts in the blogs directory with their IDs.
    This is a helper function for command-line usage.

    Returns:
        List of blog post dictionaries with id, title, and published status
    """
    try:
        # Ensure the blogs directory exists
        os.makedirs(BLOG_POSTS_DIR, exist_ok=True)

        # Get all JSON files in the blogs directory
        file_paths = glob.glob(os.path.join(BLOG_POSTS_DIR, "*.json"))
        blog_posts = []

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    blog_data = json.load(f)

                # Extract basic info for display
                blog_id = os.path.splitext(os.path.basename(file_path))[0]
                blog_posts.append(
                    {
                        "id": blog_id,
                        "title": blog_data.get("title", "Untitled"),
                        "published": blog_data.get("published", False),
                        "ghost_url": blog_data.get("ghost_url", None),
                    }
                )
            except Exception as e:
                log_error("blog_list", f"Error reading file {file_path}: {str(e)}")
                continue

        return blog_posts
    except Exception as e:
        log_error("blog_list", f"Error listing blog posts: {str(e)}")
        return []


# Modified main function with argument parsing
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Blog Management Agent")

    # Create a mutually exclusive group for the main action
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--file", help="Create a blog post from a JSON or markdown file"
    )
    action_group.add_argument(
        "--dir", help="Create blog posts from JSON or markdown files in a directory"
    )
    action_group.add_argument(
        "--publish", help="Publish an existing blog post to Ghost by ID"
    )
    action_group.add_argument(
        "--list",
        action="store_true",
        help="List all available blog posts with their IDs",
    )
    action_group.add_argument(
        "prompt",
        nargs="*",
        help="Blog management request in natural language",
        default=[],
    )

    # Parse the arguments
    args = parser.parse_args()

    # Check if Anthropic is available
    if not ANTHROPIC_AVAILABLE and not (
        args.file or args.dir or args.publish or args.list
    ):
        print(
            "Error: The Anthropic package is not installed. Please install it with 'pip install anthropic'."
        )
        sys.exit(1)

    # Handle list option
    if args.list:
        blog_posts = list_all_blog_posts()
        if blog_posts:
            print(f"Found {len(blog_posts)} blog posts:")
            for post in blog_posts:
                status = "Published" if post["published"] else "Draft"
                ghost_info = (
                    f" (Ghost URL: {post['ghost_url']})" if post["ghost_url"] else ""
                )
                print(f"  - ID: {post['id']}")
                print(f"    Title: {post['title']}")
                print(f"    Status: {status}{ghost_info}")
                print()
        else:
            print("No blog posts found.")
        sys.exit(0)

    # Setup the database connection if using file or directory mode
    db_conn = None
    if args.file or args.dir:
        try:
            db_conn, _ = setup_database()
        except Exception as e:
            log_error("main", f"Error setting up database: {str(e)}")
            print(
                f"Warning: Database connection failed. Will continue without tracking published files: {str(e)}"
            )

    # Handle file option
    if args.file:
        file_path = args.file
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

        blog_post_id = create_blog_from_file(file_path, db_conn)
        if blog_post_id:
            print(f"Successfully created blog post with ID: {blog_post_id}")
        else:
            print(f"Failed to create blog post from file: {file_path}")

        if db_conn:
            db_conn.close()
        sys.exit(0 if blog_post_id else 1)

    # Handle directory option
    elif args.dir:
        dir_path = args.dir
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            print(f"Error: Directory not found: {dir_path}")
            sys.exit(1)

        created_posts = process_directory(dir_path, db_conn)
        if created_posts:
            print(f"Successfully created {len(created_posts)} blog posts")
            for post_id in created_posts:
                print(f"  - Blog post ID: {post_id}")
        else:
            print(f"No blog posts were created from directory: {dir_path}")

        if db_conn:
            db_conn.close()
        sys.exit(0 if created_posts else 1)

    # Handle publish option
    elif args.publish:
        blog_id = args.publish

        # Verify the blog post exists
        read_result = read_blog_post(blog_id)
        if not read_result.success:
            print(f"Error: {read_result.message}")
            sys.exit(1)

        # Publish to Ghost
        publish_result = publish_blog_post(blog_id)
        if publish_result.success:
            print(f"Success: {publish_result.message}")
            if publish_result.data.get("ghost_url"):
                print(f"Ghost URL: {publish_result.data.get('ghost_url')}")
            sys.exit(0)
        else:
            print(f"Error: {publish_result.message}")
            sys.exit(1)

    # Handle normal prompt mode
    else:
        # Check if API key is set
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable is not set.")
            print("Please set it with: export ANTHROPIC_API_KEY=your-api-key")
            sys.exit(1)

        # Create Anthropic client
        client = Anthropic(api_key=api_key)

        # Get prompt from command line arguments or interactively
        if args.prompt:
            prompt = " ".join(args.prompt)
        else:
            print("\nWelcome to the Blog Management Agent!")
            print("-----------------------------------")
            print("This agent can help you manage blog posts, including:")
            print("- Creating new posts")
            print("- Updating existing posts")
            print("- Searching for posts")
            print("- Publishing and unpublishing posts")
            print("\nPlease enter your request below (or type 'exit' to quit):")
            prompt = input("> ")

            if prompt.lower() == "exit":
                sys.exit(0)

        # Run the agent
        run_agent(client, prompt)
