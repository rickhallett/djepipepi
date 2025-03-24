#!/usr/bin/env python3


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
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

# Try to import rich for pretty console output, but handle gracefully if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Try to import Anthropic for Claude integration, but handle gracefully if not available
try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

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

# Create a data directory for blog posts
BLOG_POSTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "blog_posts"
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


# Blog CRUD Operations
def create_blog_post(
    title: str, content: str, author: str, tags: list = None
) -> BlogOperationResult:
    """
    Create a new blog post.

    Args:
        title: Title of the blog post
        content: Content of the blog post
        author: Author of the blog post
        tags: Optional list of tags

    Returns:
        BlogOperationResult with result or error message
    """
    log_info("create_tool", f"Creating blog post: {title}")

    try:
        # Create directory if it doesn't exist
        os.makedirs(BLOG_POSTS_DIR, exist_ok=True)

        # Generate a unique ID and timestamps
        post_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        # Create the blog post
        blog_post = BlogPost(
            id=post_id,
            title=title,
            content=content,
            author=author,
            tags=tags or [],
            published=False,
            created_at=current_time,
            updated_at=current_time,
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
) -> BlogOperationResult:
    """
    Update a blog post by ID.

    Args:
        post_id: The ID of the blog post to update
        title: Optional new title
        content: Optional new content
        tags: Optional new tags
        published: Optional new publication status

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

        if tags is not None:
            blog_post_data["tags"] = tags

        if published is not None:
            blog_post_data["published"] = published

        # Update the updated_at timestamp
        blog_post_data["updated_at"] = datetime.now().isoformat()

        # Save the updated blog post to the JSON file
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
    Publish a blog post by ID.

    Args:
        post_id: The ID of the blog post to publish

    Returns:
        BlogOperationResult with the published blog post or error message
    """
    log_info("update_tool", f"Publishing blog post with ID: {post_id}")
    return update_blog_post(post_id, published=True)


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
    Blog agent that provides an interface for AI-assisted blog management.
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
        system_prompt = """You are a helpful AI assistant with blog management capabilities.
You have access to tools that can create, read, update, delete, and search blog posts.
Always think step by step about what you need to do before taking any action.
Be helpful in suggesting blog post ideas and improvements when asked.

Available commands:
- create_post: Create a new blog post (title, content, author, tags)
- get_post: Get a blog post by ID (post_id)
- update_post: Update a blog post (post_id, title?, content?, tags?, published?)
- delete_post: Delete a blog post (post_id)
- list_posts: List blog posts (tag?, author?, published_only?)
- search_posts: Search blog posts (query, search_content?, tag?, author?)
- publish_post: Publish a blog post (post_id)
- unpublish_post: Unpublish a blog post (post_id)
"""

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


if __name__ == "__main__":
    # Check if Anthropic is available
    if not ANTHROPIC_AVAILABLE:
        print(
            "Error: The Anthropic package is not installed. Please install it with 'pip install anthropic'."
        )
        sys.exit(1)

    # Check if API key is set
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Please set it with: export ANTHROPIC_API_KEY=your-api-key")
        sys.exit(1)

    # Create Anthropic client
    client = Anthropic(api_key=api_key)

    # Get prompt from command line arguments or interactively
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
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

        while prompt.lower() == "exit":
            sys.exit(0)

    # Run the agent
    run_agent(client, prompt)
