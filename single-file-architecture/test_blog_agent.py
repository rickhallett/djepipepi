#!/usr/bin/env python3

import os
import json
import pytest
import shutil
import tempfile
from unittest.mock import patch, MagicMock, ANY

# Mock key dependencies before import
mock_anthropic = MagicMock()
mock_anthropic_class = MagicMock()
mock_anthropic.Anthropic = mock_anthropic_class

mock_modules = {
    "anthropic": mock_anthropic,
    "dotenv": MagicMock(),
    "frontmatter": MagicMock(),
    "markdown": MagicMock(),
    "rich": MagicMock(),
}

with patch.dict("sys.modules", mock_modules):
    # Import the blog agent module after mocking dependencies
    import sfa_blog_agent_v2 as blog_agent
    from sfa_blog_agent_v2 import (
        BlogPost,
        BlogOperationResult,
    )

# Constants for testing
TEST_TITLE = "Test Blog Post"
TEST_CONTENT = "# Test Content\n\nThis is a test blog post content."
TEST_AUTHOR = "Richard Hallett"
TEST_TAGS = ["science"]


# Test the BlogPost class
class TestBlogPost:
    def test_blog_post_creation(self):
        """Test creating a BlogPost object."""
        blog_post = BlogPost(
            title=TEST_TITLE, content=TEST_CONTENT, author=TEST_AUTHOR, tags=TEST_TAGS
        )

        assert blog_post.title == TEST_TITLE
        assert blog_post.content == TEST_CONTENT
        assert blog_post.author == TEST_AUTHOR
        assert blog_post.tags == TEST_TAGS
        assert blog_post.published is False
        assert blog_post.id is None

    def test_blog_post_to_dict(self):
        """Test converting a BlogPost to a dictionary."""
        blog_post = BlogPost(
            id="test-id",
            title=TEST_TITLE,
            content=TEST_CONTENT,
            author=TEST_AUTHOR,
            tags=TEST_TAGS,
            published=True,
            created_at="2025-03-24T00:00:00",
            updated_at="2025-03-24T01:00:00",
            ghost_id="ghost-id",
            ghost_url="https://example.com/post",
        )

        blog_dict = blog_post.to_dict()

        assert isinstance(blog_dict, dict)
        assert blog_dict["id"] == "test-id"
        assert blog_dict["title"] == TEST_TITLE
        assert blog_dict["content"] == TEST_CONTENT
        assert blog_dict["author"] == TEST_AUTHOR
        assert blog_dict["tags"] == TEST_TAGS
        assert blog_dict["published"] is True
        assert blog_dict["created_at"] == "2025-03-24T00:00:00"
        assert blog_dict["updated_at"] == "2025-03-24T01:00:00"
        assert blog_dict["ghost_id"] == "ghost-id"
        assert blog_dict["ghost_url"] == "https://example.com/post"

    def test_blog_post_from_dict(self):
        """Test creating a BlogPost from a dictionary."""
        blog_dict = {
            "id": "test-id",
            "title": TEST_TITLE,
            "content": TEST_CONTENT,
            "author": TEST_AUTHOR,
            "tags": TEST_TAGS,
            "published": True,
            "created_at": "2025-03-24T00:00:00",
            "updated_at": "2025-03-24T01:00:00",
            "ghost_id": "ghost-id",
            "ghost_url": "https://example.com/post",
        }

        blog_post = BlogPost.from_dict(blog_dict)

        assert blog_post.id == "test-id"
        assert blog_post.title == TEST_TITLE
        assert blog_post.content == TEST_CONTENT
        assert blog_post.author == TEST_AUTHOR
        assert blog_post.tags == TEST_TAGS
        assert blog_post.published is True
        assert blog_post.created_at == "2025-03-24T00:00:00"
        assert blog_post.updated_at == "2025-03-24T01:00:00"
        assert blog_post.ghost_id == "ghost-id"
        assert blog_post.ghost_url == "https://example.com/post"


# Test the BlogOperationResult class
class TestBlogOperationResult:
    def test_blog_operation_result_creation(self):
        """Test creating a BlogOperationResult object."""
        result = BlogOperationResult(
            success=True, message="Operation successful", data={"key": "value"}
        )

        assert result.success is True
        assert result.message == "Operation successful"
        assert result.data == {"key": "value"}

    def test_blog_operation_result_to_dict(self):
        """Test converting a BlogOperationResult to a dictionary."""
        result = BlogOperationResult(
            success=True, message="Operation successful", data={"key": "value"}
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["message"] == "Operation successful"
        assert result_dict["data"] == {"key": "value"}


if __name__ == "__main__":
    pytest.main(["-v"])
