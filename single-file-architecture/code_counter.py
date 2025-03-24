#!/usr/bin/env python3
"""
Code Counter - Counts lines of code in Python files within the audio_transcription_pipeline
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import re
from collections import defaultdict
from dataclasses import dataclass
import json

# Add parent directory to path if running the script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class CodeStats:
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    docstring_lines: int = 0
    files_analyzed: int = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Count lines of code in Python files")
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory to analyze (default: current directory)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=["venv", ".venv", "env", ".env", "__pycache__", ".git"],
        help="Directories to exclude",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--show-files", action="store_true", help="Show details for each file"
    )
    return parser.parse_args()


def count_lines(file_path: Path) -> Tuple[int, int, int, int, int]:
    """Count different types of lines in a Python file"""
    total_lines = 0
    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    docstring_lines = 0

    in_docstring = False
    docstring_delimiter = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_lines += 1

            # Remove trailing whitespace for analysis but keep original for length
            stripped_line = line.strip()

            # Check for docstring start/end
            if not in_docstring and (
                stripped_line.startswith('"""') or stripped_line.startswith("'''")
            ):
                in_docstring = True
                docstring_delimiter = stripped_line[0:3]
                docstring_lines += 1

                # Check if docstring ends on the same line
                if (
                    stripped_line.endswith(docstring_delimiter)
                    and len(stripped_line) > 3
                ):
                    in_docstring = False
                continue

            if in_docstring:
                docstring_lines += 1
                if stripped_line.endswith(docstring_delimiter):
                    in_docstring = False
                continue

            # Check for blank lines
            if not stripped_line:
                blank_lines += 1
                continue

            # Check for comments
            if stripped_line.startswith("#"):
                comment_lines += 1
                continue

            # If we get here, it's a code line
            code_lines += 1

    return total_lines, code_lines, comment_lines, blank_lines, docstring_lines


def scan_directory(
    directory: Path, exclude_dirs: List[str]
) -> Dict[str, Dict[str, int]]:
    """Scan directory recursively for Python files"""
    file_stats = {}

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # Process Python files
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                rel_path = file_path.relative_to(directory)

                total, code, comment, blank, docstring = count_lines(file_path)

                file_stats[str(rel_path)] = {
                    "total_lines": total,
                    "code_lines": code,
                    "comment_lines": comment,
                    "blank_lines": blank,
                    "docstring_lines": docstring,
                }

    return file_stats


def summarize_stats(file_stats: Dict[str, Dict[str, int]]) -> CodeStats:
    """Summarize statistics across all files"""
    summary = CodeStats()
    summary.files_analyzed = len(file_stats)

    for stats in file_stats.values():
        summary.total_lines += stats["total_lines"]
        summary.code_lines += stats["code_lines"]
        summary.comment_lines += stats["comment_lines"]
        summary.blank_lines += stats["blank_lines"]
        summary.docstring_lines += stats["docstring_lines"]

    return summary


def summarize_by_directory(
    file_stats: Dict[str, Dict[str, int]], base_dir: Path
) -> Dict[str, CodeStats]:
    """Summarize statistics by directory"""
    dir_stats = defaultdict(lambda: CodeStats())

    for file_path, stats in file_stats.items():
        # Get the directory part of the path
        file_path_obj = Path(file_path)
        if file_path_obj.parent == Path("."):
            # For files in the root directory
            dir_name = "[root]"
        else:
            dir_name = str(file_path_obj.parent)

        dir_stats[dir_name].files_analyzed += 1
        dir_stats[dir_name].total_lines += stats["total_lines"]
        dir_stats[dir_name].code_lines += stats["code_lines"]
        dir_stats[dir_name].comment_lines += stats["comment_lines"]
        dir_stats[dir_name].blank_lines += stats["blank_lines"]
        dir_stats[dir_name].docstring_lines += stats["docstring_lines"]

    return dict(dir_stats)


def display_results_rich(
    summary: CodeStats,
    dir_stats: Dict[str, CodeStats],
    file_stats: Dict[str, Dict[str, int]],
    show_files: bool = False,
):
    """Display results using rich formatting"""
    console = Console()

    # Show header with project summary
    console.print(
        Panel.fit(
            f"[bold blue]Python Code Analysis[/] - {summary.files_analyzed} files analyzed",
            title="Code Counter",
        )
    )

    # Summary table
    summary_table = Table(show_header=True, header_style="bold cyan", expand=False)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Count", justify="right")
    summary_table.add_column("Percentage", justify="right")

    total = summary.total_lines or 1  # Avoid division by zero
    summary_table.add_row("Total Lines", str(summary.total_lines), "100.0%")
    summary_table.add_row(
        "Code Lines", str(summary.code_lines), f"{summary.code_lines/total*100:.1f}%"
    )
    summary_table.add_row(
        "Comment Lines",
        str(summary.comment_lines),
        f"{summary.comment_lines/total*100:.1f}%",
    )
    summary_table.add_row(
        "Docstring Lines",
        str(summary.docstring_lines),
        f"{summary.docstring_lines/total*100:.1f}%",
    )
    summary_table.add_row(
        "Blank Lines", str(summary.blank_lines), f"{summary.blank_lines/total*100:.1f}%"
    )
    summary_table.add_row("Files Analyzed", str(summary.files_analyzed), "")

    console.print(summary_table)

    # Directory breakdown
    console.print("\n[bold]Directory Breakdown:[/]")
    dir_table = Table(show_header=True, header_style="bold magenta", expand=True)
    dir_table.add_column("Directory", style="dim")
    dir_table.add_column("Files", justify="right")
    dir_table.add_column("Code Lines", justify="right")
    dir_table.add_column("Comment Lines", justify="right")
    dir_table.add_column("Docstring Lines", justify="right")
    dir_table.add_column("Blank Lines", justify="right")
    dir_table.add_column("Total Lines", justify="right")

    # Sort directories by code lines in descending order
    sorted_dirs = sorted(dir_stats.items(), key=lambda x: x[1].code_lines, reverse=True)

    for dir_name, stats in sorted_dirs:
        dir_table.add_row(
            dir_name,
            str(stats.files_analyzed),
            str(stats.code_lines),
            str(stats.comment_lines),
            str(stats.docstring_lines),
            str(stats.blank_lines),
            str(stats.total_lines),
        )

    console.print(dir_table)

    # File details if requested
    if show_files:
        console.print("\n[bold]File Details:[/]")
        file_table = Table(show_header=True, header_style="bold blue", expand=True)
        file_table.add_column("File", style="dim")
        file_table.add_column("Code Lines", justify="right")
        file_table.add_column("Comment Lines", justify="right")
        file_table.add_column("Docstring Lines", justify="right")
        file_table.add_column("Blank Lines", justify="right")
        file_table.add_column("Total Lines", justify="right")

        # Sort files by total lines in descending order
        sorted_files = sorted(
            file_stats.items(), key=lambda x: x[1]["total_lines"], reverse=True
        )

        for file_path, stats in sorted_files:
            file_table.add_row(
                str(file_path),
                str(stats["code_lines"]),
                str(stats["comment_lines"]),
                str(stats["docstring_lines"]),
                str(stats["blank_lines"]),
                str(stats["total_lines"]),
            )

        console.print(file_table)


def display_results_text(
    summary: CodeStats,
    dir_stats: Dict[str, CodeStats],
    file_stats: Dict[str, Dict[str, int]],
    show_files: bool = False,
):
    """Display results in plain text format"""
    print(f"=== Python Code Analysis - {summary.files_analyzed} files analyzed ===")
    print("\nSummary:")
    total = summary.total_lines or 1  # Avoid division by zero
    print(f"  Total Lines:      {summary.total_lines}")
    print(
        f"  Code Lines:       {summary.code_lines} ({summary.code_lines/total*100:.1f}%)"
    )
    print(
        f"  Comment Lines:    {summary.comment_lines} ({summary.comment_lines/total*100:.1f}%)"
    )
    print(
        f"  Docstring Lines:  {summary.docstring_lines} ({summary.docstring_lines/total*100:.1f}%)"
    )
    print(
        f"  Blank Lines:      {summary.blank_lines} ({summary.blank_lines/total*100:.1f}%)"
    )
    print(f"  Files Analyzed:   {summary.files_analyzed}")

    print("\nDirectory Breakdown:")
    # Sort directories by code lines in descending order
    sorted_dirs = sorted(dir_stats.items(), key=lambda x: x[1].code_lines, reverse=True)

    print(
        f"{'Directory':<30} {'Files':>5} {'Code':>8} {'Comment':>8} {'Docstring':>10} {'Blank':>8} {'Total':>8}"
    )
    print("-" * 80)
    for dir_name, stats in sorted_dirs:
        name = dir_name[:28] + ".." if len(dir_name) > 30 else dir_name
        print(
            f"{name:<30} {stats.files_analyzed:>5} {stats.code_lines:>8} {stats.comment_lines:>8} "
            f"{stats.docstring_lines:>10} {stats.blank_lines:>8} {stats.total_lines:>8}"
        )

    if show_files:
        print("\nFile Details:")
        # Sort files by total lines in descending order
        sorted_files = sorted(
            file_stats.items(), key=lambda x: x[1]["total_lines"], reverse=True
        )

        print(
            f"{'File':<40} {'Code':>8} {'Comment':>8} {'Docstring':>10} {'Blank':>8} {'Total':>8}"
        )
        print("-" * 90)
        for file_path, stats in sorted_files:
            name = str(file_path)
            if len(name) > 40:
                name = name[:37] + "..."
            print(
                f"{name:<40} {stats['code_lines']:>8} {stats['comment_lines']:>8} "
                f"{stats['docstring_lines']:>10} {stats['blank_lines']:>8} {stats['total_lines']:>8}"
            )


def export_json(
    summary: CodeStats,
    dir_stats: Dict[str, CodeStats],
    file_stats: Dict[str, Dict[str, int]],
) -> str:
    """Export results as JSON"""
    # Convert dataclasses to dictionaries
    dir_stats_dict = {
        k: {
            "total_lines": v.total_lines,
            "code_lines": v.code_lines,
            "comment_lines": v.comment_lines,
            "blank_lines": v.blank_lines,
            "docstring_lines": v.docstring_lines,
            "files_analyzed": v.files_analyzed,
        }
        for k, v in dir_stats.items()
    }

    summary_dict = {
        "total_lines": summary.total_lines,
        "code_lines": summary.code_lines,
        "comment_lines": summary.comment_lines,
        "blank_lines": summary.blank_lines,
        "docstring_lines": summary.docstring_lines,
        "files_analyzed": summary.files_analyzed,
    }

    result = {
        "summary": summary_dict,
        "directories": dir_stats_dict,
        "files": file_stats,
    }

    return json.dumps(result, indent=2)


def main():
    args = parse_args()

    base_dir = Path(args.dir)
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return 1

    # Scan directory for Python files
    file_stats = scan_directory(base_dir, args.exclude)
    if not file_stats:
        print(f"No Python files found in {base_dir}")
        return 0

    # Calculate summary statistics
    summary = summarize_stats(file_stats)
    dir_stats = summarize_by_directory(file_stats, base_dir)

    # Output results
    if args.json:
        json_output = export_json(summary, dir_stats, file_stats)
        print(json_output)
    elif RICH_AVAILABLE:
        display_results_rich(summary, dir_stats, file_stats, args.show_files)
    else:
        display_results_text(summary, dir_stats, file_stats, args.show_files)

    return 0


if __name__ == "__main__":
    sys.exit(main())
