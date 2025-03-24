# /// script
# dependencies = [
#   "google-genai>=1.1.0",
# ]
# ///

"""
/// Example Usage

# generates jq command and executes it
uv run sfa_jq_gemini_v1.py --exe "Filter scores above 80 from data/analytics.json and save to high_scores.json"

# generates jq command only
uv run sfa_jq_gemini_v1.py "Filter scores above 80 from data/analytics.json and save to high_scores.json"

///
"""

import os
import sys
import argparse
import subprocess
from google import genai

JQ_PROMPT = """<purpose>
    You are a world-class expert at crafting precise jq commands for JSON processing.
    Your goal is to generate accurate, minimal jq commands that exactly match the user's data manipulation needs.
</purpose>

<instructions>
    <instruction>Return ONLY the jq command - no explanations, comments, or extra text.</instruction>
    <instruction>Always reference the input file specified in the user request (e.g., using -f flag if needed).</instruction>
    <instruction>Ensure the command follows jq best practices for efficiency and readability.</instruction>
    <instruction>Use the examples to understand different types of jq command patterns.</instruction>
    <instruction>When user asks to pipe or output to a file, use the correct syntax for the command and create a file name (if not specified) based on a shorted version of the user-request and the input file name.</instruction>
    <instruction>If the user request asks to pipe or output to a file, and no explicit directory is specified, use the directory of the input file.</instruction>
    <instruction>Output your response by itself, do not use backticks or markdown formatting. We're going to run your response as a shell command immediately.</instruction>
    <instruction>If your results you're working with a list of objects, default to outputting a valid json array.</instruction>
</instructions>

<examples>
    <example>
        <user-request>
            Select the "name" and "age" fields from data.json where age > 30
        </user-request>
        <jq-command>
            jq '[.[] | select(.age > 30) | {name, age}]' data.json
        </jq-command>
    </example>
    <example>
        <user-request>
            Count the number of entries in users.json with status "active"
        </user-request>
        <jq-command>
            jq '[.[] | select(.status == "active")] | length' users.json
        </jq-command>
    </example>
    <example>
        <user-request>
            Extract nested phone numbers from contacts.json using compact output
        </user-request>
        <jq-command>
            jq -c '.contact.info.phones' contacts.json
        </jq-command>
    </example>
    <example>
        <user-request>
            Convert log.json entries to CSV format with timestamp, level, message
        </user-request>
        <jq-command>
            jq -r '.[] | [.timestamp, .level, .message] | @csv' log.json
        </jq-command>
    </example>
    <example>
        <user-request>
            Sort records in people.json by age in descending order
        </user-request>
        <jq-command>
            jq 'sort_by(.age) | reverse' people.json
        </jq-command>
    </example>
    <example>
        <user-request>
            Save active users from data/users.json to a new file
        </user-request>
        <jq-command>
            jq '[.[] | select(.status == "active")]' data/users.json > data/active_users.json
        </jq-command>
    </example>
    <example>
        <user-request>
            Convert data.json to CSV for keys name, age, city and save in same directory
        </user-request>
        <jq-command>
            jq -r '.[] | [.name, .age, .city] | @csv' data/testing/data.json > data/testing/data_csv.csv
        </jq-command>
    </example>
    <example>
        <user-request>
            Filter scores above 80 from data/mock.json and save to ./high_scores.json
        </user-request>
        <jq-command>
            jq '[.[] | select(.score > 80)]' data/mock.json > ./high_scores.json
        </jq-command>
    </example>
</examples>


<user-request>
    {{user_request}}
</user-request>

Your jq command:"""


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate text using Gemini API")
    parser.add_argument(
        "prompt",
        help="The JQ command request to send to Gemini",
    )
    parser.add_argument(
        "--exe",
        action="store_true",
        help="Execute the generated JQ command",
    )
    args = parser.parse_args()

    # Configure the API key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable is not set")
        print("Please get your API key from https://aistudio.google.com/app/apikey")
        print("Then set it with: export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Initialize client
    client = genai.Client(
        api_key=GEMINI_API_KEY, http_options={"api_version": "v1alpha"}
    )

    try:
        # Replace {{user_request}} in the prompt template
        prompt = JQ_PROMPT.replace("{{user_request}}", args.prompt)

        # Generate JQ command
        response = client.models.generate_content(
            model="gemini-2.0-flash-001", contents=prompt
        )
        jq_command = response.text.strip()
        print("\n🤖 Generated JQ command:", jq_command)

        # Execute the command if --exe flag is present
        if args.exe:
            print("\n🔍 Executing command...")
            # Execute the command using subprocess
            result = subprocess.run(
                jq_command, shell=True, text=True, capture_output=True
            )
            if result.returncode != 0:
                print("\n❌ Error executing command:", result.stderr)
                sys.exit(1)
            print(result.stdout + result.stderr)

            if not result.stderr:
                print("\n✅ Command executed successfully")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
