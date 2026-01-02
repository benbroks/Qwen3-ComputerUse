#!/usr/bin/env python3
"""CLI entry point for Qwen3-VL Computer Use agent."""

import argparse
import sys
import os
from pathlib import Path

from agent import BrowserAgent
from computers.playwright import PlaywrightComputer


SCREEN_SIZE = (1440, 900)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Browser automation agent using Qwen3-VL via Ollama"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The task for the agent to execute",
    )
    parser.add_argument(
        "--initial-url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (default: https://www.google.com)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum actions before forced stop (default: 50)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=5,
        help="Number of recent turns to keep in context (default: 5)",
    )
    parser.add_argument(
        "--save-screenshots",
        action="store_true",
        default=False,
        help="Save screenshots to ./screenshots/<session_id>/",
    )
    parser.add_argument(
        "--highlight-mouse",
        action="store_true",
        default=False,
        help="Show visual cursor indicator (for debugging)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl:8b",
        help="Ollama model name (default: qwen3-vl:8b)",
    )

    args = parser.parse_args()

    # Create screenshot directory if needed
    screenshot_dir = None
    if args.save_screenshots:
        import uuid
        session_id = str(uuid.uuid4())[:8]
        screenshot_dir = Path("screenshots") / session_id
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving screenshots to: {screenshot_dir}")

    env = PlaywrightComputer(
        screen_size=SCREEN_SIZE,
        initial_url=args.initial_url,
        highlight_mouse=args.highlight_mouse,
    )

    with env as browser:
        agent = BrowserAgent(
            computer=browser,
            query=args.query,
            model_name=args.model,
            max_steps=args.max_steps,
            context_window=args.context_window,
            screenshot_dir=screenshot_dir,
        )
        result = agent.run()

    # Print final result
    print("\n" + "=" * 60)
    print(f"Status: {result['status']}")
    print(f"Actions: {result['action_count']}")
    print(f"Final URL: {result['final_url']}")
    if result.get('reasoning'):
        print(f"Reasoning: {result['reasoning']}")
    print("=" * 60)

    return 0 if result['status'] == 'success' else 1


if __name__ == "__main__":
    sys.exit(main())
