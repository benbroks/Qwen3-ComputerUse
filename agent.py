"""BrowserAgent: Orchestrates Qwen3-VL via Ollama for browser automation."""

import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image
from rich.console import Console
from rich.table import Table

from computers.computer import Computer, EnvState


# Normalized coordinate space used by Qwen3-VL
NORMALIZED_SIZE = 1000

# System prompt with detailed tool descriptions
SYSTEM_PROMPT = f"""You are a GUI automation assistant. Use mouse and keyboard to interact with a browser.

* This is an interface to a browser GUI. You control it by outputting JSON actions.
* The screen's resolution is {NORMALIZED_SIZE}x{NORMALIZED_SIZE}. All coordinates use this grid.
* Whenever you intend to click on an element, consult the screenshot to determine the coordinates.
* If clicking failed to activate an element, try adjusting your cursor position so the tip visually falls on the element.
* Make sure to click buttons, links, icons, etc with the cursor tip in the CENTER of the element. Don't click on edges.
* Some actions may take time to complete, so you may need to wait and observe the results.

Available actions:

* `left_click`: Click the left mouse button at a specified (x, y) coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) coordinate (selects entire line/paragraph).
* `right_click`: Click the right mouse button at a specified (x, y) coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) coordinate on the screen.
* `mouse_move`: Move the cursor to a specified (x, y) coordinate without clicking.
* `left_click_drag`: Click and drag the cursor from current position to a specified (x, y) coordinate.
* `type`: Type a string of text on the keyboard. The current field will be cleared first.
* `key`: Performs key down presses on the keys passed in order, then releases in reverse order. Use for shortcuts like ["Control", "c"] or single keys like ["Enter"].
* `scroll`: Scroll the mouse wheel. Positive values scroll up, negative values scroll down.
* `wait`: Wait for specified number of seconds for the page to load or update.
* `terminate`: End the task and report completion status as "success" or "failure"."""

# JSON Schema for structured output - enforces correct attributes per action type
ACTION_SCHEMA = {
    "oneOf": [
        # Coordinate-based actions (click, move, drag)
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["left_click", "double_click", "triple_click", "right_click", "middle_click", "mouse_move", "left_click_drag"]
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "(x, y): The x (pixels from left edge) and y (pixels from top edge) coordinates."
                }
            },
            "required": ["action", "coordinate"],
            "additionalProperties": False
        },
        # Type action
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "const": "type"
                },
                "text": {
                    "type": "string",
                    "description": "The text to type on the keyboard."
                }
            },
            "required": ["action", "text"],
            "additionalProperties": False
        },
        # Key action
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "const": "key"
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of keys to press, e.g. [\"Control\", \"c\"] for copy, [\"Enter\"] for enter."
                }
            },
            "required": ["action", "keys"],
            "additionalProperties": False
        },
        # Scroll action
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "const": "scroll"
                },
                "pixels": {
                    "type": "integer",
                    "description": "Scroll amount. Positive values scroll up, negative values scroll down."
                }
            },
            "required": ["action", "pixels"],
            "additionalProperties": False
        },
        # Wait action
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "const": "wait"
                },
                "time": {
                    "type": "number",
                    "description": "The number of seconds to wait."
                }
            },
            "required": ["action", "time"],
            "additionalProperties": False
        },
        # Terminate action
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "const": "terminate"
                },
                "status": {
                    "type": "string",
                    "enum": ["success", "failure"],
                    "description": "The completion status of the task."
                }
            },
            "required": ["action", "status"],
            "additionalProperties": False
        }
    ]
}

console = Console()


class BrowserAgent:
    """Agent that uses Qwen3-VL via Ollama for browser automation."""

    def __init__(
        self,
        computer: Computer,
        query: str,
        model_name: str = "qwen3-vl:8b",
        max_steps: int = 50,
        context_window: int = 5,
        screenshot_dir: Optional[Path] = None,
    ):
        self._computer = computer
        self._query = query
        self._model_name = model_name
        self._max_steps = max_steps
        self._context_window = context_window
        self._screenshot_dir = screenshot_dir

        # Ollama native API endpoint
        self._ollama_url = "http://localhost:11434/api/chat"

        # Conversation history: list of (screenshot_b64, action_json, reasoning)
        self._history: list[dict[str, Any]] = []
        self._action_count = 0
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3

    def run(self) -> dict[str, Any]:
        """Main agent loop. Returns result dict."""
        console.print(f"[bold cyan]Task:[/bold cyan] {self._query}\n")

        # Get initial screenshot
        state = self._computer.current_state()
        self._save_screenshot(state.screenshot, "initial")

        while self._action_count < self._max_steps:
            try:
                result = self._run_one_step(state)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user[/yellow]")
                return self._make_result("failure", state.url, "User interrupted")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_consecutive_failures:
                    return self._make_result("failure", state.url, str(e))
                # Retry with fresh screenshot
                state = self._computer.current_state()
                continue

            if result["done"]:
                return self._make_result(
                    result.get("status", "success"),
                    state.url,
                    result.get("reasoning"),
                )

            state = result["state"]
            self._consecutive_failures = 0

        console.print(f"[yellow]Reached max steps ({self._max_steps})[/yellow]")
        return self._make_result("failure", state.url, "Max steps reached")

    def _run_one_step(self, state: EnvState) -> dict[str, Any]:
        """Execute one step of the agent loop."""
        # Resize screenshot to 1000x1000 for model
        screenshot_b64 = self._prepare_screenshot(state.screenshot)

        # Build messages for Ollama
        messages = self._build_messages(screenshot_b64)

        # Call Ollama
        with console.status("[bold green]Thinking...", spinner="dots"):
            response, thinking = self._call_ollama(messages)

        # Parse response
        action, _ = self._parse_response(response)
        reasoning = thinking  # Use the thinking field as reasoning

        if action is None:
            console.print(f"[yellow]Could not parse action from response[/yellow]")
            console.print(f"Raw response: {response[:500]}")
            raise ValueError("Failed to parse action")

        # Display action
        self._display_action(reasoning, action)

        # Check for termination
        if action.get("action") == "terminate":
            return {
                "done": True,
                "status": action.get("status", "success"),
                "reasoning": reasoning,
            }

        # Check for answer action
        if action.get("action") == "answer":
            console.print(f"\n[bold green]Answer:[/bold green] {action.get('text', '')}\n")
            return {
                "done": True,
                "status": "success",
                "reasoning": action.get("text", ""),
            }

        # Execute action
        new_state = self._execute_action(action)
        self._action_count += 1
        self._save_screenshot(new_state.screenshot, f"step_{self._action_count:03d}")

        # Update history (sliding window)
        self._history.append({
            "screenshot_b64": screenshot_b64,
            "action": action,
            "reasoning": reasoning,
        })
        if len(self._history) > self._context_window:
            self._history.pop(0)

        return {"done": False, "state": new_state}

    def _prepare_screenshot(self, screenshot_bytes: bytes) -> str:
        """Resize screenshot to 1000x1000 and return base64."""
        img = Image.open(BytesIO(screenshot_bytes))
        img_resized = img.resize((NORMALIZED_SIZE, NORMALIZED_SIZE), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img_resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_messages(self, current_screenshot_b64: str) -> list[dict]:
        """Build message list for Ollama native API with conversation history."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Build history summary of previous actions taken
        if self._history:
            previous_actions = [json.dumps(turn["action"]) for turn in self._history]
            history_text = "Previous actions taken:\n" + "\n".join(previous_actions)
        else:
            history_text = ""

        # Current turn with screenshot and task (include history in user message)
        user_content = f"Task: {self._query}"
        if history_text:
            user_content = f"{history_text}\n\n{user_content}"

        messages.append({
            "role": "user",
            "content": user_content,
            "images": [current_screenshot_b64],
        })

        return messages

    def _call_ollama(self, messages: list[dict], debug: bool = False) -> str:
        """Call Ollama native API and return response text."""
        if debug:
            console.print("\n[dim]--- DEBUG: Messages being sent ---[/dim]")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                images = msg.get("images", [])
                # Truncate long content
                display = content[:300] + "..." if len(content) > 300 else content
                img_info = f" + {len(images)} image(s)" if images else ""
                console.print(f"[dim][{i}] {role}: {display}{img_info}[/dim]")
            console.print("[dim]--- END DEBUG ---[/dim]\n")

        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "format": ACTION_SCHEMA,  # Enforce structured output
        }

        try:
            response = requests.post(
                self._ollama_url,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            if debug:
                console.print(f"[dim]--- DEBUG: Raw response ---[/dim]")
                console.print(f"[dim]Done: {data.get('done')}[/dim]")
                console.print(f"[dim]Done reason: {data.get('done_reason')}[/dim]")
                msg = data.get("message", {})
                console.print(f"[dim]Data:[/dim]")
                console.print(f"[dim]{data}[/dim]")
                console.print(f"[dim]Message role: {msg.get('role')}[/dim]")
                content = msg.get("content", "")
                thinking = msg.get("thinking", "")
                console.print(f"[dim]Message content:[/dim]")
                console.print(f"[dim]{repr(content)}[/dim]")
                if thinking:
                    console.print(f"[dim]Thinking:[/dim]")
                    console.print(f"[dim]{thinking}[/dim]")
                console.print("[dim]--- END DEBUG ---[/dim]\n")

            if "error" in data:
                raise Exception(f"Ollama error: {data['error']}")

            msg = data.get("message", {})
            content = msg.get("content", "")
            thinking = msg.get("thinking", "")

            return content, thinking

        except requests.exceptions.ConnectionError as e:
            console.print("[red bold]Cannot connect to Ollama.[/red bold]")
            console.print("Make sure Ollama is running: [cyan]ollama serve[/cyan]")
            console.print(f"And the model is available: [cyan]ollama pull {self._model_name}[/cyan]")
            raise
        except Exception as e:
            console.print(f"[red]Exception type: {type(e).__name__}[/red]")
            console.print(f"[red]Exception: {e}[/red]")
            raise

    def _parse_response(self, response: str) -> tuple[Optional[dict], Optional[str]]:
        """Parse JSON action from model response (structured output guarantees valid JSON)."""
        text = response.strip()

        if not text:
            return None, None

        try:
            action = json.loads(text)
            if "action" in action:
                return action, None
            else:
                console.print(f"[yellow]Response missing 'action' field: {action}[/yellow]")
                return None, None
        except json.JSONDecodeError as e:
            console.print(f"[yellow]JSON parse error: {e}[/yellow]")
            return None, None

    def _execute_action(self, action: dict) -> EnvState:
        """Execute action and return new state."""
        action_type = action.get("action")
        coordinate = action.get("coordinate", [0, 0])

        # Denormalize coordinates
        x = self._denormalize_x(coordinate[0]) if coordinate else 0
        y = self._denormalize_y(coordinate[1]) if len(coordinate) > 1 else 0

        if action_type == "left_click":
            return self._computer.click_at(x, y)

        elif action_type == "double_click":
            return self._computer.double_click_at(x, y)

        elif action_type == "triple_click":
            return self._computer.triple_click_at(x, y)

        elif action_type == "right_click":
            return self._computer.right_click_at(x, y)

        elif action_type == "middle_click":
            return self._computer.middle_click_at(x, y)

        elif action_type == "mouse_move":
            return self._computer.hover_at(x, y)

        elif action_type == "left_click_drag":
            # For drag, we need start position (from previous action or center)
            # and destination (from coordinate)
            return self._computer.drag_to(x, y)

        elif action_type == "type":
            text = action.get("text", "")
            return self._computer.type_text(text)

        elif action_type == "key":
            keys = action.get("keys", [])
            return self._computer.key_combination(keys)

        elif action_type == "scroll":
            pixels = action.get("pixels", 0)
            # Denormalize scroll amount
            actual_pixels = int((abs(pixels) / NORMALIZED_SIZE) * self._computer.screen_size()[1])
            if pixels < 0:
                actual_pixels = -actual_pixels
            return self._computer.scroll(actual_pixels)

        elif action_type == "wait":
            wait_time = action.get("time", 1)
            time.sleep(wait_time)
            return self._computer.current_state()

        else:
            console.print(f"[yellow]Unknown action: {action_type}[/yellow]")
            return self._computer.current_state()

    def _denormalize_x(self, x: int) -> int:
        """Convert normalized x (0-1000) to actual screen x."""
        return int((x / NORMALIZED_SIZE) * self._computer.screen_size()[0])

    def _denormalize_y(self, y: int) -> int:
        """Convert normalized y (0-1000) to actual screen y."""
        return int((y / NORMALIZED_SIZE) * self._computer.screen_size()[1])

    def _display_action(self, reasoning: Optional[str], action: dict):
        """Display action in rich table format."""
        table = Table(expand=True)
        table.add_column("Model Reasoning", header_style="magenta", ratio=1)
        table.add_column("Action", header_style="cyan", ratio=1)

        action_str = f"Name: {action.get('action', 'unknown')}"
        for key, value in action.items():
            if key != "action":
                action_str += f"\nArgs:\n  {key}: {value}"

        table.add_row(reasoning or "(no reasoning)", action_str)
        console.print(table)
        console.print()

    def _save_screenshot(self, screenshot_bytes: bytes, name: str):
        """Save screenshot to disk if screenshot_dir is set."""
        if self._screenshot_dir:
            path = self._screenshot_dir / f"{name}.png"
            path.write_bytes(screenshot_bytes)

    def _make_result(self, status: str, final_url: str, reasoning: Optional[str] = None) -> dict:
        """Create result dictionary."""
        return {
            "status": status,
            "action_count": self._action_count,
            "final_url": final_url,
            "reasoning": reasoning,
        }
