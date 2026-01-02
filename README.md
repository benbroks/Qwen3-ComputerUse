# Qwen3-VL Computer Use with Playwright

A local computer use agent using **Qwen3-VL:8B** via **Ollama** with **Playwright** as the browser automation layer.

## What is this?

This project lets you control a web browser using natural language. You describe what you want to do, and an AI vision model (Qwen3-VL running locally via Ollama) looks at the browser screen and performs the actions for you.

The agent:
1. Takes a screenshot of the browser
2. Sends it to Qwen3-VL along with your task
3. Receives an action (click, type, scroll, etc.) with coordinates
4. Executes the action via Playwright
5. Repeats until the task is complete

## Prerequisites

- **macOS or Linux** (Windows untested)
- **Python 3.10+**
- **Ollama** installed and running

## Installation

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Or download from [ollama.ai](https://ollama.ai)

### 2. Pull the Qwen3-VL model

```bash
ollama pull qwen3-vl:8b
```

This downloads the ~8GB model. It may take a few minutes.

### 3. Clone and set up Python environment

```bash
git clone <this-repo>
cd Qwen3-ComputerUse

# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Always activate the virtual environment (`source venv/bin/activate`) before running the agent.

### 4. Install Playwright browsers

```bash
playwright install chromium
```

## Usage

### 1. Start Ollama

In one terminal:
```bash
ollama serve
```

### 2. Run the agent

In another terminal:
```bash
python main.py --query "your task here"
```

### CLI Options

```
python main.py --query "search for cats on google" [options]

Required:
  --query TEXT          The task for the agent to execute

Options:
  --initial-url URL     Starting URL (default: https://www.google.com)
  --max-steps N         Maximum actions before forced stop (default: 50)
  --context-window N    Number of recent turns to keep (default: 5)
  --save-screenshots    Save screenshots to ./screenshots/<session_id>/
  --highlight-mouse     Show visual cursor indicator (debugging)
  --model NAME          Ollama model name (default: qwen3-vl:8b)
```

## Examples

### Search Google
```bash
python main.py --query "search for 'best pizza in NYC'"
```

### Navigate and interact
```bash
python main.py --query "go to wikipedia and search for Alan Turing"
```

### Multi-step tasks
```bash
python main.py --query "search for weather in San Francisco and tell me the temperature"
```

### Debug with screenshots
```bash
python main.py --query "click the search box" --save-screenshots --highlight-mouse
```

## Architecture

```
                              main.py
                         (CLI entry point)
                                │
                                ▼
                          BrowserAgent
           - Manages conversation history (sliding window)
           - Calls Ollama for inference
           - Parses JSON action responses
           - Maps Qwen actions → Playwright calls
           - Handles coordinate denormalization (1000→actual)
                                │
                                ▼
                        PlaywrightComputer
           - Browser lifecycle (launch, screenshot, close)
           - Action execution (click, type, scroll, etc.)
           - Returns EnvState (screenshot bytes + URL)
                                │
                                ▼
                       Ollama + Qwen3-VL:8B
           - Local inference server
           - OpenAI-compatible API at localhost:11434
```

## How It Works

### Coordinate System

Qwen3-VL is fine-tuned on a 1000x1000 normalized coordinate system. The agent:
1. Resizes screenshots to 1000x1000 before sending to the model
2. Receives coordinates in the 0-1000 range
3. Denormalizes to actual screen dimensions (1440x900) at execution time

### Context Management

The agent uses a sliding window to manage conversation history:
- Keeps the last N screenshot+action pairs in context (default: 5)
- Older screenshots are dropped to fit within model limits

### Available Actions

| Action | Description |
|--------|-------------|
| `left_click` | Single left click |
| `double_click` | Double left click |
| `triple_click` | Triple click (select line) |
| `right_click` | Right click |
| `middle_click` | Middle mouse button |
| `mouse_move` | Move cursor without clicking |
| `left_click_drag` | Drag to destination |
| `type` | Type text (clears existing first) |
| `key` | Keyboard shortcut |
| `scroll` | Vertical scroll |
| `wait` | Wait N seconds |
| `answer` | Answer information queries |
| `terminate` | End the task |

## Troubleshooting

### "Cannot connect to Ollama"

Make sure Ollama is running:
```bash
ollama serve
```

### "Model not found"

Pull the model first:
```bash
ollama pull qwen3-vl:8b
```

### Browser doesn't open

Install Playwright browsers:
```bash
playwright install chromium
```

### Actions are clicking wrong locations

The model may need a clearer task description. Try being more specific:
- Instead of: "click the button"
- Try: "click the blue Submit button in the center of the page"

### Agent gets stuck

- Increase `--max-steps` if the task requires many actions
- Try `--highlight-mouse` to see where clicks are landing
- Use `--save-screenshots` to review what the model sees

## Limitations

- **Single tab only**: New tabs are automatically redirected to the main tab
- **No headless mode**: The browser must be visible (model needs to "see" the screen)
- **Local inference speed**: Qwen3-VL:8B on CPU can be slow. GPU recommended.
- **Fixed screen size**: 1440x900 viewport (not configurable in v1)

## Credits

- **Agent loop & Playwright harness**: Adapted from [Gemini Computer Use Preview](https://github.com/google-gemini/computer-use-preview)
- **Qwen3-VL inference pattern**: From [Qwen3-VL Cookbooks](https://github.com/QwenLM/Qwen3-VL)
