"""Playwright implementation of the Computer interface."""

import os
import sys
import time

import playwright.sync_api
from playwright.sync_api import sync_playwright

from ..computer import Computer, EnvState


# Key name mapping for Playwright
PLAYWRIGHT_KEY_MAP = {
    "backspace": "Backspace",
    "tab": "Tab",
    "return": "Enter",
    "enter": "Enter",
    "shift": "Shift",
    "control": "Control",
    "ctrl": "Control",
    "alt": "Alt",
    "escape": "Escape",
    "esc": "Escape",
    "space": "Space",
    "pageup": "PageUp",
    "pagedown": "PageDown",
    "end": "End",
    "home": "Home",
    "left": "ArrowLeft",
    "up": "ArrowUp",
    "right": "ArrowRight",
    "down": "ArrowDown",
    "insert": "Insert",
    "delete": "Delete",
    "f1": "F1",
    "f2": "F2",
    "f3": "F3",
    "f4": "F4",
    "f5": "F5",
    "f6": "F6",
    "f7": "F7",
    "f8": "F8",
    "f9": "F9",
    "f10": "F10",
    "f11": "F11",
    "f12": "F12",
    "command": "Meta",
    "cmd": "Meta",
    "meta": "Meta",
    "win": "Meta",
    "windows": "Meta",
}


class PlaywrightComputer(Computer):
    """Browser automation using Playwright."""

    def __init__(
        self,
        screen_size: tuple[int, int],
        initial_url: str = "https://www.google.com",
        highlight_mouse: bool = False,
    ):
        self._initial_url = initial_url
        self._screen_size = screen_size
        self._highlight_mouse = highlight_mouse
        self._last_mouse_pos = (0, 0)

    def _handle_new_page(self, new_page: playwright.sync_api.Page):
        """Force single-tab behavior by redirecting new tabs."""
        new_url = new_page.url
        new_page.close()
        self._page.goto(new_url)

    def __enter__(self):
        print("Launching browser...")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            args=[
                "--disable-extensions",
                "--disable-file-system",
                "--disable-plugins",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
                "--disable-default-apps",
                "--disable-sync",
            ],
            headless=bool(os.environ.get("PLAYWRIGHT_HEADLESS", False)),
        )
        self._context = self._browser.new_context(
            viewport={
                "width": self._screen_size[0],
                "height": self._screen_size[1],
            }
        )
        self._page = self._context.new_page()
        self._page.goto(self._initial_url)
        self._context.on("page", self._handle_new_page)

        print(f"Browser ready at: {self._initial_url}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            self._context.close()
        try:
            self._browser.close()
        except Exception as e:
            if "Connection closed" not in str(e):
                raise
        self._playwright.stop()

    def screen_size(self) -> tuple[int, int]:
        viewport = self._page.viewport_size
        if viewport:
            return viewport["width"], viewport["height"]
        return self._screen_size

    def current_state(self) -> EnvState:
        self._page.wait_for_load_state()
        time.sleep(0.5)  # Allow rendering to complete
        screenshot = self._page.screenshot(type="png", full_page=False)
        return EnvState(screenshot=screenshot, url=self._page.url)

    # Click actions
    def click_at(self, x: int, y: int) -> EnvState:
        self._show_cursor(x, y)
        self._page.mouse.click(x, y)
        self._last_mouse_pos = (x, y)
        self._page.wait_for_load_state()
        return self.current_state()

    def double_click_at(self, x: int, y: int) -> EnvState:
        self._show_cursor(x, y)
        self._page.mouse.dblclick(x, y)
        self._last_mouse_pos = (x, y)
        self._page.wait_for_load_state()
        return self.current_state()

    def triple_click_at(self, x: int, y: int) -> EnvState:
        self._show_cursor(x, y)
        self._page.mouse.click(x, y, click_count=3)
        self._last_mouse_pos = (x, y)
        self._page.wait_for_load_state()
        return self.current_state()

    def right_click_at(self, x: int, y: int) -> EnvState:
        self._show_cursor(x, y)
        self._page.mouse.click(x, y, button="right")
        self._last_mouse_pos = (x, y)
        self._page.wait_for_load_state()
        return self.current_state()

    def middle_click_at(self, x: int, y: int) -> EnvState:
        self._show_cursor(x, y)
        self._page.mouse.click(x, y, button="middle")
        self._last_mouse_pos = (x, y)
        self._page.wait_for_load_state()
        return self.current_state()

    # Mouse movement
    def hover_at(self, x: int, y: int) -> EnvState:
        self._show_cursor(x, y)
        self._page.mouse.move(x, y)
        self._last_mouse_pos = (x, y)
        self._page.wait_for_load_state()
        return self.current_state()

    def drag_to(self, x: int, y: int) -> EnvState:
        """Drag from current mouse position to (x, y)."""
        start_x, start_y = self._last_mouse_pos
        self._show_cursor(start_x, start_y)

        self._page.mouse.move(start_x, start_y)
        self._page.mouse.down()
        self._page.wait_for_load_state()

        self._show_cursor(x, y)
        self._page.mouse.move(x, y)
        self._page.mouse.up()
        self._last_mouse_pos = (x, y)
        return self.current_state()

    # Keyboard actions
    def type_text(self, text: str) -> EnvState:
        """Type text, clearing existing content first."""
        # Select all and delete
        if sys.platform == "darwin":
            self.key_combination(["Meta", "a"])
        else:
            self.key_combination(["Control", "a"])
        self.key_combination(["Delete"])

        # Type new text
        self._page.keyboard.type(text)
        self._page.wait_for_load_state()
        return self.current_state()

    def key_combination(self, keys: list[str]) -> EnvState:
        """Press key combination."""
        # Normalize key names
        normalized = [PLAYWRIGHT_KEY_MAP.get(k.lower(), k) for k in keys]

        # Hold modifier keys, press last key, release modifiers
        for key in normalized[:-1]:
            self._page.keyboard.down(key)

        self._page.keyboard.press(normalized[-1])

        for key in reversed(normalized[:-1]):
            self._page.keyboard.up(key)

        return self.current_state()

    # Scrolling
    def scroll(self, pixels: int) -> EnvState:
        """Scroll vertically. Positive=up, negative=down."""
        # Playwright's wheel uses positive=down convention, so we negate
        self._page.mouse.wheel(0, -pixels)
        self._page.wait_for_load_state()
        return self.current_state()

    # Visual feedback
    def _show_cursor(self, x: int, y: int):
        """Show cursor indicator if enabled."""
        if not self._highlight_mouse:
            return
        self._page.evaluate(
            f"""
            () => {{
                const div = document.createElement('div');
                div.style.pointerEvents = 'none';
                div.style.border = '4px solid red';
                div.style.borderRadius = '50%';
                div.style.width = '20px';
                div.style.height = '20px';
                div.style.position = 'fixed';
                div.style.zIndex = '9999';
                div.style.left = {x} - 10 + 'px';
                div.style.top = {y} - 10 + 'px';
                document.body.appendChild(div);
                setTimeout(() => div.remove(), 2000);
            }}
        """
        )
        time.sleep(0.5)
