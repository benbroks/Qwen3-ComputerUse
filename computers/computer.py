"""Abstract Computer interface for browser automation."""

import abc
from typing import Literal

import pydantic


class EnvState(pydantic.BaseModel):
    """State returned after each action."""
    screenshot: bytes  # PNG format
    url: str


class Computer(abc.ABC):
    """Abstract interface for browser automation environments."""

    @abc.abstractmethod
    def screen_size(self) -> tuple[int, int]:
        """Returns (width, height) of the browser viewport."""

    @abc.abstractmethod
    def current_state(self) -> EnvState:
        """Returns current screenshot and URL."""

    # Click actions
    @abc.abstractmethod
    def click_at(self, x: int, y: int) -> EnvState:
        """Single left click at (x, y)."""

    @abc.abstractmethod
    def double_click_at(self, x: int, y: int) -> EnvState:
        """Double left click at (x, y)."""

    @abc.abstractmethod
    def triple_click_at(self, x: int, y: int) -> EnvState:
        """Triple click at (x, y) - typically selects a line."""

    @abc.abstractmethod
    def right_click_at(self, x: int, y: int) -> EnvState:
        """Right click at (x, y)."""

    @abc.abstractmethod
    def middle_click_at(self, x: int, y: int) -> EnvState:
        """Middle mouse button click at (x, y)."""

    # Mouse movement
    @abc.abstractmethod
    def hover_at(self, x: int, y: int) -> EnvState:
        """Move mouse to (x, y) without clicking."""

    @abc.abstractmethod
    def drag_to(self, x: int, y: int) -> EnvState:
        """Drag from current position to (x, y)."""

    # Keyboard actions
    @abc.abstractmethod
    def type_text(self, text: str) -> EnvState:
        """Type text (clears existing content first)."""

    @abc.abstractmethod
    def key_combination(self, keys: list[str]) -> EnvState:
        """Press key combination (e.g., ["Control", "c"])."""

    # Scrolling
    @abc.abstractmethod
    def scroll(self, pixels: int) -> EnvState:
        """Scroll vertically. Positive=up, negative=down."""
