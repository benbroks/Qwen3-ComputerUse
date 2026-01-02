"""Computer interfaces for browser automation."""

from .computer import Computer, EnvState
from .playwright import PlaywrightComputer

__all__ = ["Computer", "EnvState", "PlaywrightComputer"]
