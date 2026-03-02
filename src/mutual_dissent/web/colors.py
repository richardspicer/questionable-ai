"""Shared CSS color constants for web UI model styling.

Maps model aliases to Tailwind CSS classes for consistent color coding
across the debate view and dashboard. Mirrors MODEL_COLORS from display.py
but uses Tailwind class names instead of Rich color names.
"""

from __future__ import annotations

# Rich name → Tailwind CSS class mapping.
# claude=magenta→fuchsia, gpt=green, gemini=cyan, grok=yellow.
MODEL_CSS_COLORS: dict[str, dict[str, str]] = {
    "claude": {
        "border": "border-fuchsia-500",
        "text": "text-fuchsia-400",
        "bg": "bg-fuchsia-500/10",
    },
    "gpt": {
        "border": "border-green-500",
        "text": "text-green-400",
        "bg": "bg-green-500/10",
    },
    "gemini": {
        "border": "border-cyan-500",
        "text": "text-cyan-400",
        "bg": "bg-cyan-500/10",
    },
    "grok": {
        "border": "border-yellow-500",
        "text": "text-yellow-400",
        "bg": "bg-yellow-500/10",
    },
}

_DEFAULT_CSS_COLORS: dict[str, str] = {
    "border": "border-gray-500",
    "text": "text-gray-400",
    "bg": "bg-gray-500/10",
}


def get_css_colors(alias: str) -> dict[str, str]:
    """Get Tailwind CSS color classes for a model alias.

    Args:
        alias: Model short name (e.g. "claude", "gpt").

    Returns:
        Dict with "border", "text", and "bg" Tailwind class strings.
    """
    return MODEL_CSS_COLORS.get(alias.lower(), _DEFAULT_CSS_COLORS)
