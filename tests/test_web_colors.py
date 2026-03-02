"""Tests for web color mapping."""

from __future__ import annotations


class TestModelCssColors:
    """MODEL_CSS_COLORS maps aliases to Tailwind classes."""

    def test_claude_has_border_text_bg(self) -> None:
        """claude entry has border, text, and bg keys."""
        from mutual_dissent.web.colors import MODEL_CSS_COLORS

        assert "claude" in MODEL_CSS_COLORS
        assert "border" in MODEL_CSS_COLORS["claude"]
        assert "text" in MODEL_CSS_COLORS["claude"]
        assert "bg" in MODEL_CSS_COLORS["claude"]

    def test_all_four_models_present(self) -> None:
        """All four default models have entries."""
        from mutual_dissent.web.colors import MODEL_CSS_COLORS

        for alias in ("claude", "gpt", "gemini", "grok"):
            assert alias in MODEL_CSS_COLORS

    def test_get_css_colors_known_alias(self) -> None:
        """get_css_colors returns correct dict for known alias."""
        from mutual_dissent.web.colors import get_css_colors

        colors = get_css_colors("claude")
        assert "fuchsia" in colors["border"]

    def test_get_css_colors_unknown_alias(self) -> None:
        """get_css_colors returns default gray for unknown alias."""
        from mutual_dissent.web.colors import get_css_colors

        colors = get_css_colors("unknown_model")
        assert "gray" in colors["border"]
