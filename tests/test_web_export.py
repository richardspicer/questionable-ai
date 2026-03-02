"""Tests for transcript export (JSON and CSV)."""

from __future__ import annotations

import csv
import io
import json
from typing import Any


def _make_summary(**kwargs: Any) -> dict[str, Any]:
    """Build a transcript summary dict with defaults."""
    defaults: dict[str, Any] = {
        "id": "abc12345-full-uuid",
        "short_id": "abc12345",
        "date": "2026-02-28",
        "query": "Test query",
        "file": "2026-02-28_abc12345.json",
        "panel": "claude, gpt",
        "synthesizer": "claude",
        "tokens": 300,
        "cost": 0.05,
        "rounds": 2,
        "experiment_id": None,
    }
    defaults.update(kwargs)
    return defaults


class TestExportJson:
    """export_json produces valid JSON from transcript summaries."""

    def test_produces_valid_json(self) -> None:
        from mutual_dissent.web.components.export import export_json

        summaries = [_make_summary(), _make_summary(short_id="xyz98765")]
        result = export_json(summaries)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_preserves_fields(self) -> None:
        from mutual_dissent.web.components.export import export_json

        summaries = [_make_summary(query="My query", cost=0.123)]
        result = export_json(summaries)
        parsed = json.loads(result)
        assert parsed[0]["query"] == "My query"
        assert parsed[0]["cost"] == 0.123

    def test_empty_list(self) -> None:
        from mutual_dissent.web.components.export import export_json

        result = export_json([])
        assert json.loads(result) == []


class TestExportCsv:
    """export_csv produces valid CSV with header row."""

    def test_produces_valid_csv(self) -> None:
        from mutual_dissent.web.components.export import export_csv

        summaries = [_make_summary()]
        result = export_csv(summaries)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 2  # header + 1 data row

    def test_header_row(self) -> None:
        from mutual_dissent.web.components.export import export_csv

        summaries = [_make_summary()]
        result = export_csv(summaries)
        reader = csv.reader(io.StringIO(result))
        header = next(reader)
        assert "short_id" in header
        assert "query" in header
        assert "cost" in header

    def test_data_values(self) -> None:
        from mutual_dissent.web.components.export import export_csv

        summaries = [_make_summary(short_id="aaa11111", query="My Q", cost=0.05)]
        result = export_csv(summaries)
        reader = csv.reader(io.StringIO(result))
        _header = next(reader)
        row = next(reader)
        assert "aaa11111" in row
        assert "My Q" in row

    def test_empty_list(self) -> None:
        from mutual_dissent.web.components.export import export_csv

        result = export_csv([])
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 1  # header only
