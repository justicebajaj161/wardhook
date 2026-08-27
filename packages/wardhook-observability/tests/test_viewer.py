"""The viewer is self-contained, escapes everything, and the CLI works."""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest
from typer.testing import CliRunner

from wardhook.observability import (
    JSONLTraceStore,
    TokenUsage,
    Trace,
    TraceStep,
    render_html,
)
from wardhook.observability.viewer.cli import app

runner = CliRunner()


class TestSelfContainment:
    def test_the_page_is_a_complete_document(self, sample_trace):
        page = render_html(sample_trace)
        assert page.startswith("<!doctype html>")
        assert page.rstrip().endswith("</html>")
        assert "<style>" in page and "<script>" in page

    def test_no_external_resource_is_referenced(self, sample_trace):
        # This is the "opens anywhere, offline, forever" promise, mechanised.
        page = render_html(sample_trace)
        external = re.findall(r'(?:src|href)\s*=\s*["\'](?!#)([^"\']+)', page)
        assert external == [], f"page references external resources: {external}"
        for scheme in ("http://", "https://", "//cdn", "@import"):
            assert scheme not in page

    def test_the_pricing_vintage_is_stated_on_the_page(self, sample_trace):
        from wardhook.observability import PRICES_AS_OF

        # A cost estimate with no date on it invites someone to trust a stale one.
        assert PRICES_AS_OF in render_html(sample_trace)


class _Inspector(HTMLParser):
    """Collects the tags and attribute values a browser would actually see.

    Substring assertions on raw HTML are a weak test of escaping: escaped text
    legitimately *contains* the dangerous characters. Parsing the document and
    asking what elements and attributes it really declares is the property that
    matters.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.attr_names: list[str] = []
        self.script_text: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self._in_script = tag == "script"
        for name, _ in attrs:
            self.attr_names.append(name.lower())

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script:
            self.script_text.append(data)


class TestEscaping:
    @pytest.mark.parametrize(
        "payload",
        [
            "<script>alert(1)</script>",
            '"><img src=x onerror=alert(1)>',
            "</style><script>alert(1)</script>",
        ],
    )
    def test_hostile_content_never_becomes_markup(self, payload):
        # Trace data carries user text. If a rendered trace is ever served over
        # HTTP, unescaped content is stored XSS -- so escaping is a security
        # control here, not formatting.
        trace = Trace(
            run_id=payload,
            started_at=payload,
            error=payload,
            metadata={payload: payload},
            steps=(
                TraceStep(
                    node=payload,
                    run_id=payload,
                    started_at=payload,
                    latency_ms=1.0,
                    model=payload,
                    error=payload,
                ),
            ),
        )
        page = render_html(trace, title=payload)

        inspector = _Inspector()
        inspector.feed(page)

        # The payload tried to open <script>, <img> and </style>. None of them
        # may exist as real elements, and the only script in the document must
        # be the viewer's own.
        assert inspector.tags.count("script") == 1
        assert "img" not in inspector.tags
        assert "alert(1)" not in "".join(inspector.script_text)
        # Nor may it have introduced an attribute -- an event handler above all.
        # (A `title="&lt;img ...&gt;"` VALUE containing the text is fine and is
        # exactly what escaping produces; a new `onerror` NAME would not be.)
        handlers = [name for name in inspector.attr_names if name.startswith("on")]
        assert handlers == [], f"payload injected event handlers: {handlers}"
        # And the raw payload must not survive verbatim anywhere.
        assert payload not in page

    def test_the_only_script_tag_is_our_own(self, sample_trace):
        for trace in (Trace(run_id="<script>alert(1)</script>"), sample_trace):
            inspector = _Inspector()
            inspector.feed(render_html(trace))
            assert inspector.tags.count("script") == 1
            assert "addEventListener" in "".join(inspector.script_text)


class TestRendering:
    def test_steps_and_totals_appear(self, sample_trace):
        page = render_html(sample_trace)
        assert "retrieve" in page and "call_model" in page
        assert "claude-opus-5" in page
        assert "900" in page and "120" in page

    def test_a_failed_step_is_marked(self):
        trace = Trace(
            run_id="r1",
            error="ValueError: boom",
            steps=(TraceStep("tools", "r1", "t", 5.0, error="ValueError: boom"),),
        )
        page = render_html(trace)
        assert 'class="failed"' in page
        assert "Run failed" in page

    def test_an_empty_trace_file_says_so(self):
        assert "This trace file is empty" in render_html([])

    def test_a_trace_with_no_steps_still_renders(self):
        assert "no steps recorded" in render_html(Trace("r1"))

    def test_multiple_traces_are_summarised_together(self, sample_trace):
        page = render_html([sample_trace, sample_trace])
        assert "2 runs" in page
        assert page.count("<details") == 2

    def test_sub_cent_costs_stay_visible(self):
        # Rounding per-node costs to 2dp renders an entire trace as $0.00.
        trace = Trace(
            "r1",
            (
                TraceStep(
                    "call_model",
                    "r1",
                    "t",
                    1.0,
                    TokenUsage(input_tokens=10),
                    cost=0.00004,
                ),
            ),
        )
        assert "$0.00004" in render_html(trace)


class TestCli:
    def _file(self, tmp_path, *traces):
        store = JSONLTraceStore(tmp_path / "traces.jsonl")
        for trace in traces:
            store.append(trace)
        return store.path

    def test_view_writes_a_page(self, tmp_path, sample_trace):
        source = self._file(tmp_path, sample_trace)
        output = tmp_path / "out" / "trace.html"

        result = runner.invoke(app, ["view", str(source), "-o", str(output)])

        assert result.exit_code == 0, result.output
        assert "no external requests" in result.output
        assert output.read_text(encoding="utf-8").startswith("<!doctype html>")

    def test_view_can_select_one_run(self, tmp_path, sample_trace):
        other = Trace(run_id="run-2")
        source = self._file(tmp_path, sample_trace, other)
        output = tmp_path / "trace.html"

        result = runner.invoke(app, ["view", str(source), "-o", str(output), "--run-id", "run-2"])

        assert result.exit_code == 0, result.output
        assert "1 run" in result.output
        assert "run-2" in output.read_text(encoding="utf-8")

    def test_view_rejects_an_unknown_run_and_lists_what_is_there(self, tmp_path, sample_trace):
        source = self._file(tmp_path, sample_trace)
        result = runner.invoke(app, ["view", str(source), "--run-id", "nope"])
        assert result.exit_code != 0
        assert "run-1" in result.output

    def test_view_reports_a_missing_file_clearly(self, tmp_path):
        result = runner.invoke(app, ["view", str(tmp_path / "absent.jsonl")])
        assert result.exit_code != 0
        assert "No trace file" in result.output

    def test_view_reports_a_corrupt_file_with_its_line_number(self, tmp_path):
        source = tmp_path / "bad.jsonl"
        source.write_text("{nope\n", encoding="utf-8")
        result = runner.invoke(app, ["view", str(source)])
        assert result.exit_code != 0
        assert "bad.jsonl:1" in result.output

    def test_summary_prints_a_per_node_breakdown(self, tmp_path, sample_trace):
        source = self._file(tmp_path, sample_trace)
        result = runner.invoke(app, ["summary", str(source)])

        assert result.exit_code == 0, result.output
        assert "call_model" in result.output
        assert "TOTAL" in result.output

    def test_summary_totals_multiple_runs(self, tmp_path, sample_trace):
        source = self._file(tmp_path, sample_trace, sample_trace)
        result = runner.invoke(app, ["summary", str(source)])
        assert "2 runs" in result.output

    def test_summary_of_an_empty_file(self, tmp_path):
        source = tmp_path / "empty.jsonl"
        source.write_text("", encoding="utf-8")
        result = runner.invoke(app, ["summary", str(source)])
        assert result.exit_code == 0
        assert "no runs" in result.output

    def test_the_console_script_entry_point_exists(self):
        from wardhook.observability.viewer.cli import main

        assert callable(main)

    def test_no_arguments_shows_help(self):
        result = runner.invoke(app, [])
        assert "Inspect Wardhook trace files" in result.output
