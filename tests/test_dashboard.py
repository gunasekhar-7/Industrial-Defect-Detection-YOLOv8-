"""Dashboard smoke test: the Streamlit script must run end to end.

Drives the real script through Streamlit's AppTest harness, so the page is
actually executed (widgets built, conveyor tab rendered) rather than merely
imported — a blank page or a raised exception fails the suite.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "ui" / "dashboard.py"

EXPECTED_TABS = [
    "🔬 Single Part Inspection",
    "🔄 Conveyor Simulation",
    "📊 Batch Quality Audit",
]


def _run_dashboard() -> AppTest:
    return AppTest.from_file(str(DASHBOARD), default_timeout=120).run()


def test_dashboard_renders_all_tabs_without_exception():
    at = _run_dashboard()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert [tab.label for tab in at.tabs] == EXPECTED_TABS


def test_dashboard_reports_no_errors_before_any_upload():
    at = _run_dashboard()
    assert not at.error, [e.value for e in at.error]
