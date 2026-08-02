import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def summary_sample() -> dict:
    return json.loads((FIXTURES / "summary_sample.json").read_text())


@pytest.fixture
def transcript_sample() -> dict:
    return json.loads((FIXTURES / "transcript_sample.json").read_text())


@pytest.fixture
def note_sample() -> str:
    return (FIXTURES / "note_sample.md").read_text()
