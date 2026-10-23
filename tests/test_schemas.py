import pytest
from pydantic import ValidationError

from basal_station.api.schemas import DecideRequest, DownloadRequest, QuestionSpec
from basal_station.config import AppSettings


def test_decide_request_minimal():
    req = DecideRequest(
        state="hello", questions={"verdict": {"type": "noul", "instructions": "ok?"}}
    )
    assert req.questions["verdict"].type == "noul"
    assert req.knowledge is None


def test_decide_request_defaults_choice():
    req = DecideRequest(state="s", questions={"q": {}})
    assert req.questions["q"].type == "choice"
    assert req.questions["q"].criteria == {}


def test_decide_request_rejects_bad_type():
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={"q": {"type": "essay"}})


def test_decide_request_requires_questions():
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={})


def test_decide_request_extra_fields_allowed():
    req = DecideRequest.model_validate(
        {"state": "s", "questions": {"q": {"type": "score", "custom": 1}}, "media": [{"type": "image"}]}
    )
    assert req.media == [{"type": "image"}]


def test_question_spec_direct():
    spec = QuestionSpec(type="choice", instructions="pick", criteria={"a": "alpha"}, max_error=0.05)
    assert spec.max_error == 0.05


def test_download_request_valid_format():
    assert DownloadRequest(model="omni-1.0-max", format="gguf", quant="Q4_K_M").quant == "Q4_K_M"
    with pytest.raises(ValidationError):
        DownloadRequest(model="m", format="torrent")


def test_settings_defaults_and_env():
    s = AppSettings()
    assert s.engine.port == 8000
    assert s.knowledge.command == ["kp", "mcp"]
    assert s.default_model == "omni-1.0-max"
