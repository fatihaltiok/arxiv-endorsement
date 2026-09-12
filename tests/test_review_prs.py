"""Tests for the pull-request handling in review_prs.py (no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review_prs = _load(ROOT / "scripts" / "review_prs.py", "review_prs")

WRONG_SUBJECT = "LinkedIn: https://www.linkedin.com/in/someone\nPaper: https://example.org/p.pdf\nRepo: https://github.com/someone/repo\nSubject: cs.AI\nEndorsementCode: A1B2C3\n"
MALFORMED = "LinkedIn: https://www.linkedin.com/in/someone\nSubject: cs.SE\n"

PR = {"number": 42, "files": [{"path": "requests/someone.txt"}], "headRefOid": "deadbeef"}


@pytest.fixture
def stubbed(monkeypatch):
    calls: dict[str, list] = {"comments": [], "closed": [], "commented_markers": []}
    monkeypatch.setattr(review_prs, "post_comment", lambda repo, n, body: calls["comments"].append((n, body)))
    monkeypatch.setattr(review_prs, "already_commented", lambda repo, n, marker=review_prs.COMMENT_MARKER: False)

    def fake_close(repo, n, body):
        calls["comments"].append((n, body))
        calls["closed"].append(n)

    monkeypatch.setattr(review_prs, "close_pr", fake_close)
    return calls


def test_wrong_subject_pr_is_closed_with_the_automated_reply(stubbed, monkeypatch):
    monkeypatch.setattr(review_prs, "fetch_file_content", lambda repo, path, ref: WRONG_SUBJECT)
    review_prs.process_pr("owner/repo", PR, dry_run=False, update=False)
    assert stubbed["closed"] == [42]
    (_, body), = stubbed["comments"]
    assert review_prs.WRONG_SUBJECT_MESSAGE in body
    assert review_prs.WRONG_SUBJECT_MARKER in body


def test_wrong_subject_pr_is_not_closed_in_dry_run(stubbed, monkeypatch):
    monkeypatch.setattr(review_prs, "fetch_file_content", lambda repo, path, ref: WRONG_SUBJECT)
    review_prs.process_pr("owner/repo", PR, dry_run=True, update=False)
    assert stubbed["closed"] == []
    assert stubbed["comments"] == []


def test_wrong_subject_pr_is_answered_only_once(stubbed, monkeypatch):
    monkeypatch.setattr(review_prs, "fetch_file_content", lambda repo, path, ref: WRONG_SUBJECT)
    monkeypatch.setattr(
        review_prs, "already_commented",
        lambda repo, n, marker=review_prs.COMMENT_MARKER: marker == review_prs.WRONG_SUBJECT_MARKER,
    )
    review_prs.process_pr("owner/repo", PR, dry_run=False, update=False)
    assert stubbed["closed"] == []
    assert stubbed["comments"] == []


def test_completion_prefers_sonnet_and_falls_back(monkeypatch, tmp_path):
    sonnet = tmp_path / "claude-sonnet-5-completions.py"
    fallback = tmp_path / "best-effort-completions.py"
    for p in (sonnet, fallback):
        p.write_text("")
    monkeypatch.setattr(review_prs, "COMPLETION_BACKENDS", (sonnet, fallback))

    tried: list[str] = []

    class Proc:
        def __init__(self, code, out):
            self.returncode, self.stdout, self.stderr = code, out, ""

    def fake_run(cmd, **kwargs):
        name = Path(cmd[1]).name
        tried.append(name)
        if name.startswith("claude-sonnet"):
            return Proc(1, "")  # e.g. 429 out of credits
        return Proc(0, '{"model": "claude-haiku-4-5", "choices": []}')

    monkeypatch.setattr(review_prs.subprocess, "run", fake_run)
    response = review_prs.best_effort_completion({"messages": []})
    assert tried == ["claude-sonnet-5-completions.py", "best-effort-completions.py"]
    assert response["model"] == "claude-haiku-4-5"
    assert "claude-haiku-4-5" in review_prs._MODELS_USED


def test_completion_raises_when_every_backend_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(review_prs, "COMPLETION_BACKENDS", (tmp_path / "missing.py",))
    with pytest.raises(RuntimeError, match="no completion backend succeeded"):
        review_prs.best_effort_completion({"messages": []})


def test_format_messages_flattens_roles_and_content():
    payload = {
        "messages": [
            {"role": "system", "content": "be strict"},
            {"role": "user", "content": "evaluate this"},
        ]
    }
    assert review_prs.format_messages(payload["messages"]) == "SYSTEM:\nbe strict\n\nUSER:\nevaluate this"


def test_agent_completion_wraps_stdout_as_openai_response(monkeypatch, tmp_path):
    script = tmp_path / "agent-kimi-k3.py"
    script.write_text("")
    seen: dict = {}

    class Proc:
        returncode = 0
        stdout = '{"overall_verdict": true}\n'
        stderr = ""

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["input"] = kwargs["input"]
        return Proc()

    monkeypatch.setattr(review_prs.subprocess, "run", fake_run)
    review_prs._MODELS_USED.clear()
    response = review_prs.agent_completion(
        {"messages": [{"role": "user", "content": "hi"}]}, script, "kimi-k3 (agentknit)"
    )
    assert seen["cmd"] == [review_prs.sys.executable, str(script), "--non-interactive"]
    assert "USER:\nhi" in seen["input"]
    assert "ONLY the requested JSON" in seen["input"]
    assert response["choices"][0]["message"]["content"] == '{"overall_verdict": true}'
    assert review_prs._LAST_MODEL == "kimi-k3 (agentknit)"
    assert "kimi-k3 (agentknit)" in review_prs._MODELS_USED


def test_agent_completion_tolerates_agentknit_console_noise(monkeypatch, tmp_path):
    script = tmp_path / "agent-kimi-k3.py"
    script.write_text("")

    class Proc:
        returncode = 0
        stdout = (
            "\x1b[2m\x1b[35m[budget] 1,047,136/1,048,576 tokens remaining\x1b[0m\n"
            "\x1b[2m\x1b[35m[tokens] prompt 1,440\x1b[0m\n\n"
            "I'll evaluate the paper now.\n"
            '{"overall_verdict": true}\n'
            "\x1b[2m\x1b[35m[session tokens] prompt 1,440\x1b[0m\n"
        )
        stderr = ""

    monkeypatch.setattr(review_prs.subprocess, "run", lambda *a, **k: Proc())
    response = review_prs.agent_completion(
        {"messages": [{"role": "user", "content": "hi"}]}, script, "kimi-k3 (agentknit)"
    )
    parsed = review_prs.check_paper.parse_evaluation_response(response)
    assert parsed == {"overall_verdict": True}


def test_agent_completion_raises_when_no_json_in_output(monkeypatch, tmp_path):
    script = tmp_path / "agent-kimi-k3.py"
    script.write_text("")

    class Proc:
        returncode = 0
        stdout = "The agent chatted but produced no JSON.\n"
        stderr = ""

    monkeypatch.setattr(review_prs.subprocess, "run", lambda *a, **k: Proc())
    with pytest.raises(RuntimeError, match="could not parse JSON"):
        review_prs.agent_completion({"messages": []}, script, "kimi-k3 (agentknit)")


def test_agent_completion_raises_on_nonzero_exit(monkeypatch, tmp_path):
    script = tmp_path / "agent-kimi-k3.py"
    script.write_text("")

    class Proc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(review_prs.subprocess, "run", lambda *a, **k: Proc())
    with pytest.raises(RuntimeError, match="boom"):
        review_prs.agent_completion({"messages": []}, script, "kimi-k3 (agentknit)")


def test_agent_completion_raises_when_script_missing(tmp_path):
    with pytest.raises(RuntimeError, match="agent script not found"):
        review_prs.agent_completion({"messages": []}, tmp_path / "nope.py", "x")


def test_make_completion_defaults_to_claude_backends():
    assert review_prs.make_completion("claude") is review_prs.best_effort_completion


def test_make_completion_returns_agent_lambda():
    complete = review_prs.make_completion("kimi-k3")
    assert complete is not review_prs.best_effort_completion
    assert callable(complete)


def test_other_validation_errors_leave_the_pr_open(stubbed, monkeypatch):
    monkeypatch.setattr(review_prs, "fetch_file_content", lambda repo, path, ref: MALFORMED)
    review_prs.process_pr("owner/repo", PR, dry_run=False, update=False)
    assert stubbed["closed"] == []
    assert stubbed["comments"] == []
