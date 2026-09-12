#!/usr/bin/env python3
"""Run check-paper.py against each valid endorsement request PR and post the result as a PR comment.

A request PR is "valid" when its requests/*.txt file passes scripts/validate_request_files.py.
Skips PRs that already carry a check comment (idempotent). A request for a subject other
than cs.SE is answered and closed.

Usage:
  ./scripts/review_prs.py [--repo owner/repo] [--dry-run] [--pr N] [--agent claude|kimi-k3]
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "monperrus/arxiv-endorsement"
COMMENT_MARKER = "<!-- arxiv-endorsement-check -->"
WRONG_SUBJECT_MARKER = "<!-- arxiv-endorsement-wrong-subject -->"
WRONG_SUBJECT_MESSAGE = "[automated reply] we only endorse for cs.SE"
# Sonnet 5 first: best-effort-completions.py starts with Haiku, whose gate verdicts
# proved unstable across identical runs (same PDF checksum, opposite overall verdict).
# best-effort stays as the fallback for when the Sonnet endpoint is out of credits.
COMPLETION_BACKENDS = (
    Path.home() / "bin" / "claude-sonnet-5-completions.py",
    Path.home() / "bin" / "best-effort-completions.py",
)
# Agent launchers (agentknit one-shot tasks on stdin). Each maps to a model name
# reported in the posted check comment.
AGENT_SCRIPTS = {
    "kimi-k3": Path.home() / "bin" / "agent-kimi-k3.py",
}
AGENT_MODEL_NAMES = {
    "kimi-k3": "kimi-k3 (agentknit)",
}
AGENT_CHOICES = ("claude", *AGENT_SCRIPTS)
DEFAULT_AGENT = "claude"
# Agents reason before answering and may be slow on a large paper.
AGENT_TIMEOUT_SECONDS = 1800


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_paper = load_module(ROOT / "check-paper.py", "check_paper")
check_repo = load_module(ROOT / "scripts" / "check_repo.py", "check_repo")
validate_request_files = load_module(ROOT / "scripts" / "validate_request_files.py", "validate_request_files")


def gh_json(*args: str):
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


def list_open_prs(repo: str) -> list[dict]:
    return gh_json(
        "pr", "list", "--repo", repo, "--state", "open",
        "--json", "number,url,files,headRefOid",
    )


def fetch_file_content(repo: str, path: str, ref: str) -> str:
    proc = subprocess.run(
        ["gh", "api", f"repos/{repo}/contents/{path}?ref={ref}", "--jq", ".content"],
        capture_output=True, text=True, check=True,
    )
    return base64.b64decode(proc.stdout.strip()).decode("utf-8")


def already_commented(repo: str, pr_number: int, marker: str = COMMENT_MARKER) -> bool:
    comments = gh_json("pr", "view", str(pr_number), "--repo", repo, "--json", "comments")["comments"]
    return any(marker in c.get("body", "") for c in comments)


def close_pr(repo: str, pr_number: int, body: str) -> None:
    post_comment(repo, pr_number, body)
    subprocess.run(["gh", "pr", "close", str(pr_number), "--repo", repo], check=True)


def parse_fields(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        name, value = line.split(":", 1)
        fields[name.strip()] = value.strip()
    return fields


_LAST_MODEL = "unknown"
# Every model that answered while reviewing the current PR; a gate that fell back to
# another backend must be visible in the posted comment.
_MODELS_USED: list[str] = []


def format_messages(messages: list[dict]) -> str:
    """Flatten OpenAI-style messages into a single prompt string."""
    return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)


def agent_completion(payload: dict, agent_script: Path, model_name: str) -> dict:
    """Complete a payload through an agentknit agent (task on stdin, JSON on stdout)."""
    global _LAST_MODEL
    if not agent_script.exists():
        raise RuntimeError(f"agent script not found: {agent_script}")
    proc = subprocess.run(
        [sys.executable, str(agent_script), "--non-interactive"],
        input=(
            format_messages(payload["messages"])
            + "\n\nRespond with ONLY the requested JSON object, no markdown fences, no extra text."
        ),
        capture_output=True,
        text=True,
        timeout=AGENT_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{agent_script.name}: {(proc.stderr or proc.stdout).strip()[:500]}")
    _LAST_MODEL = model_name
    if _LAST_MODEL not in _MODELS_USED:
        _MODELS_USED.append(_LAST_MODEL)
    return {"choices": [{"message": {"role": "assistant", "content": proc.stdout.strip()}}]}


def make_completion(agent: str):
    """Return the completion function for the chosen agent."""
    if agent == DEFAULT_AGENT:
        return best_effort_completion
    script = AGENT_SCRIPTS[agent]
    model_name = AGENT_MODEL_NAMES[agent]
    return lambda payload: agent_completion(payload, script, model_name)


def best_effort_completion(payload: dict) -> dict:
    """Complete a payload with the first backend that answers, Sonnet 5 first."""
    global _LAST_MODEL
    errors = []
    for backend in COMPLETION_BACKENDS:
        if not backend.exists():
            errors.append(f"{backend.name}: not found")
            continue
        proc = subprocess.run(
            [sys.executable, str(backend)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            errors.append(f"{backend.name}: {(proc.stderr or proc.stdout).strip()[:200]}")
            print(f"  {backend.name} failed, trying next backend", file=sys.stderr)
            continue
        response = json.loads(proc.stdout)
        _LAST_MODEL = response.get("best_effort", {}).get("model") or response.get("model") or _LAST_MODEL
        if _LAST_MODEL not in _MODELS_USED:
            _MODELS_USED.append(_LAST_MODEL)
        return response
    raise RuntimeError("no completion backend succeeded: " + "; ".join(errors))


def evaluate_paper(text: str, complete=best_effort_completion) -> tuple[dict, str]:
    """Evaluate the paper gates and report which model actually answered."""
    response = complete(check_paper.build_evaluation_payload(text))
    return check_paper.parse_evaluation_response(response), _LAST_MODEL


def build_comment(
    result: dict, paper_url: str, repo_url: str, model: str, checksum: str,
    repo_result: dict | None = None,
) -> str:
    gates = check_paper.GATES

    def verdict_md(v: bool) -> str:
        return "✓ PASS" if v else "✗ FAIL"

    def conf_md(c: str) -> str:
        return {"high": "", "medium": " *(medium confidence)*", "low": " *(low confidence)*"}.get(c, "")

    overall = result.get("overall_verdict", all(result[k]["verdict"] for k, _ in gates))
    if repo_result is not None:
        overall = overall and repo_result["verdict"]
    overall_str = "✓ SUITABLE for arXiv cs.SE endorsement" if overall else "✗ NOT suitable for arXiv cs.SE endorsement"

    lines = [
        COMMENT_MARKER,
        "## arXiv SE endorsement check (automated)",
        "",
        f"Paper: {paper_url}",
        f"Repo: {repo_url}",
        f"Model: {model}",
        f"SHA-256: `{checksum}`",
        "",
        f"**Overall: {overall_str}**",
        "",
    ]
    for key, label in gates:
        g = result[key]
        lines.append(f"- {verdict_md(g['verdict'])} {label}{conf_md(g['confidence'])}")
        if not g["verdict"] and g.get("feedback"):
            lines.append(f"  - {g['feedback']}")

    if repo_result is not None:
        lines.append(
            f"- {verdict_md(repo_result['verdict'])} {check_paper.GATE4_LABEL}"
            f"{conf_md(repo_result.get('confidence', ''))}"
        )
        if repo_result.get("n_claims"):
            lines.append(
                f"  - {repo_result['n_backed']}/{repo_result['n_claims']} empirical numbers backed by a "
                f"data file or a script ({repo_result['traceability']:.0%})"
            )
        lines += ["", "<details><summary>Gate 4 detail — traceability of the paper's numbers</summary>", ""]
        lines.append(check_repo.render_gate4(repo_result))
        lines += ["</details>", ""]

    lines += ["", result.get("summary", "")]
    if repo_result is not None and repo_result.get("summary"):
        lines += ["", repo_result["summary"]]
    return "\n".join(lines)


def post_comment(repo: str, pr_number: int, body: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        body_path = f.name
    try:
        subprocess.run(
            ["gh", "pr", "comment", str(pr_number), "--repo", repo, "--body-file", body_path],
            check=True,
        )
    finally:
        Path(body_path).unlink(missing_ok=True)


def process_pr(
    repo: str, pr: dict, dry_run: bool, update: bool,
    skip_repo: bool = False, threshold: float = check_repo.DEFAULT_TRACEABILITY_THRESHOLD,
    complete=best_effort_completion,
) -> None:
    number = pr["number"]
    txt_files = [f["path"] for f in pr["files"] if f["path"].startswith("requests/") and f["path"].endswith(".txt")]
    if not txt_files:
        return

    was_commented = already_commented(repo, number)
    if was_commented and not update:
        print(f"PR #{number}: already checked, skipping", file=sys.stderr)
        return

    content = fetch_file_content(repo, txt_files[0], pr["headRefOid"])
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content)
        request_path = Path(f.name)
    try:
        errors = validate_request_files.parse_request_file(request_path)
    finally:
        request_path.unlink(missing_ok=True)

    if errors:
        # A wrong subject is not fixable by editing the request, so the PR is closed
        # instead of being left open for a revision.
        if validate_request_files.SUBJECT_ERROR in errors:
            if already_commented(repo, number, WRONG_SUBJECT_MARKER):
                print(f"PR #{number}: wrong subject, already answered", file=sys.stderr)
                return
            body = f"{WRONG_SUBJECT_MARKER}\n{WRONG_SUBJECT_MESSAGE}"
            if dry_run:
                print(f"\n--- PR #{number} (dry run, not closed) ---\n{body}\n")
            else:
                close_pr(repo, number, body)
                print(f"PR #{number}: wrong subject, closed", file=sys.stderr)
            return
        print(f"PR #{number}: invalid request, skipping ({'; '.join(errors)})", file=sys.stderr)
        return

    fields = parse_fields(content)
    paper_url = fields["Paper"]
    repo_url = fields.get("Repo")
    repo_label = repo_url or "(not provided — filed before Repo was required)"
    print(f"PR #{number}: checking {paper_url} …", file=sys.stderr)

    try:
        tmp_pdf = check_paper.download_pdf(paper_url)
    except Exception as e:
        print(f"PR #{number}: failed to download paper: {e}", file=sys.stderr)
        return

    checksum = check_paper.sha256_file(str(tmp_pdf))
    _MODELS_USED.clear()
    try:
        text = check_paper.pdf_to_text(str(tmp_pdf))
        result, _ = evaluate_paper(text, complete)
    except Exception as e:
        print(f"PR #{number}: evaluation failed: {e}", file=sys.stderr)
        return
    finally:
        tmp_pdf.unlink(missing_ok=True)

    repo_result = None
    if repo_url and not skip_repo:
        print(f"PR #{number}: checking repository {repo_url} …", file=sys.stderr)
        try:
            repo_result = check_repo.check_repo(repo_url, text, complete, threshold)
        except Exception as e:
            print(f"PR #{number}: repository check failed: {e}", file=sys.stderr)

    comment = build_comment(result, paper_url, repo_label, ", ".join(_MODELS_USED) or "unknown", checksum, repo_result)
    if dry_run:
        print(f"\n--- PR #{number} (dry run, not posted) ---\n{comment}\n")
    else:
        post_comment(repo, number, comment)
        print(f"PR #{number}: comment posted", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--dry-run", action="store_true", help="print comments instead of posting them")
    parser.add_argument("--update", action="store_true", help="re-run already-checked PRs and post a new check comment (does not edit prior comments)")
    parser.add_argument("--pr", type=int, action="append", help="only process these PR numbers (repeatable)")
    parser.add_argument("--no-repo-check", action="store_true", help="skip gate 4 (open science repository)")
    parser.add_argument(
        "--traceability-threshold",
        type=float,
        default=check_repo.DEFAULT_TRACEABILITY_THRESHOLD,
        help="fraction of the paper's empirical numbers that must be backed by the repository",
    )
    parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        default=DEFAULT_AGENT,
        help="which agent evaluates the paper (default: %(default)s)",
    )
    args = parser.parse_args()

    complete = make_completion(args.agent)

    for pr in list_open_prs(args.repo):
        if args.pr and pr["number"] not in args.pr:
            continue
        process_pr(
            args.repo, pr, args.dry_run, args.update,
            skip_repo=args.no_repo_check, threshold=args.traceability_threshold,
            complete=complete,
        )


if __name__ == "__main__":
    main()
