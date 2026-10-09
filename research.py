"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import (FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, TEMPLATE_PATH, VALIDATOR_PATH, WORKDIR,
                    build_lead_agent)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator
TEMPLATE_SOURCE = ROOT / "REPORT_TEMPLATE.md"       # provided: the lead reads the required structure

SLUG_MAX_CHARS = 60
RECURSION_LIMIT = 1000  # lead graph steps (GUIDE 3, step 5); subagents are capped by agents.SUB_LIMITS
FAMILIES = ("arxiv", "hf-daily", "hf-search", "web")
INVOKE_ATTEMPTS = 3     # a provider 5xx must not throw away a run that is already most of the way there
PROVIDER_TRANSIENT = ("no healthy backends", "503", "502", "504", "rate limit", "temporarily unavailable",
                      "overloaded", "timeout", "timed out", "connection reset")


def _enable_tracing():
    """Turn on LangSmith tracing when a key is configured (GUIDE 5). No key -> tracing stays off."""
    if (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")) and \
            not (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")):
        os.environ["LANGSMITH_TRACING"] = "true"
    if os.getenv("LANGSMITH_TRACING") == "true" or os.getenv("LANGCHAIN_TRACING_V2") == "true":
        print("[lead] LangSmith tracing enabled")


def _provider_transient(error):
    """True for a transient inference-provider failure (5xx, overloaded, timeout) - worth one more run."""
    text = f"{type(error).__name__}: {error}".lower()
    return any(needle in text for needle in PROVIDER_TRANSIENT)


def _invoke_with_retry(agent, payload, *, attempts=INVOKE_ATTEMPTS, sleep=None):
    """Run the lead graph, retrying ONLY a transient provider failure.

    The whole graph is re-run, which costs tokens - but a 5xx late in a run would otherwise discard every
    note and every source, and LangGraph cannot resume without a checkpointer. The retry is bounded, and a
    non-provider error (a bug, a recursion limit) is never retried.
    """
    sleep = sleep or time.sleep
    for attempt in range(max(1, attempts)):
        try:
            return agent.invoke(payload, config={"recursion_limit": RECURSION_LIMIT})
        except Exception as error:  # noqa: BLE001 - decide by classification, not by type
            if attempt == max(1, attempts) - 1 or not _provider_transient(error):
                raise
            delay = 5.0 * (2 ** attempt)
            print(f"[lead] provider error ({type(error).__name__}: {str(error)[:120]}); "
                  f"re-running the topic in {delay:.0f}s (attempt {attempt + 2}/{attempts})", file=sys.stderr)
            sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^0-9a-z]+", "-", str(topic or "").strip().lower()).strip("-")
    return slug[:SLUG_MAX_CHARS].strip("-") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Research topic: {topic}\n\n"
        "Produce ONE cited survey report for this topic, exactly as your instructions describe:\n"
        f"1. plan with write_todos and split the topic into at least 3 independent sub-questions;\n"
        "2. delegate every sub-question to the `researcher` subagent with the `task` tool, all in one message, "
        "each delegation self-contained (topic, sub-question, required source families, notes path, note format);\n"
        "3. read and check each notes file before using it;\n"
        f"4. merge the verified records into {SOURCES_PATH} (numbered from 1, no duplicate URL), making sure at "
        f"least 3 of the 4 source families {list(FAMILIES)} are present, and delegate one more researcher if a "
        "family is missing;\n"
        f"5. write the report body to {REPORT_PATH} following REPORT_TEMPLATE.md (synthesis by theme, inline [n] "
        "citations, no `## References` section);\n"
        f"6. run the finalizer `python3 {FINALIZER_PATH}`, then the validator `python3 {VALIDATOR_PATH}`, and fix "
        "the report until the validator prints OK;\n"
        "7. spot-check 3-5 claims with the `citation-checker` subagent;\n"
        "8. reply with a short final summary (report path, number of sources, source families, validator output)."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    Walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_calls = Counter()
    input_tokens = output_tokens = 0
    for message in messages or []:
        for call in _tool_calls_of(message):
            tool_calls[_tool_name(call)] += 1
        usage = _usage_of(message)
        if usage:
            input_tokens += _as_int(usage.get("input_tokens"))
            output_tokens += _as_int(usage.get("output_tokens"))
    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": tool_calls["task"],
        "tool_calls": dict(sorted(tool_calls.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def _messages_of(result):
    """Messages out of whatever `agent.invoke(...)` returned (dict, state object, or a list)."""
    if isinstance(result, dict):
        return result.get("messages") or []
    messages = getattr(result, "messages", None)
    if messages is not None:
        return messages
    return result if isinstance(result, list) else []


def _tool_calls_of(message):
    """tool_calls of one message; works for dicts and for LangChain message objects."""
    calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    if not calls:
        # Some providers expose only the raw OpenAI-style key, with the name nested under "function".
        calls = message.get("additional_kwargs", {}).get("tool_calls") if isinstance(message, dict) else None
    return calls or []


def _tool_name(call):
    """The tool name of one tool_call; accepts {"name": ...} and {"function": {"name": ...}}."""
    if isinstance(call, dict):
        if call.get("name"):
            return str(call["name"])
        function = call.get("function")
        if isinstance(function, dict) and function.get("name"):
            return str(function["name"])
        return "unknown"
    name = getattr(call, "name", None)
    return str(name) if name else "unknown"


def _usage_of(message):
    """usage_metadata of one message, when the provider reported it."""
    if isinstance(message, dict):
        return message.get("usage_metadata") or message.get("response_metadata", {}).get("usage_metadata")
    return getattr(message, "usage_metadata", None)


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _todos_of(result):
    """The lead's todo list out of whatever `agent.invoke` returned (GUIDE 5 suggests tracing; this is cheap)."""
    if not isinstance(result, dict):
        return []
    todos = result.get("todos")
    if todos is None:
        todos = getattr(result, "todos", None)
    return todos if isinstance(todos, list) else []


def _describe_todos(todos):
    """One line: how many steps the lead planned and how far it got."""
    done = sum(1 for todo in todos if isinstance(todo, dict)
               and str(todo.get("status", "")).lower() in ("completed", "done"))
    return f"{len(todos)} planned, {done} completed"


def _describe_calls(tool_calls):
    return ", ".join(f"{name}x{count}" for name, count in sorted(tool_calls.items())) or "none"


def _log_progress(result, elapsed):
    """Print what the lead did. A long run is otherwise silent until the sandbox is torn down."""
    messages = _messages_of(result)
    tool_calls = Counter()
    for message in messages:
        for call in _tool_calls_of(message):
            tool_calls[_tool_name(call)] += 1
    if tool_calls:
        print(f"[lead] tool calls: {_describe_calls(tool_calls)}")
    todos = _todos_of(result)
    if todos:
        print(f"[lead] todos: {_describe_todos(todos)}")
    print(f"[lead] finished its graph in {elapsed:.0f}s with {len(messages)} messages")


def _model_name():
    """The model id reported in meta.json: the LAB_MODEL/OPENAI_DEPLOYMENT_MODEL value that model.py used."""
    return (os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or "").strip() or "unknown"


def _decode(content):
    """Sandbox bytes -> text; a download failure (None) stays None."""
    if content is None:
        return None
    return content.decode("utf-8", errors="replace")


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    Fails closed (GUIDE 3, RUBRIC 3.2): a failed run raises RuntimeError and writes NOTHING, so a broken
    run can never leave an empty report that looks like a success.
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_text = _decode(files.get(REPORT_PATH))
    sources_text = _decode(files.get(SOURCES_PATH))

    if report_text is None or not report_text.strip():
        raise RuntimeError(f"the run produced no report at {REPORT_PATH} (nothing was written to reports/)")
    if sources_text is None:
        raise RuntimeError(f"the run produced no {SOURCES_PATH} (nothing was written to reports/)")
    try:
        sources = json.loads(sources_text)
    except ValueError as exc:
        raise RuntimeError(f"{SOURCES_PATH} is not valid JSON: {exc} (nothing was written to reports/)") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError(f"{SOURCES_PATH} must be a non-empty JSON array (nothing was written to reports/)")
    if not report_text.lstrip().startswith("#") or "## References" not in report_text:
        raise RuntimeError("the report is not finished (title/`## References` missing): run the finalizer and "
                           "the validator before saving (nothing was written to reports/)")

    families = sorted({str(entry.get("source")) for entry in sources if isinstance(entry, dict) and entry.get("source")})
    meta = {"topic": topic}
    meta.update(summarize(messages, elapsed, model_name))
    meta["n_sources"] = len(sources)
    meta["source_families"] = families

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    report_path = reports_dir / f"{slug}.md"
    (reports_dir / f"{slug}.sources.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (reports_dir / f"{slug}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def run_topic(backend, model, topic, reports_dir=REPORTS):
    """Run one deep-research topic inside an already-open sandbox and save the reports. Returns the report path."""
    backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
    upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                     FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                     TEMPLATE_PATH: TEMPLATE_SOURCE.read_bytes()})
    agent = build_lead_agent(backend, model)
    started = time.monotonic()
    print(f"[lead] running the deep-research graph (recursion_limit={RECURSION_LIMIT})")
    result = _invoke_with_retry(agent, {"messages": [{"role": "user", "content": build_prompt(topic)}]})
    elapsed = time.monotonic() - started
    _log_progress(result, elapsed)
    return save_outputs(backend, topic, _messages_of(result), elapsed, _model_name(), reports_dir=reports_dir)


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print('usage: python research.py "<topic>"   (topics.md lists the five preset topics)', file=sys.stderr)
        return 2
    _enable_tracing()                              # a no-op unless a LangSmith key is configured
    try:
        model = make_model()
    except Exception as exc:  # noqa: BLE001 - a missing model config is a failed run, not a crash
        print(f"FAILED: cannot build the model: {exc}", file=sys.stderr)
        return 1
    try:
        with open_sandbox() as backend:            # the sandbox is always stopped/removed, even on errors
            report_path = run_topic(backend, model, topic, reports_dir=REPORTS)
    except Exception as exc:  # noqa: BLE001 - any failed run exits 1 and never writes a report
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"saved {report_path} (+ .sources.json, .meta.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
