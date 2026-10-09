"""Offline tests for research.py: metadata (summarize / meta.json) and fail-closed error handling."""
import json
import os
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stubs import imported  # noqa: E402  (tests/ is on sys.path)

GOOD_REPORT = """# Survey about world models
## TL;DR
- A claim about world models [1].
- Another claim about evaluation [2].
- A Hugging Face record [3].

## Background
Foundational work [1].

## Trends and open problems
Still disputed [2].

## References
[1] World Models. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)
[2] A blog post. web. https://example.org/post (2025-02-10)
[3] HF paper. hf-search. https://huggingface.co/papers/2503.00003 (2025-03-04)
"""

GOOD_SOURCES = [
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "World Models", "date": "2025-01-02", "source": "arxiv"},
    {"n": 2, "id": "https://example.org/post", "url": "https://example.org/post", "title": "A blog post", "date": "2025-02-10", "source": "web"},
    {"n": 3, "id": "2503.00003", "url": "https://huggingface.co/papers/2503.00003", "title": "HF paper", "date": "2025-03-04", "source": "hf-search"},
]


class FakeMessage:
    """Minimal stand-in for a LangChain AIMessage with tool calls and usage metadata."""

    def __init__(self, tool_calls=(), usage=None):
        self.tool_calls = list(tool_calls)
        self.usage_metadata = usage


class FakeSleep:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self._ctx = imported("research", "agents")
        modules = self._ctx.__enter__()
        self.research = modules["research"]
        self.agents = modules["agents"]
        self.modules = modules
        self._tmp = []
        self.addCleanup(self._drop_tmp)
        self.addCleanup(self._ctx.__exit__, None, None, None)

    def _drop_tmp(self):
        import shutil
        for path in self._tmp:
            shutil.rmtree(path, ignore_errors=True)

    def tmpdir(self):
        """A scratch directory inside the workspace (tempfile.mkdtemp is not usable under the DSH sandbox)."""
        import uuid
        path = Path(__file__).parent / f".scratch-{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        self._tmp.append(path)
        return path

    # ---- summarize -------------------------------------------------------------------------------
    def test_summarize_counts_task_calls_and_tokens(self):
        messages = [
            FakeMessage(tool_calls=[{"name": "write_todos", "args": {}}], usage={"input_tokens": 100, "output_tokens": 20}),
            FakeMessage(tool_calls=[{"name": "task"}, {"name": "task"}, {"name": "task"}]),
            FakeMessage(tool_calls=[{"name": "write_file"}], usage={"input_tokens": 50, "output_tokens": 5}),
            FakeMessage(tool_calls=[{"name": "execute"}], usage={"input_tokens": 10, "output_tokens": 1}),
        ]
        meta = self.research.summarize(messages, 12.34, "stub-model")
        self.assertEqual(meta["model"], "stub-model")
        self.assertEqual(meta["elapsed_s"], 12.3)
        self.assertEqual(meta["subagent_calls"], 3)                       # RUBRIC 2.1
        self.assertEqual(meta["tool_calls"], {"execute": 1, "task": 3, "write_file": 1, "write_todos": 1})
        self.assertEqual(meta["tokens"], {"input": 160, "output": 26})

    def test_summarize_accepts_dicts_and_openai_style_tool_calls(self):
        messages = [
            {"tool_calls": [{"name": "task"}, {"function": {"name": "task"}}]},
            {"additional_kwargs": {"tool_calls": [{"function": {"name": "task"}}]}, "usage_metadata": {"input_tokens": 7, "output_tokens": 3}},
        ]
        meta = self.research.summarize(messages, 1.0, "m")
        self.assertEqual(meta["subagent_calls"], 3)
        self.assertEqual(meta["tokens"], {"input": 7, "output": 3})

    def test_summarize_is_robust_to_empty_and_malformed_messages(self):
        meta = self.research.summarize([None, {}, FakeMessage()], 0.04, "m")
        self.assertEqual(meta["subagent_calls"], 0)
        self.assertEqual(meta["tool_calls"], {})
        self.assertEqual(meta["tokens"], {"input": 0, "output": 0})
        self.assertEqual(meta["elapsed_s"], 0.0)

    def test_summarize_handles_garbage_usage_values(self):
        messages = [FakeMessage(usage={"input_tokens": "n/a", "output_tokens": None})]
        self.assertEqual(self.research.summarize(messages, 0, "m")["tokens"], {"input": 0, "output": 0})

    # ---- save_outputs: happy path ----------------------------------------------------------------
    def _run_save(self, report=GOOD_REPORT, sources=GOOD_SOURCES, topic="survey about world model",
                  messages=None, elapsed=3.0, download_override=None):
        tmp = self.tmpdir()
        payload = {self.agents.REPORT_PATH: None if report is None else report.encode(),
                   self.agents.SOURCES_PATH: None if sources is None else
                   (sources if isinstance(sources, bytes) else json.dumps(sources).encode())}
        if download_override is not None:
            payload = download_override
        backend = mock.Mock()
        with mock.patch.object(self.research, "download", return_value=payload):
            path = self.research.save_outputs(backend, topic, messages or [], elapsed, "stub-model", reports_dir=tmp)
            written = sorted(p.name for p in tmp.iterdir())
            slug = self.research.slugify(topic)
            meta = json.loads((tmp / f"{slug}.meta.json").read_text(encoding="utf-8"))
        return Path(path).name, written, meta

    def test_save_outputs_writes_three_files_and_the_metadata(self):
        messages = [FakeMessage(tool_calls=[{"name": "task"}] * 4, usage={"input_tokens": 9, "output_tokens": 2})]
        name, written, meta = self._run_save(messages=messages)
        self.assertEqual(name, "survey-about-world-model.md")
        self.assertEqual(written, ["survey-about-world-model.md", "survey-about-world-model.meta.json",
                                   "survey-about-world-model.sources.json"])
        self.assertEqual(meta["topic"], "survey about world model")
        self.assertEqual(meta["model"], "stub-model")
        self.assertEqual(meta["elapsed_s"], 3.0)
        self.assertEqual(meta["subagent_calls"], 4)
        self.assertEqual(meta["n_sources"], 3)
        self.assertEqual(meta["source_families"], ["arxiv", "hf-search", "web"])   # RUBRIC 2.2
        self.assertEqual(meta["tokens"], {"input": 9, "output": 2})

    def test_source_families_are_the_distinct_sources_in_sources_json(self):
        sources = [dict(entry, n=index + 1, source=source)
                   for index, (entry, source) in enumerate(zip(GOOD_SOURCES * 2, ["web", "web", "arxiv", "arxiv"]))]
        _, _, meta = self._run_save(sources=sources)
        self.assertEqual(meta["source_families"], ["arxiv", "web"])
        self.assertEqual(meta["n_sources"], len(sources))

    def test_saved_metadata_meets_the_rubric_thresholds(self):
        """RUBRIC 2.1/2.2 and self_check.py: >= 3 task calls and >= 3 of the 4 source families."""
        messages = [FakeMessage(tool_calls=[{"name": "task"}] * 4)]
        _, _, meta = self._run_save(messages=messages)
        self.assertGreaterEqual(meta["subagent_calls"], 3)
        self.assertGreaterEqual(len(set(meta["source_families"]) & set(self.research.FAMILIES)), 3)
        for key in ("topic", "model", "elapsed_s", "subagent_calls", "tool_calls", "tokens", "n_sources",
                    "source_families"):
            self.assertIn(key, meta)                     # every key meta.json is graded on is present

    def test_saved_report_passes_the_validators_six_rules(self):
        from check_citations import check
        _, _, _ = self._run_save()
        tmp = self._tmp[-1]
        report = (tmp / "survey-about-world-model.md").read_text(encoding="utf-8")
        sources = json.loads((tmp / "survey-about-world-model.sources.json").read_text(encoding="utf-8"))
        self.assertEqual(check(report, sources), [])

    def test_report_is_written_to_the_slug_of_the_topic(self):
        name, _, _ = self._run_save(topic="LLM   agents & tool use")
        self.assertEqual(name, "llm-agents-tool-use.md")

    # ---- save_outputs: fail closed ---------------------------------------------------------------
    def _expect_failure(self, **kwargs):
        tmp = self.tmpdir()
        payload = kwargs.pop("payload")
        backend = mock.Mock()
        with mock.patch.object(self.research, "download", return_value=payload):
            with self.assertRaises(RuntimeError) as caught:
                self.research.save_outputs(backend, "survey about world model", [], 1.0, "m", reports_dir=tmp)
        self.assertEqual(list(tmp.iterdir()), [])            # NOTHING was written
        return str(caught.exception)

    def test_missing_report_fails_closed(self):
        message = self._expect_failure(payload={self.agents.REPORT_PATH: None, self.agents.SOURCES_PATH: b"[]"})
        self.assertIn("no report", message)

    def test_empty_report_fails_closed(self):
        message = self._expect_failure(payload={self.agents.REPORT_PATH: b"   \n", self.agents.SOURCES_PATH: b"[]"})
        self.assertIn("no report", message)

    def test_missing_sources_fails_closed(self):
        message = self._expect_failure(payload={self.agents.REPORT_PATH: GOOD_REPORT.encode(),
                                                self.agents.SOURCES_PATH: None})
        self.assertIn("sources.json", message)

    def test_invalid_sources_json_fails_closed(self):
        message = self._expect_failure(payload={self.agents.REPORT_PATH: GOOD_REPORT.encode(),
                                                self.agents.SOURCES_PATH: b"{not json"})
        self.assertIn("not valid JSON", message)

    def test_empty_sources_list_fails_closed(self):
        message = self._expect_failure(payload={self.agents.REPORT_PATH: GOOD_REPORT.encode(),
                                                self.agents.SOURCES_PATH: b"[]"})
        self.assertIn("non-empty", message)

    def test_unfinalized_report_fails_closed(self):
        body = GOOD_REPORT.split("## References")[0]
        message = self._expect_failure(payload={self.agents.REPORT_PATH: body.encode(),
                                                self.agents.SOURCES_PATH: json.dumps(GOOD_SOURCES).encode()})
        self.assertIn("not finished", message)

    # ---- main -------------------------------------------------------------------------------------
    def test_main_without_a_topic_returns_2(self):
        self.assertEqual(self.research.main("   "), 2)

    def test_main_returns_1_when_the_model_cannot_be_built(self):
        with mock.patch.object(self.research, "make_model", side_effect=RuntimeError("no LAB_MODEL")):
            self.assertEqual(self.research.main("survey about world models"), 1)

    def test_main_returns_1_when_the_agent_run_crashes(self):
        with mock.patch.object(self.research, "open_sandbox", side_effect=RuntimeError("sandbox exploded")):
            self.assertEqual(self.research.main("survey about world models"), 1)

    def test_main_returns_1_when_the_run_produced_nothing(self):
        with mock.patch.object(self.research, "build_lead_agent", return_value=mock.Mock(invoke=lambda *a, **k: {"messages": []})), \
                mock.patch.object(self.research, "download", return_value={self.agents.REPORT_PATH: None,
                                                                          self.agents.SOURCES_PATH: None}):
            self.assertEqual(self.research.main("survey about world models"), 1)

    # ---- run_topic wiring -------------------------------------------------------------------------
    def test_run_topic_creates_directories_uploads_scripts_and_sets_the_recursion_limit(self):
        """Drive the real sandbox path: seed, invoke with the recursion limit, then download."""
        agent = mock.Mock()
        agent.invoke.return_value = {"messages": [FakeMessage(tool_calls=[{"name": "task"}] * 3)]}
        backend = mock.Mock()
        uploaded = self.modules["_uploaded"]
        self.modules["_download_result"].update({
            self.agents.REPORT_PATH: GOOD_REPORT.encode(),
            self.agents.SOURCES_PATH: json.dumps(GOOD_SOURCES).encode()})

        with mock.patch.object(self.research, "build_lead_agent", return_value=agent):
            self.research.run_topic(backend, "stub-model", "survey about world model", reports_dir=self.tmpdir())

        commands = [call.args[0] for call in backend.execute.call_args_list]
        self.assertTrue(any(self.agents.NOTES_DIR in command and f"{self.agents.WORKDIR}/report" in command
                            for command in commands), commands)
        self.assertIn(self.agents.VALIDATOR_PATH, uploaded)      # our validator is uploaded (RUBRIC 3.1)
        self.assertIn(self.agents.FINALIZER_PATH, uploaded)      # the provided finalizer too (GUIDE 2.6)
        self.assertIn(self.agents.TEMPLATE_PATH, uploaded)       # and the required report structure
        self.assertTrue(uploaded[self.agents.VALIDATOR_PATH].startswith(b'"""check_citations.py'))
        self.assertIn(b"def finalize(", uploaded[self.agents.FINALIZER_PATH])
        self.assertIn(b"## TL;DR", uploaded[self.agents.TEMPLATE_PATH])
        _, kwargs = agent.invoke.call_args
        self.assertEqual(kwargs["config"]["recursion_limit"], self.research.RECURSION_LIMIT)
        self.assertEqual(agent.invoke.call_args[0][0]["messages"][0]["role"], "user")

    # ---- transient provider failures --------------------------------------------------------------
    def test_transient_provider_error_re_runs_the_topic(self):
        agent = mock.Mock()
        agent.invoke.side_effect = [RuntimeError("no healthy backends"), {"messages": []}]
        result = self.research._invoke_with_retry(agent, {"messages": []}, sleep=FakeSleep())
        self.assertEqual(result, {"messages": []})
        self.assertEqual(agent.invoke.call_count, 2)

    def test_permanent_error_is_not_retried(self):
        agent = mock.Mock()
        agent.invoke.side_effect = ValueError("my bug")
        with self.assertRaises(ValueError):
            self.research._invoke_with_retry(agent, {"messages": []}, sleep=FakeSleep())
        self.assertEqual(agent.invoke.call_count, 1)

    def test_provider_retry_gives_up_after_the_bounded_attempts(self):
        agent = mock.Mock()
        agent.invoke.side_effect = RuntimeError("503 no healthy backends")
        sleep = FakeSleep()
        with self.assertRaises(RuntimeError):
            self.research._invoke_with_retry(agent, {"messages": []}, attempts=3, sleep=sleep)
        self.assertEqual(agent.invoke.call_count, 3)
        self.assertEqual(sleep.calls, [5.0, 10.0])       # waits only between attempts

    def test_provider_error_detection(self):
        transient = self.research._provider_transient
        self.assertTrue(transient(RuntimeError("503 no healthy backends")))
        self.assertTrue(transient(RuntimeError("Connection reset by peer")))
        self.assertTrue(transient(RuntimeError("Request timed out")))
        self.assertFalse(transient(ValueError("bad prompt")))

    def test_main_reports_success_and_writes_the_three_files(self):
        agent = mock.Mock()
        agent.invoke.return_value = {"messages": [FakeMessage(tool_calls=[{"name": "task"}] * 3)]}
        tmp = self.tmpdir()
        self.modules["_download_result"].update({
            self.agents.REPORT_PATH: GOOD_REPORT.encode(),
            self.agents.SOURCES_PATH: json.dumps(GOOD_SOURCES).encode()})
        with mock.patch.object(self.research, "build_lead_agent", return_value=agent), \
                mock.patch.object(self.research, "REPORTS", tmp):
            self.assertEqual(self.research.main("survey about world model"), 0)
        self.assertEqual(sorted(path.name for path in tmp.iterdir()),
                         ["survey-about-world-model.md", "survey-about-world-model.meta.json",
                          "survey-about-world-model.sources.json"])
        meta = json.loads((tmp / "survey-about-world-model.meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["subagent_calls"], 3)
        self.assertEqual(meta["source_families"], ["arxiv", "hf-search", "web"])

    def test_prompt_carries_the_topic_and_the_workspace_paths(self):
        prompt = self.research.build_prompt("survey about world models")
        self.assertIn("survey about world models", prompt)
        self.assertIn(self.agents.SOURCES_PATH, prompt)
        self.assertIn(self.agents.REPORT_PATH, prompt)
        self.assertIn(self.agents.FINALIZER_PATH, prompt)
        self.assertIn(self.agents.VALIDATOR_PATH, prompt)

    # ---- observability ----------------------------------------------------------------------------
    def test_progress_logging_reads_todos_tool_calls_and_messages(self):
        result = {"messages": [FakeMessage(tool_calls=[{"name": "task"}, {"name": "task"}, {"name": "task"},
                                                       {"name": "write_file"}])],
                  "todos": [{"content": "plan", "status": "completed"},
                            {"content": "delegate", "status": "completed"},
                            {"content": "write", "status": "in_progress"}]}
        self.assertEqual(self.research._describe_todos(self.research._todos_of(result)), "3 planned, 2 completed")
        calls = Counter()
        for message in self.research._messages_of(result):
            for call in self.research._tool_calls_of(message):
                calls[self.research._tool_name(call)] += 1
        self.assertEqual(self.research._describe_calls(calls), "taskx3, write_filex1")
        self.assertEqual(self.research._describe_calls(Counter()), "none")
        self.assertEqual(self.research._todos_of("not a dict"), [])

    def test_progress_logging_survives_a_result_without_todos(self):
        self.research._log_progress({"messages": []}, 1.0)          # must not raise

    # ---- agent wiring -----------------------------------------------------------------------------
    def test_subagents_and_limits_are_wired(self):
        subagents = self.agents.build_subagents()
        by_name = {spec["name"]: spec for spec in subagents}
        self.assertEqual(set(by_name), {"researcher", "citation-checker"})
        self.assertEqual(len(by_name["researcher"]["tools"]), 5)                # the whole SOURCE_TOOLS registry
        self.assertEqual([getattr(tool, "name", getattr(tool, "__name__", "?"))
                          for tool in by_name["citation-checker"]["tools"]], ["web_fetch"])
        for spec in subagents:
            limits = {type(middleware).__name__: middleware.run_limit for middleware in spec["middleware"]}
            self.assertEqual(set(limits), {"ModelCallLimitMiddleware", "ToolCallLimitMiddleware"})
            self.assertIsNotNone(limits["ModelCallLimitMiddleware"])
        self.assertIn("notes", by_name["researcher"]["description"].lower())
        self.assertIn("URL", by_name["citation-checker"]["description"])

    def test_lead_agent_gets_todo_middleware_and_limits(self):
        backend = mock.Mock()
        built = self.agents.build_lead_agent(backend, "stub-model")
        self.assertIs(built["backend"], backend)
        self.assertEqual(built["system_prompt"], self.agents.LEAD_PROMPT)
        kinds = [type(middleware).__name__ for middleware in built["middleware"]]
        self.assertEqual(kinds[0], "TodoListMiddleware")                 # without it there is no write_todos
        self.assertIn("ModelCallLimitMiddleware", kinds)
        self.assertIn("ToolCallLimitMiddleware", kinds)
        self.assertEqual(len(built["subagents"]), 2)

    def test_lead_prompt_requires_the_guide_steps(self):
        prompt = self.agents.LEAD_PROMPT
        for fragment in ("write_todos", "task", "sources.json", "## References", "3 of the 4",
                         "finalize_citations.py", "check_citations.py", "citation-checker"):
            self.assertIn(fragment, prompt)
        self.assertIn(self.agents.NOTES_DIR, prompt)

    def test_researcher_prompt_treats_web_content_as_untrusted(self):
        prompt = self.agents.RESEARCHER_PROMPT
        self.assertIn("UNTRUSTED", prompt)
        for fragment in ("arxiv_search", "hf_daily_papers", "hf_search_papers", "web_search", "web_fetch"):
            self.assertIn(fragment, prompt)
        self.assertIn("Never add anything from memory", prompt)
        self.assertIn("key_points", prompt)

    def test_checker_prompt_defines_the_verdicts(self):
        for verdict in ("SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNVERIFIABLE"):
            self.assertIn(verdict, self.agents.CHECKER_PROMPT)


if __name__ == "__main__":
    unittest.main()
