"""Offline tests: retry/backoff, query sanitising, arXiv Atom parsing, slugify.

These import the lab modules through tests/stubs.py, so they run on a bare Python 3.11+ without langchain,
deepagents, daytona or pytest (they also run under pytest).
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stubs import imported  # noqa: E402  (tests/ is on sys.path)

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v2</id>
    <published>2025-01-02T18:00:00Z</published>
    <title>World   Models
      for   Agents</title>
    <summary>
      A survey of   world models.
      Second line.
    </summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2502.00002v1</id>
    <published>2025-02-03T09:30:00Z</published>
    <title>Latent Dynamics</title>
    <summary>Short abstract.</summary>
  </entry>
</feed>
"""

EMPTY_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""


class FakeSleep:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


class RetryTests(unittest.TestCase):
    def setUp(self):
        self._ctx = imported("tools")
        self.tools = self._ctx.__enter__()["tools"]
        self.tools._ARXIV_LIMIT_HITS[0], self.tools._ARXIV_OFF_UNTIL[0] = 0, 0.0
        self.addCleanup(self._reset_breaker)

    def _reset_breaker(self):
        self.tools._ARXIV_LIMIT_HITS[0], self.tools._ARXIV_OFF_UNTIL[0] = 0, 0.0

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def make_client(self, responder):
        """An httpx.Client whose transport answers from `responder(request) -> httpx.Response`."""
        transport = self.tools.httpx.MockTransport(responder)
        return self.tools.httpx.Client(transport=transport, base_url="https://example.test")

    # ---- backoff maths ---------------------------------------------------------------------------
    def test_backoff_is_exponential_and_capped(self):
        tools = self.tools
        no_jitter = lambda: 0.0  # noqa: E731
        self.assertEqual(tools.backoff_delay(0, base=1.0, cap=30.0, jitter=no_jitter), 1.0)
        self.assertEqual(tools.backoff_delay(1, base=1.0, cap=30.0, jitter=no_jitter), 2.0)
        self.assertEqual(tools.backoff_delay(2, base=1.0, cap=30.0, jitter=no_jitter), 4.0)
        self.assertEqual(tools.backoff_delay(9, base=1.0, cap=30.0, jitter=no_jitter), 30.0)

    def test_backoff_adds_jitter_below_the_cap(self):
        tools = self.tools
        self.assertEqual(tools.backoff_delay(1, base=1.0, cap=30.0, jitter=lambda: 0.5), 2.5)

    def test_retry_after_wins_and_is_not_jittered(self):
        tools = self.tools
        self.assertEqual(tools.backoff_delay(0, base=1.0, cap=30.0, retry_after=7, jitter=lambda: 0.9), 7.0)
        # a nonsensical server hint is clamped instead of sleeping for hours
        self.assertEqual(tools.backoff_delay(0, base=1.0, cap=30.0, retry_after=10_000, jitter=lambda: 0.0), 300.0)

    def test_retry_after_header_parsing(self):
        tools = self.tools
        self.assertEqual(tools._retry_after_seconds("12"), 12.0)
        self.assertEqual(tools._retry_after_seconds("  3.5 "), 3.5)
        self.assertIsNone(tools._retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT"))
        self.assertIsNone(tools._retry_after_seconds("-2"))
        self.assertIsNone(tools._retry_after_seconds(None))

    # ---- with_retry behaviour --------------------------------------------------------------------
    def test_retries_then_succeeds(self):
        tools = self.tools
        sleep = FakeSleep()
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise tools.RetryableError("HTTP 429")
            return "value"

        result = tools.with_retry(flaky, attempts=5, sleep=sleep)
        self.assertEqual(result, "value")
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(sleep.calls), 2)  # exactly one wait between attempts

    def test_gives_up_after_attempts_and_never_sleeps_after_the_last_one(self):
        tools = self.tools
        sleep = FakeSleep()
        calls = []

        def always_failing():
            calls.append(1)
            raise tools.RetryableError("HTTP 503")

        with self.assertRaises(tools.RetryableError):
            tools.with_retry(always_failing, attempts=3, sleep=sleep)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(sleep.calls), 2)  # sleeps only between attempts, not after the last

    def test_non_retryable_error_is_raised_immediately(self):
        tools = self.tools
        sleep = FakeSleep()
        calls = []

        def broken():
            calls.append(1)
            raise ValueError("bug in my code")

        with self.assertRaises(ValueError):
            tools.with_retry(broken, attempts=5, sleep=sleep)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sleep.calls, [])

    def test_server_retry_after_is_honoured(self):
        tools = self.tools
        sleep = FakeSleep()
        calls = []

        def limited():
            calls.append(1)
            if len(calls) == 1:
                raise tools.RetryableError("HTTP 429", retry_after=42)
            return "ok"

        self.assertEqual(tools.with_retry(limited, attempts=4, sleep=sleep), "ok")
        self.assertEqual(sleep.calls, [42.0])

    def test_transport_error_is_retryable(self):
        tools = self.tools
        sleep = FakeSleep()
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise tools.httpx.ConnectTimeout("timed out")
            return "ok"

        self.assertEqual(tools.with_retry(flaky, attempts=3, sleep=sleep), "ok")
        self.assertEqual(len(sleep.calls), 1)

    # ---- status classification -------------------------------------------------------------------
    def test_retryable_statuses(self):
        tools = self.tools
        for status in (429, 500, 502, 503, 504):
            error = tools.retryable_from_status(status)
            self.assertIsInstance(error, tools.RetryableError, f"HTTP {status} must be retryable")

    def test_permanent_statuses_are_not_retryable(self):
        tools = self.tools
        for status in (400, 401, 403, 404, 422):
            error = tools.retryable_from_status(status)
            self.assertNotIsInstance(error, tools.RetryableError, f"HTTP {status} must not be retried")

    def test_retry_after_is_read_from_the_response_headers(self):
        tools = self.tools
        response = mock.Mock(status_code=429, headers={"Retry-After": "9"})
        error = tools.retryable_from_status(response.status_code, response.headers)
        self.assertEqual(error.retry_after, 9.0)

    def test_http_status_error_is_classified(self):
        tools = self.tools
        request = tools.httpx.Request("GET", "https://example.org")
        retryable = tools.httpx.HTTPStatusError("503", request=request,
                                                response=tools.httpx.Response(503, request=request))
        permanent = tools.httpx.HTTPStatusError("404", request=request,
                                                response=tools.httpx.Response(404, request=request))
        self.assertEqual(tools.classify_exception(retryable)[0], True)
        self.assertEqual(tools.classify_exception(permanent), (False, None))
        self.assertEqual(tools.classify_exception(ValueError("x")), (False, None))

    # ---- arXiv query sanitising + Atom parsing ---------------------------------------------------
    def test_query_sanitising_drops_dangerous_syntax(self):
        tools = self.tools
        terms = tools._arxiv_terms('all:"world model" AND (agent:tool)')
        self.assertEqual(terms, ["all", "world", "model", "AND", "agent", "tool"])
        self.assertEqual(tools._arxiv_search_query(["world", "model"]), "all:world AND all:model")
        self.assertEqual(tools._arxiv_terms('":::"'), [])
        self.assertEqual(tools._arxiv_terms(""), [])

    def test_empty_query_returns_no_records_without_network(self):
        tools = self.tools
        with mock.patch.object(tools, "_http_get", side_effect=AssertionError("must not call the network")):
            self.assertEqual(tools.arxiv_records('"::"'), [])

    def test_arxiv_search_returns_no_results_on_empty_feed(self):
        tools = self.tools
        with mock.patch.object(tools, "_http_get", return_value=mock.Mock(text=EMPTY_ATOM)), \
                mock.patch.object(tools.time, "sleep"):
            self.assertEqual(tools.arxiv_search.invoke({"query": "world model"}), "NO RESULTS")

    def test_atom_parsing_normalises_records(self):
        tools = self.tools
        records = tools.parse_arxiv_atom(ATOM)
        self.assertEqual(len(records), 2)
        first = records[0]
        self.assertEqual(first["id"], "2501.00001")                       # version suffix dropped
        self.assertEqual(first["url"], "https://arxiv.org/abs/2501.00001")  # https, no vN
        self.assertEqual(first["published"], "2025-01-02")
        self.assertEqual(first["title"], "World Models for Agents")       # whitespace collapsed
        self.assertEqual(first["summary"], "A survey of world models. Second line.")
        self.assertEqual(set(first), {"id", "url", "published", "title", "summary"})

    def test_atom_parsing_cuts_a_long_summary(self):
        tools = self.tools
        long_atom = ATOM.replace("A survey of   world models.", "x" * 900)
        summary = tools.parse_arxiv_atom(long_atom)[0]["summary"]
        self.assertLessEqual(len(summary), tools.SUMMARY_CHARS)
        self.assertTrue(summary.endswith("\u2026"))

    def test_arxiv_search_serialises_records_to_json(self):
        tools = self.tools
        with mock.patch.object(tools, "_http_get", return_value=mock.Mock(text=ATOM)), \
                mock.patch.object(tools.time, "sleep"):
            payload = tools.arxiv_search.invoke({"query": "world model", "max_results": 3})
        records = json.loads(payload)
        self.assertEqual([record["id"] for record in records], ["2501.00001", "2502.00002"])
        self.assertEqual(records[0]["url"], "https://arxiv.org/abs/2501.00001")

    def test_arxiv_tool_never_raises(self):
        tools = self.tools
        with mock.patch.object(tools, "arxiv_records", side_effect=RuntimeError("HTTP 429 after retries")):
            answer = tools.arxiv_search.invoke({"query": "world model"})
        self.assertTrue(answer.startswith("ERROR: "), answer)

    # ---- arXiv shared-IP cooldown (GUIDE 1.2 / 5) -------------------------------------------------
    def test_repeated_429s_pause_the_family_and_skip_the_network(self):
        tools = self.tools
        tools._ARXIV_LIMIT_HITS[0], tools._ARXIV_OFF_UNTIL[0] = 0, 0.0

        def responder(request):
            return tools.httpx.Response(429, request=request)

        # NOTE: patch the shared `time` module attribute. `with_retry(sleep=time.sleep)` is a DEFAULT
        # argument bound at import, so patching tools.time.sleep would not intercept it.
        with mock.patch("time.sleep") as slept:
            for _ in range(tools.ARXIV_COOLDOWN_AFTER):
                with self.make_client(responder) as client:
                    with self.assertRaises(Exception) as caught:
                        tools.arxiv_records("world model", client=client)
                    self.assertIn("429", str(caught.exception))
            self.assertTrue(slept.called, "the retry loop must have waited between attempts")
        self.assertGreater(tools._ARXIV_OFF_UNTIL[0], 0.0)

        calls = []
        with mock.patch.object(tools, "_http_get", side_effect=lambda *a, **k: (calls.append(1), mock.Mock(text=EMPTY_ATOM))[1]):
            with self.assertRaises(tools.RetryableError) as caught:
                tools.arxiv_records("world model")
        self.assertEqual(calls, [], "a paused family must not touch the network")
        self.assertIn("paused", str(caught.exception))
        self.assertIn("web_search", str(caught.exception))     # the message tells the agent what to use instead

        # the tool still answers with a string, never an exception
        with mock.patch.object(tools, "_http_get",
                               side_effect=lambda *a, **k: (calls.append(1), mock.Mock(text=EMPTY_ATOM))[1]):
            answer = tools.arxiv_search.invoke({"query": "world model"})
        self.assertTrue(answer.startswith("ERROR: "), answer)
        self.assertEqual(calls, [])

    def test_the_cooldown_expires_by_itself(self):
        tools = self.tools
        # an EXPIRED pause (a real past timestamp, actually set) must reset and allow the next call
        tools._ARXIV_LIMIT_HITS[0], tools._ARXIV_OFF_UNTIL[0] = 9, tools.time.monotonic() - 1
        self.assertIsNone(tools._arxiv_cooldown_active())
        self.assertEqual((tools._ARXIV_LIMIT_HITS[0], tools._ARXIV_OFF_UNTIL[0]), (0, 0.0))

    def test_a_successful_call_resets_the_consecutive_failure_count(self):
        tools = self.tools
        tools._ARXIV_LIMIT_HITS[0], tools._ARXIV_OFF_UNTIL[0] = 2, 0.0
        with mock.patch.object(tools.time, "sleep"), \
                mock.patch.object(tools, "_http_get", return_value=mock.Mock(text=EMPTY_ATOM)):
            tools.arxiv_records("world model")
        self.assertEqual(tools._ARXIV_LIMIT_HITS[0], 0)

    def test_arxiv_pacing_waits_between_calls(self):
        tools = self.tools
        slept = FakeSleep()
        paced = [None]
        clock = mock.Mock(side_effect=[100.0])
        tools._pace(tools.ARXIV_MIN_INTERVAL, paced, sleep=slept, clock=clock)
        self.assertEqual(slept.calls, [])              # the very first call has nothing to wait for
        self.assertEqual(paced[0], 100.0)

        clock = mock.Mock(side_effect=[101.0, 104.0])  # a second call 1s later still owes 2s of the window
        tools._pace(tools.ARXIV_MIN_INTERVAL, paced, sleep=slept, clock=clock)
        self.assertEqual(len(slept.calls), 1)
        self.assertAlmostEqual(slept.calls[0], tools.ARXIV_MIN_INTERVAL - 1.0)

        slept.calls.clear()
        clock = mock.Mock(side_effect=[110.0])         # a call long after the window: no wait at all
        tools._pace(tools.ARXIV_MIN_INTERVAL, paced, sleep=slept, clock=clock)
        self.assertEqual(slept.calls, [])

    def test_pacing_is_disabled_for_a_zero_interval(self):
        tools = self.tools
        slept = FakeSleep()
        tools._pace(0, [None], sleep=slept)
        self.assertEqual(slept.calls, [])


class SlugifyTests(unittest.TestCase):
    def setUp(self):
        self._ctx = imported("research")
        self.research = self._ctx.__enter__()["research"]

    def tearDown(self):
        self._ctx.__exit__(None, None, None)

    def test_basic_topic(self):
        self.assertEqual(self.research.slugify("survey about world model"), "survey-about-world-model")

    def test_runs_of_separators_collapse(self):
        self.assertEqual(self.research.slugify("LLM   agents  &  tool-use"), "llm-agents-tool-use")

    def test_path_traversal_cannot_escape_reports(self):
        for topic in ("../../x", "/etc/passwd", "..\\..\\windows", "....//....//etc"):
            slug = self.research.slugify(topic)
            self.assertNotIn("/", slug)
            self.assertNotIn("\\", slug)
            self.assertNotIn("..", slug)
            resolved = (self.research.REPORTS / f"{slug}.md").resolve()
            self.assertEqual(resolved.parent, self.research.REPORTS.resolve())

    def test_empty_or_symbol_only_topic_falls_back_to_topic(self):
        for topic in ("", "   ", "!!!", "///", None):
            self.assertEqual(self.research.slugify(topic), "topic")

    def test_long_topic_is_cut_to_60_chars_without_trailing_dash(self):
        slug = self.research.slugify("word " * 40)
        self.assertLessEqual(len(slug), 60)
        self.assertFalse(slug.endswith("-"))
        self.assertTrue(slug.startswith("word"))


if __name__ == "__main__":
    unittest.main()
