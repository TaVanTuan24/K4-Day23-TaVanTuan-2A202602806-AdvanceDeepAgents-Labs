"""Offline tests for the five source tools, driven by httpx.MockTransport (no network, no real waiting).

Covers: Hugging Face record mapping on both endpoints, "NO RESULTS" / "ERROR: ..." contracts, the Exa MCP
SSE answer, the free-tier rate limit Exa hides behind HTTP 200, EXA_API_KEY redaction, and web_fetch
truncation.
"""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stubs import imported  # noqa: E402  (tests/ is on sys.path)

DAILY_PAYLOAD = [
    {
        "paper": {
            "id": "2501.00001",
            "title": "World Models for Control",
            "summary": "A daily paper about world models.",
            "upvotes": 12,
            "githubRepo": "https://github.com/example/world-models",
            "githubStars": 340,
            "publishedAt": "2025-01-02T00:00:00.000Z",
        },
        "title": "World Models for Control",
        "publishedAt": "2025-01-02T00:00:00.000Z",
    },
    {
        "paper": {
            "id": "2501.00002",
            "title": "Unrelated Vision Paper",
            "summary": "Something else entirely.",
            "upvotes": 99,
            "githubRepo": None,
            "githubStars": None,
            "publishedAt": "2025-01-03T00:00:00.000Z",
        }
    },
    {"paper": {"title": "no id at all", "summary": "must be skipped"}},
]

SEARCH_PAYLOAD = [
    {
        "paper": {
            "id": "2503.00003",
            "title": "Efficient Agents",
            "summary": "long raw summary",
            "ai_summary": "short ai summary",
            "ai_keywords": ["agents"],
            "upvotes": 4,
            "publishedAt": "2025-03-04T00:00:00.000Z",
        }
    }
]


def sse(payload):
    return "event: message\ndata: " + json.dumps(payload) + "\n\n"


def mcp_text(text, meta=None):
    result = {"content": [{"type": "text", "text": text}]}
    if meta is not None:
        result["_meta"] = meta
    return sse({"jsonrpc": "2.0", "id": 1, "result": result})


class FakeSleep:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


class SourceToolTests(unittest.TestCase):
    def setUp(self):
        self._ctx = imported("tools")
        self.tools = self._ctx.__enter__()["tools"]
        self.addCleanup(self._ctx.__exit__, None, None, None)
        self.tools._ARXIV_PACING[0] = None
        self.tools._EXA_PACING[0] = None

    def make_client(self, responder):
        """An httpx.Client whose transport answers from `responder(request) -> httpx.Response`."""
        transport = self.tools.httpx.MockTransport(responder)
        return self.tools.httpx.Client(transport=transport, base_url="https://example.test")

    # ---- Hugging Face -----------------------------------------------------------------------------
    def test_hf_daily_maps_records_and_skips_items_without_an_id(self):
        tools = self.tools

        def responder(request):
            return tools.httpx.Response(200, json=DAILY_PAYLOAD, request=request)

        with self.make_client(responder) as client:
            records = tools.hf_daily_records(limit=20, client=client)
        self.assertEqual([record["id"] for record in records], ["2501.00002", "2501.00001"])  # sorted by upvotes
        first = records[1]
        self.assertEqual(first["url"], "https://huggingface.co/papers/2501.00001")
        self.assertEqual(first["published"], "2025-01-02")
        self.assertEqual(first["upvotes"], 12)
        self.assertEqual(first["github"], "https://github.com/example/world-models")
        self.assertEqual(first["stars"], 340)

    def test_hf_daily_filters_by_keyword_case_insensitively(self):
        tools = self.tools

        def responder(request):
            return tools.httpx.Response(200, json=DAILY_PAYLOAD, request=request)

        with self.make_client(responder) as client:
            records = tools.hf_daily_records(limit=20, keyword="WORLD", client=client)
        self.assertEqual([record["id"] for record in records], ["2501.00001"])

    def test_hf_daily_sends_the_date_parameter_only_when_given(self):
        tools = self.tools
        seen = {}

        def responder(request):
            seen["url"] = str(request.url)
            return tools.httpx.Response(200, json=[], request=request)

        with self.make_client(responder) as client:
            tools.hf_daily_records(limit=5, client=client)
            self.assertNotIn("date=", seen["url"])
            tools.hf_daily_records(limit=5, date="2025-01-02", client=client)
            self.assertIn("date=2025-01-02", seen["url"])
            self.assertIn("limit=5", seen["url"])

    def test_hf_search_prefers_ai_summary(self):
        tools = self.tools

        def responder(request):
            return tools.httpx.Response(200, json=SEARCH_PAYLOAD, request=request)

        with self.make_client(responder) as client:
            records = tools.hf_search_records("agents", client=client)
        self.assertEqual(records[0]["summary"], "short ai summary")
        self.assertEqual(records[0]["url"], "https://huggingface.co/papers/2503.00003")

    def test_hf_tools_return_no_results_when_the_payload_is_empty(self):
        tools = self.tools
        with mock.patch.object(tools, "hf_daily_records", return_value=[]), \
                mock.patch.object(tools, "hf_search_records", return_value=[]):
            self.assertEqual(tools.hf_daily_papers.invoke({"limit": 5}), "NO RESULTS")
            self.assertEqual(tools.hf_search_papers.invoke({"query": "x"}), "NO RESULTS")

    def test_hf_tool_error_is_a_string_not_an_exception(self):
        tools = self.tools
        with mock.patch.object(tools, "hf_search_records", side_effect=RuntimeError("HTTP 503 after retries")):
            answer = tools.hf_search_papers.invoke({"query": "agents"})
        self.assertTrue(answer.startswith("ERROR: "), answer)
        self.assertIn("503", answer)

    def test_hf_tool_handles_a_non_json_body(self):
        tools = self.tools
        with mock.patch.object(tools, "_http_get",
                               return_value=mock.Mock(json=mock.Mock(side_effect=ValueError("not json")))):
            answer = tools.hf_daily_papers.invoke({"limit": 5})
        self.assertTrue(answer.startswith("ERROR: "), answer)

    # ---- Exa via the MCP endpoint -----------------------------------------------------------------
    def test_exa_search_reads_the_sse_data_line(self):
        tools = self.tools

        def responder(request):
            body = json.loads(request.content.decode())
            self.assertEqual(body["method"], "tools/call")
            self.assertEqual(body["params"]["name"], "web_search_exa")
            self.assertEqual(body["params"]["arguments"]["query"], "world models")
            self.assertTrue(body["params"]["arguments"]["objective"])
            return tools.httpx.Response(200, text=mcp_text("Result page text with https://example.org"),
                                        request=request)

        with self.make_client(responder) as client:
            text = tools.exa_call("web_search_exa", {"query": "world models", "objective": "o"},
                                  client=client, pace=False)
        self.assertIn("https://example.org", text)

    def test_exa_accept_header_asks_for_sse(self):
        tools = self.tools
        seen = {}

        def responder(request):
            seen["accept"] = request.headers.get("accept", "")
            return tools.httpx.Response(200, text=mcp_text("ok"), request=request)

        with self.make_client(responder) as client:
            tools.exa_call("web_search_exa", {"query": "q", "objective": "o"}, client=client, pace=False)
        self.assertIn("text/event-stream", seen["accept"])

    def test_exa_jsonrpc_error_becomes_an_exception(self):
        tools = self.tools

        def responder(request):
            return tools.httpx.Response(200, text=sse({"jsonrpc": "2.0", "id": 1,
                                                       "error": {"code": -32602, "message": "bad arguments"}}),
                                        request=request)

        with self.make_client(responder) as client:
            with self.assertRaises(Exception) as caught:
                tools.exa_call("web_search_exa", {"query": "q", "objective": "o"}, client=client, pace=False)
        self.assertIn("bad arguments", str(caught.exception))

    def test_free_tier_rate_limit_is_detected_through_the_meta_flag_and_retried(self):
        tools = self.tools
        calls = []
        sleep = FakeSleep()

        def responder(request):
            calls.append(1)
            if len(calls) < 3:
                return tools.httpx.Response(
                    200, text=mcp_text("You have hit the rate limit. Please try again later.",
                                       meta={"exaRateLimited": True}), request=request)
            return tools.httpx.Response(200, text=mcp_text("real page content"), request=request)

        with self.make_client(responder) as client:
            text = tools.exa_call("web_search_exa", {"query": "q", "objective": "o"},
                                  client=client, attempts=5, sleep=sleep, pace=False)
        self.assertEqual(text, "real page content")     # the rate-limit note never reaches the agent as content
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(sleep.calls), 2)

    def test_rate_limit_without_a_meta_flag_is_also_detected(self):
        tools = self.tools

        def responder(request):
            return tools.httpx.Response(200, text=mcp_text("Error: too many requests, slow down"), request=request)

        with self.make_client(responder) as client:
            with self.assertRaises(tools.RetryableError):
                tools.exa_call("web_search_exa", {"query": "q", "objective": "o"},
                               client=client, attempts=2, sleep=FakeSleep(), pace=False)

    def test_rate_limit_exhausting_the_attempts_is_reported(self):
        tools = self.tools
        sleep = FakeSleep()

        def responder(request):
            return tools.httpx.Response(200, text=mcp_text("rate limit reached", meta={"exaRateLimited": True}),
                                        request=request)

        with self.make_client(responder) as client:
            with self.assertRaises(tools.RetryableError):
                tools.exa_call("web_search_exa", {"query": "q", "objective": "o"},
                               client=client, attempts=3, sleep=sleep, pace=False)
        self.assertEqual(len(sleep.calls), 2)            # two waits, none after the final attempt

    def test_web_search_reports_exa_failure_as_an_error_string(self):
        tools = self.tools
        with mock.patch.object(tools, "EXA_MIN_INTERVAL", 0), \
                mock.patch.object(tools, "exa_call", side_effect=tools.RetryableError("Exa rate limit")):
            answer = tools.web_search.invoke({"query": "world models"})
        self.assertTrue(answer.startswith("ERROR: "), answer)
        self.assertIn("rate limit", answer.lower())

    def test_web_search_builds_a_default_objective_and_never_raises(self):
        tools = self.tools
        with mock.patch.object(tools, "exa_call", return_value="text") as call:
            self.assertEqual(tools.web_search.invoke({"query": "world models"}), "text")
        self.assertTrue(call.call_args[0][1]["objective"])
        with mock.patch.object(tools, "exa_call", side_effect=RuntimeError("boom")):
            self.assertTrue(tools.web_search.invoke({"query": "x"}).startswith("ERROR: "))

    def test_empty_web_search_answer_is_no_results(self):
        tools = self.tools
        with mock.patch.object(tools, "web_search_text", return_value=""):
            self.assertEqual(tools.web_search.invoke({"query": "x"}), "NO RESULTS")

    def test_retry_after_is_clamped_for_a_saturated_free_tier(self):
        tools = self.tools
        # a real free-tier Exa answer carries Retry-After in the tens of thousands of seconds
        self.assertEqual(tools.backoff_delay(0, retry_after=50_638, retry_after_max=120.0, jitter=lambda: 0.0), 120.0)
        self.assertEqual(tools.backoff_delay(0, retry_after=7, retry_after_max=120.0, jitter=lambda: 0.0), 7.0)

    def test_response_detail_extracts_the_jsonrpc_message(self):
        tools = self.tools
        body = json.dumps({"jsonrpc": "2.0", "error": {"code": -32000,
                                                      "message": "You've hit Exa's free MCP rate limit."}})
        response = mock.Mock(text=body)
        self.assertIn("free MCP rate limit", tools._response_detail(response))
        self.assertEqual(tools._response_detail(mock.Mock(text="")), "")
        self.assertLessEqual(len(tools._response_detail(mock.Mock(text="x" * 5000))), 300)

    def test_response_detail_never_leaks_a_key(self):
        tools = self.tools
        secret = "exa-secret-abcdef0123456789"
        with mock.patch.dict(os.environ, {"EXA_API_KEY": secret}):
            detail = tools._response_detail(mock.Mock(text=f"bad key?exaApiKey={secret} and {secret}"))
        self.assertNotIn(secret, detail)
        self.assertIn("***", detail)

    def test_exa_429_body_reaches_the_agent_as_an_actionable_error(self):
        tools = self.tools
        body = json.dumps({"jsonrpc": "2.0", "error": {"code": -32000,
                                                      "message": "Create API key at https://dashboard.exa.ai/api-keys"}})

        def responder(request):
            return tools.httpx.Response(429, text=body, headers={"Retry-After": "50638"}, request=request)

        with self.make_client(responder) as client, \
                mock.patch.object(tools, "exa_call", side_effect=RuntimeError("rate limited: create a key")):
            answer = tools.web_search.invoke({"query": "world models"})
        self.assertTrue(answer.startswith("ERROR: "), answer)

        with self.make_client(responder) as client:
            with self.assertRaises(tools.RetryableError) as caught:
                tools.exa_call("web_search_exa", {"query": "q", "objective": "o"}, client=client,
                               attempts=2, sleep=FakeSleep(), pace=False)
        self.assertIn("dashboard.exa.ai", str(caught.exception))   # the server's fix reaches the agent

    # ---- secrets ----------------------------------------------------------------------------------
    def test_exa_api_key_rides_in_the_url_but_is_redacted_from_errors(self):
        tools = self.tools
        secret = "exa-secret-key-0123456789abcdef"
        with mock.patch.dict(os.environ, {"EXA_API_KEY": secret}):
            self.assertIn(f"exaApiKey={secret}", tools._exa_endpoint())
            self.assertEqual(tools._secrets(), (secret,))
            self.assertNotIn(secret, tools.redact(f"ConnectError: https://mcp.exa.ai/mcp?exaApiKey={secret}"))
            self.assertIn("***", tools.redact(f"ConnectError: https://mcp.exa.ai/mcp?exaApiKey={secret}"))
            # even a bare key in an exception message is blanked
            self.assertNotIn(secret, tools._error_text(Exception(f"header leak {secret}")))

    def test_redaction_keeps_ordinary_text_intact(self):
        tools = self.tools
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXA_API_KEY", None)
            self.assertEqual(tools.redact("plain error text"), "plain error text")

    # ---- web_fetch --------------------------------------------------------------------------------
    def test_web_fetch_uses_the_urls_array(self):
        tools = self.tools
        seen = {}

        def responder(request):
            body = json.loads(request.content.decode())
            seen["arguments"] = body["params"]["arguments"]
            seen["tool"] = body["params"]["name"]
            return tools.httpx.Response(200, text=mcp_text("page text"), request=request)

        with self.make_client(responder) as client:
            text = tools.exa_call("web_fetch_exa", {"urls": ["https://arxiv.org/abs/1803.10122"]},
                                  client=client, pace=False)
        self.assertEqual(text, "page text")
        self.assertEqual(seen["tool"], "web_fetch_exa")
        self.assertEqual(seen["arguments"]["urls"], ["https://arxiv.org/abs/1803.10122"])

    def test_web_fetch_truncates_a_long_page(self):
        tools = self.tools
        with mock.patch.object(tools, "exa_call", return_value="y" * 20_000):
            page = tools.web_fetch_text("https://arxiv.org/abs/1803.10122")
        self.assertLessEqual(len(page), tools.FETCH_CHARS + len("\n[truncated]"))
        self.assertTrue(page.endswith("[truncated]"), page[-40:])

    def test_web_fetch_keeps_a_short_page_untouched(self):
        tools = self.tools
        with mock.patch.object(tools, "exa_call", return_value="short page"):
            self.assertEqual(tools.web_fetch_text("https://example.org"), "short page")

    def test_web_fetch_rejects_a_non_http_url_without_calling_the_network(self):
        tools = self.tools
        with mock.patch.object(tools, "exa_call", side_effect=AssertionError("must not be called")):
            answer = tools.web_fetch.invoke({"url": "file:///etc/passwd"})
        self.assertTrue(answer.startswith("ERROR: "), answer)

    def test_web_fetch_error_is_a_string(self):
        tools = self.tools
        with mock.patch.object(tools, "exa_call", side_effect=RuntimeError("HTTP 500")):
            self.assertTrue(tools.web_fetch.invoke({"url": "https://example.org"}).startswith("ERROR: "))

    # ---- SSE / payload helpers --------------------------------------------------------------------
    def test_sse_payload_picks_the_data_line_and_ignores_keepalive(self):
        tools = self.tools
        payload = tools._sse_payload(": keepalive\n\nevent: message\ndata: {\"a\": 1}\n\n")
        self.assertEqual(payload, {"a": 1})
        self.assertEqual(tools._sse_payload('{"a": 2}'), {"a": 2})

    def test_sse_payload_raises_when_there_is_no_json(self):
        tools = self.tools
        with self.assertRaises(ValueError):
            tools._sse_payload("data: not json at all")

    def test_exa_text_joins_only_text_parts(self):
        tools = self.tools
        result = {"content": [{"type": "text", "text": "one"}, {"type": "image", "data": "zz"},
                              {"type": "text", "text": "two"}]}
        self.assertEqual(tools._exa_text(result), "one\ntwo")

    def test_rate_limit_detector_reads_various_meta_spellings(self):
        tools = self.tools
        self.assertTrue(tools._exa_rate_limited({"_meta": {"exaRateLimited": True}}, ""))
        self.assertTrue(tools._exa_rate_limited({"_meta": {"rateLimit": "hit"}}, ""))
        self.assertTrue(tools._exa_rate_limited({}, "Rate Limit exceeded"))
        self.assertFalse(tools._exa_rate_limited({"_meta": {"ok": True}}, "normal page text"))

    def test_int_parameter_is_clamped_and_survives_junk(self):
        tools = self.tools
        self.assertEqual(tools._int_parameter("7", 5, 1, 30), 7)
        self.assertEqual(tools._int_parameter("nonsense", 5, 1, 30), 5)
        self.assertEqual(tools._int_parameter(9999, 5, 1, 30), 30)
        self.assertEqual(tools._int_parameter(-3, 5, 1, 30), 1)


if __name__ == "__main__":
    unittest.main()
