"""Streaming debug driver for one topic (development helper, NOT part of the graded lab).

`research.py` blocks inside `agent.invoke`, so a long or stalled run gives no visibility. This driver
streams LangGraph updates instead, printing one line per graph node (model turns, tool calls, middleware)
with a timestamp, so you can see exactly where a run is spending its time.

    python -u tests/_stream_run.py "survey about world model"        # run to completion
    python -u tests/_stream_run.py "survey about world model" 40     # stop after 40 graph steps

It uses the same modules as research.py and deliberately does NOT write into `reports/`.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from agents import REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, FINALIZER_PATH, WORKDIR, build_lead_agent  # noqa: E402
from model import make_model  # noqa: E402
from research import build_prompt, slugify  # noqa: E402
from sandbox import download, open_sandbox, upload  # noqa: E402

topic = sys.argv[1] if len(sys.argv) > 1 else "survey about world model"
step_budget = int(sys.argv[2]) if len(sys.argv) > 2 else 40
started = time.time()

with open_sandbox() as backend:
    print("[dev] sandbox open", flush=True)
    backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
    upload(backend, {VALIDATOR_PATH: (ROOT / "check_citations.py").read_bytes(),
                     FINALIZER_PATH: (ROOT / "finalize_citations.py").read_bytes()})
    print("[dev] scripts uploaded", flush=True)
    agent = build_lead_agent(backend, make_model())
    steps = 0
    for chunk in agent.stream({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                              config={"recursion_limit": 1000}, stream_mode="updates"):
        steps += 1
        for node, payload in (chunk or {}).items():
            messages = (payload or {}).get("messages") if isinstance(payload, dict) else None
            if messages:
                for message in messages:
                    calls = getattr(message, "tool_calls", None) or []
                    text = str(getattr(message, "content", ""))[:160].replace("\n", " ")
                    print(f"[{time.time() - started:7.1f}s] step {steps:3d} {node:22s} "
                          f"tools={[c.get('name') for c in calls]} {text}", flush=True)
            else:
                print(f"[{time.time() - started:7.1f}s] step {steps:3d} {node:22s} (no messages)", flush=True)
        if steps >= step_budget:
            print(f"[dev] stopping at the step budget ({step_budget})", flush=True)
            break
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    for path, content in files.items():
        print(f"[dev] {path}: {'MISSING' if content is None else str(len(content)) + ' bytes'}", flush=True)
    report = files.get(REPORT_PATH)
    if report:
        print("[dev] report head:", report.decode("utf-8", "replace")[:300].replace("\n", " | "), flush=True)
