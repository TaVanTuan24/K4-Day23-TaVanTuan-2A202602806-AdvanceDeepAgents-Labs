"""Pre-submission audit: RUBRIC structure + metadata for every delivered report. Read-only."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from check_citations import check  # noqa: E402

REQUIRED_HEADINGS = ["## TL;DR", "## Background", "## Trends and open problems", "## References"]
FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}
CITATION = re.compile(r"\[\d+\]")


def topics():
    text = (ROOT / "topics.md").read_text(encoding="utf-8")
    return [m.group(1).strip() for m in re.finditer(r"^\d+\. (.+)$", text, re.M)]


def main():
    failures = []
    print(f"{'topic':<62} {'sub':>3} {'src':>4} {'fam':>3} {'thm':>3} {'words':>6} {'cites':>5}  result")
    for topic in topics():
        meta_path = None
        for path in sorted((ROOT / "reports").glob("*.meta.json")):
            if str(json.loads(path.read_text(encoding="utf-8")).get("topic", "")).strip().lower() == topic.lower():
                meta_path = path
                break
        if meta_path is None:
            failures.append(f"{topic}: no meta.json")
            continue
        stem = meta_path.name[: -len(".meta.json")]
        report_path = ROOT / "reports" / f"{stem}.md"
        sources_path = ROOT / "reports" / f"{stem}.sources.json"
        report = report_path.read_text(encoding="utf-8")
        sources = json.loads(sources_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        problems = list(check(report, sources))
        for heading in REQUIRED_HEADINGS:
            if heading not in report:
                problems.append(f"missing heading {heading!r}")
        body = report.split("## References")[0]
        themes = [line for line in report.splitlines()
                  if line.startswith("## ") and line not in REQUIRED_HEADINGS
                  and not re.match(r"^## (TL;DR|Background|Trends and open problems)\s*$", line)]
        if not 3 <= len(themes) <= 6:
            problems.append(f"{len(themes)} theme sections (need 3-6)")
        if not report.startswith("# "):
            problems.append("missing top-level title")
        if int(meta.get("subagent_calls", 0)) < 3:
            problems.append(f"subagent_calls={meta.get('subagent_calls')} (need >= 3)")
        families = set(meta.get("source_families", []))
        if len(families & FAMILIES) < 3:
            problems.append(f"only {len(families & FAMILIES)} families in meta.json")
        actual = {entry.get("source") for entry in sources}
        if families != actual:
            problems.append(f"meta families {sorted(families)} != sources.json {sorted(actual)}")
        for entry in sources:
            url, family = entry.get("url", ""), entry.get("source")
            if family == "arxiv" and not url.startswith("https://arxiv.org/abs/"):
                problems.append(f"[{entry.get('n')}] arxiv family but url {url}")
            if family in ("hf-daily", "hf-search") and not url.startswith("https://huggingface.co/papers/"):
                problems.append(f"[{entry.get('n')}] {family} family but url {url}")
            if not url.startswith(("http://", "https://")):
                problems.append(f"[{entry.get('n')}] url is not http(s): {url}")
        if "\ufffd" in report:
            problems.append("report contains U+FFFD replacement characters")
        print(f"{stem[:60]:<62} {meta.get('subagent_calls', 0):>3} {len(sources):>4} "
              f"{len(families & FAMILIES):>3} {len(themes):>3} {len(body.split()):>6} "
              f"{len(CITATION.findall(body)):>5}  {'OK' if not problems else 'PROBLEMS'}")
        for problem in problems:
            print(f"      - {problem}")
        if problems:
            failures.append(stem)
    print("\nAUDIT:", "ALL OK" if not failures else f"{len(failures)} report(s) with problems: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
