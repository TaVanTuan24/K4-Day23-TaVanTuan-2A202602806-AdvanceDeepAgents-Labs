"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.

The six rules of GUIDE.md part 4 are implemented strictly: this validator NEVER loosens them (RUBRIC 4.1
grades the report with the teacher's validator, and a loosened copy loses points in 3.1 and 6.3).
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

# --- patterns -------------------------------------------------------------------------------------
# Citation group in the body: [1]  [1, 2]  [1-3]  [2–3].  A negative lookahead drops the link form
# [1](https://...) and the citation definitions of the reference list are never scanned (body only).
_CITATION = re.compile(r"\[([0-9]+(?:[ \t]*[,;][ \t]*[0-9]+|[ \t]*[-\u2013\u2014][ \t]*[0-9]+)*)\](?!\()")
_URL = re.compile(r"https?://[^\s)\],;\"'<>]+")
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^[ \t]*\[([0-9]+)\]")
# Everything between two backticks is a code span / fenced code block: [n] inside one is not a citation.
_CODE_SPAN = re.compile(r"(`[^`\n]*`|```.*?```)", re.DOTALL)


def _extract_group(group):
    """Expand one citation group into the list of numbers it cites.

    "[1]" -> [1];  "[1, 2]" -> [1, 2];  "[1-3]" -> [1, 2, 3].  A reversed or absurd span
    (e.g. "[9-2]") falls back to its two endpoints instead of producing a huge range.
    """
    numbers = []
    for part in re.split(r"[ \t]*[,;][ \t]*", group):
        span = re.fullmatch(r"([0-9]+)[ \t]*[-\u2013\u2014][ \t]*([0-9]+)", part)
        if span:
            first, last = int(span.group(1)), int(span.group(2))
            numbers.extend(range(first, last + 1) if 0 <= last - first <= 200 else [first, last])
        else:
            numbers.append(int(part))
    return numbers


def _sub_keep_numbers(pattern, replacement, text):
    """re.sub that keeps the character count stable, so line offsets stay valid."""
    return pattern.sub(lambda match: replacement * len(match.group(0)), text)


def _blank_code(text):
    """Replace every code span with spaces (newlines kept) so [n] inside code is never a citation."""
    parts = _CODE_SPAN.split(text)
    for index in range(1, len(parts), 2):
        parts[index] = "".join("\n" if char == "\n" else " " for char in parts[index])
    return "".join(parts)


def _blank_links(text):
    """Blank the markdown link form [text](url): [1](url) is a link, not a citation (GUIDE rule 7)."""
    return _sub_keep_numbers(re.compile(r"\[[^\]\n]*\]\([^)\n]*\)"), " ", text)


def _strip_trailing_punctuation(url):
    """A URL ending a sentence may carry ".", ",", ")" ... Sentence punctuation is not part of the URL."""
    return url.rstrip(".,;:\u201d\u2019)\"'")


def _split_body(report_text):
    """(body, reference_section, heading_count).  Body = text before the LAST `## References` heading,
    exactly like finalize_citations.py splits it."""
    headings = list(_REF_HEADING.finditer(report_text))
    if not headings:
        return report_text.rstrip(), "", 0
    start = headings[-1].start()
    return report_text[:start].rstrip(), report_text[start:], len(headings)


def _source_number(entry):
    """The source number when it is a real integer, else None (rule 2: `n` must be an integer)."""
    if not isinstance(entry, dict):
        return None
    number = entry.get("n")
    if isinstance(number, bool) or not isinstance(number, int):
        return None
    return number


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).  Implements GUIDE.md part 4, rules 1-7.

    Rule 1  sources must be a non-empty list.
    Rule 2  every source: `n` integer, `url` http(s), no duplicate url.
    Rule 3  the report must carry a `## References` heading; only the text BEFORE it is the body.
    Rule 4  every [n] in the body exists in sources AND every source is cited at least once.
    Rule 5  exactly one reference line `[n] ...` per source: none missing, no number twice, no
            number that is not a source.
    Rule 6  every reference line holds exactly one URL and it equals that source's url
            (a line bundling several sources under one number is a problem).
    Rule 7  grouped citations [1, 2] / [1-3] are expanded; [n] inside code spans and markdown links
            [n](url) is not counted.
    """
    problems = []

    # ---- rules 1-2: shape of sources.json --------------------------------------------------------
    if not isinstance(sources, list):
        return ["sources.json must be a JSON list of source objects"]
    if not sources:
        return ["no sources in sources.json"]

    known_numbers, urls_by_number, urls_seen = set(), {}, {}
    for index, entry in enumerate(sources):
        number = _source_number(entry)
        if number is None:
            problems.append(f"source #{index}: 'n' must be an integer (got {entry.get('n')!r})"
                            if isinstance(entry, dict) else f"source #{index} is not an object (got {type(entry).__name__})")
            continue
        if number in known_numbers:
            problems.append(f"source [{number}]: duplicate number in sources.json")
        known_numbers.add(number)
        if not isinstance(entry, dict):
            continue
        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source [{number}]: url must start with http:// or https:// (got {url!r})")
            continue
        if url in urls_seen:
            problems.append(f"source [{number}]: duplicate url also used by [{urls_seen[url]}] ({url})")
        else:
            urls_seen[url] = number
        urls_by_number[number] = url

    # ---- rule 3: body vs reference list ----------------------------------------------------------
    body, reference_section, heading_count = _split_body(report_text)
    if heading_count == 0:
        problems.append("report has no '## References' heading")
    elif heading_count > 1:
        problems.append(f"report has {heading_count} '## References' headings (must be exactly one)")

    # ---- rules 4 and 7: citations in the body -----------------------------------------------------
    cited = set()
    for match in _CITATION.finditer(_blank_links(_blank_code(body))):
        cited.update(_extract_group(match.group(1)))
    for number in sorted(cited - known_numbers):
        problems.append(f"[{number}] is cited in the report body but missing from sources.json")
    for number in sorted(known_numbers - cited):
        problems.append(f"source [{number}] is never cited in the report body")

    # ---- rules 5 and 6: the reference lines ------------------------------------------------------
    reference_keys = []
    for line in reference_section.splitlines():
        line = _blank_code(line)                      # a URL in a code span is not the reference URL
        match = _REF_LINE.match(line)
        if not match:
            continue
        key = int(match.group(1))
        reference_keys.append(key)
        found = [_strip_trailing_punctuation(url) for url in _URL.findall(line)]
        if len(found) != 1:
            problems.append(f"reference [{key}] must hold exactly one URL (found {len(found)})")
            continue
        expected = urls_by_number.get(key)
        if expected is None:
            problems.append(f"reference [{key}] points at a number that is not a source")
        elif found[0] != expected:
            problems.append(f"reference [{key}] url {found[0]!r} != sources.json url {expected!r}")

    reference_seen = set()
    for key in reference_keys:
        if key in reference_seen:
            problems.append(f"reference [{key}] appears more than once (one line per source)")
        reference_seen.add(key)
        if key not in known_numbers:
            problems.append(f"reference [{key}] is not a source number")
    for number in sorted(known_numbers - set(reference_keys)):
        problems.append(f"reference [{number}] is missing (every source needs exactly one line)")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
