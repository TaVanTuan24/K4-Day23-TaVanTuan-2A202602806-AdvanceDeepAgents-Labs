"""Pre-submission audit: is every source URL real? (RUBRIC 4.2 - a fabricated URL zeroes that report.)

Bounded concurrency, short timeouts, one request per unique URL. Classification:
  ok        2xx / 3xx                    -> the page exists
  guarded   401 / 403 / 405 / 418 / 429  -> the server answered, it just refuses bots: NOT a fake URL
  missing   404 / 410                    -> a dead link (suspicious: a fabricated URL usually looks like this)
  dns       name resolution / connect failure
  error     anything else (timeout, TLS, 5xx)
Never writes anything.
"""
import concurrent.futures
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 20
WORKERS = 8
UA = "Mozilla/5.0 (compatible; lab-citation-audit/1.0; +https://github.com/TaVanTuan24)"


def probe(url):
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return url, "ok", response.status
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 405, 418, 429):
            return url, "guarded", exc.code
        if exc.code in (404, 410):
            return url, "missing", exc.code
        return url, "error", exc.code
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc)).lower()
        if "name or service not known" in reason or "nodename nor servname" in reason or "getaddrinfo" in reason:
            return url, "dns", 0
        return url, "error", 0
    except Exception as exc:  # noqa: BLE001 - a probe must never abort the audit
        return url, f"error:{type(exc).__name__}", 0


def main():
    urls = []
    for path in sorted((ROOT / "reports").glob("*.sources.json")):
        for entry in json.loads(path.read_text(encoding="utf-8")):
            urls.append((path.name[: -len(".sources.json")], entry.get("n"), entry.get("url")))
    unique = sorted({url for _, _, url in urls})
    print(f"reports: {len({stem for stem, _, _ in urls})} | source entries: {len(urls)} | unique URLs: {len(unique)}")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for url, verdict, code in pool.map(probe, unique):
            results[url] = (verdict, code)
    counts = Counter(verdict.split(":")[0] for verdict, _ in results.values())
    print("verdicts:", dict(counts))
    bad = {url: value for url, value in results.items() if value[0] in ("missing", "dns") or value[0].startswith("error")}
    if bad:
        print(f"\n{len(bad)} URL(s) needing a look:")
        for url, (verdict, code) in sorted(bad.items()):
            owners = [f"{stem}[{n}]" for stem, n, u in urls if u == url]
            print(f"  {verdict:14s} http={code:<4} {url}\n      used by: {', '.join(owners)}")
    else:
        print("\nno missing/DNS-failure URLs: every source URL resolves")
    by_stem = Counter()
    for stem, n, url in urls:
        if results[url][0] in ("missing", "dns") or results[url][0].startswith("error"):
            by_stem[stem] += 1
    if by_stem:
        print("per report:", dict(by_stem))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
