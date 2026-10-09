"""The contract that matters most: the PROVIDED finalize_citations.py output must satisfy check_citations.check.

If these two ever disagree, a real run can end with `FINALIZED` followed by a validator failure.  The
finalizer is imported unmodified from the repo root.
"""
import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_citations import check  # noqa: E402

_spec = importlib.util.spec_from_file_location("finalize_citations_under_test",
                                               os.path.join(ROOT, "finalize_citations.py"))
finalize_citations = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(finalize_citations)

SOURCES = [
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "World Models", "date": "2025-01-02", "source": "arxiv"},
    {"n": 2, "id": "2503.00002", "url": "https://huggingface.co/papers/2503.00002", "title": "Latent Dynamics", "date": "2025-03-04", "source": "hf-search"},
    {"n": 3, "id": "https://example.org/post", "url": "https://example.org/post", "title": "A blog post", "date": "2025-02-10", "source": "web"},
    {"n": 4, "id": "2502.00004", "url": "https://arxiv.org/abs/2502.00004", "title": "Never cited", "date": "2025-02-02", "source": "arxiv"},
]

DRAFT = """# Survey about world models

## TL;DR
- Independent latent dynamics are what the field converged on [1, 2].
- Blogs agree the benchmarks are weak [3].

## Background
Foundational work [2-3] and [1].

## Trends and open problems
Evaluation is still disputed [3]. A bundled reference follows [1].

## References
[7] Hand-written garbage that the finalizer must throw away. web. https://example.org/nowhere (2025-09-09)
"""


class FinalizerThenValidatorTests(unittest.TestCase):
    def setUp(self):
        self.report, self.sources, self.problems = finalize_citations.finalize(DRAFT, [dict(s) for s in SOURCES])

    def test_finalizer_succeeds_on_a_draft_with_grouped_and_unused_sources(self):
        self.assertEqual(self.problems, [])                       # source 4 is uncited: dropped, not an error

    def test_validator_accepts_the_finalized_report(self):
        self.assertEqual(check(self.report, self.sources), [])

    def test_finalizer_renumbers_by_first_appearance(self):
        # [1, 2] first -> the old #1 becomes [1]; [3] next -> [2]; [1] again -> still [1]; [2] -> [3]
        self.assertIn("[1][2]", self.report)
        self.assertIn("[2][3]", self.report)
        self.assertNotIn("[7]", self.report)

    def test_finalizer_drops_the_uncited_source_and_regenerates_references(self):
        self.assertEqual([entry["n"] for entry in self.sources], [1, 2, 3])
        self.assertIn("## References", self.report)
        body, references = self.report.split("## References")
        for entry in self.sources:
            self.assertIn(f"[{entry['n']}]", body)                 # every surviving source is cited in the body
            self.assertIn(entry["url"], references)
        self.assertNotIn("Never cited", self.report)
        self.assertEqual(len([line for line in references.splitlines() if line.startswith("[")]), 3)

    def test_finalizer_is_idempotent_and_still_valid(self):
        report, sources, problems = finalize_citations.finalize(self.report, self.sources)
        self.assertEqual(problems, [])
        self.assertEqual(check(report, sources), [])
        self.assertEqual(report, self.report)
        self.assertEqual(sources, self.sources)

    def test_finalizer_reports_a_citation_that_has_no_source(self):
        bad = DRAFT.replace("[1, 2]", "[1, 99]")
        _, _, problems = finalize_citations.finalize(bad, [dict(s) for s in SOURCES])
        self.assertTrue(any("99" in problem for problem in problems), problems)

    def test_finalizer_reports_a_body_with_no_citations(self):
        _, _, problems = finalize_citations.finalize("# T\n\nNo citations here.\n", [dict(s) for s in SOURCES])
        self.assertTrue(any("no citations" in problem for problem in problems), problems)

    def test_finalizer_rejects_sources_without_a_url(self):
        _, _, problems = finalize_citations.finalize(DRAFT, [{"n": 1, "title": "no url"}])
        self.assertTrue(any("no url" in problem for problem in problems), problems)

    def test_a_long_decimal_number_is_not_treated_as_a_citation_group(self):
        report, sources, _ = finalize_citations.finalize("# T\n\nScore 0.95 [1].\n", [dict(s) for s in SOURCES[:1]])
        self.assertEqual(check(report, sources), [])

    def test_a_draft_that_already_carries_references_needs_one_edit_to_converge(self):
        """Documented interaction with the PROVIDED finalizer (which is not modified here).

        The finalizer only rewrites the section after the LAST `## References` heading, so a draft that
        already carries one ends up with two headings - and the validator refuses that (rule 3).  Deleting
        the hand-written block and re-running the finalizer converges.  The lead prompt therefore tells the
        agent never to write `## References` itself, and the validator catches it if it does.
        """
        doubled = "## References\n[1] World Models. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)\n\n" + DRAFT
        report, sources, problems = finalize_citations.finalize(doubled, [dict(s) for s in SOURCES])
        self.assertEqual(problems, [])
        found = check(report, sources)
        self.assertTrue(any("headings" in problem for problem in found), found)

        cleaned = report.replace("## References\n[1] World Models. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)\n\n",
                                 "").replace("## References\n[4] Never cited.", "")
        fixed, fixed_sources, problems = finalize_citations.finalize(cleaned, sources)
        self.assertEqual(problems, [])
        self.assertEqual(check(fixed, fixed_sources), [])

    def test_validator_flags_a_report_left_with_two_headings(self):
        doubled = DRAFT + "\n## References\n[1] World Models. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)\n"
        found = check(doubled, [dict(s) for s in SOURCES])
        self.assertTrue(any("headings" in problem for problem in found), found)


if __name__ == "__main__":
    unittest.main()
