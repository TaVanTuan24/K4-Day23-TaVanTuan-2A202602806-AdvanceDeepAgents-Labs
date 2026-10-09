"""Offline tests for check_citations.check() - the six GUIDE part-4 rules.

This is the file the teacher re-runs with THEIR OWN validator, so the tests below pin every rule
strictly: any loosening here is a defect, not a convenience.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_citations import check  # noqa: E402  (repo root on sys.path)

SOURCES = [
    {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "World Models", "date": "2025-01-02", "source": "arxiv"},
    {"n": 2, "id": "2503.00002", "url": "https://huggingface.co/papers/2503.00002", "title": "Latent Dynamics", "date": "2025-03-04", "source": "hf-search"},
    {"n": 3, "id": "https://example.org/post", "url": "https://example.org/post", "title": "A blog post", "date": "2025-02-10", "source": "web"},
]

GOOD_REPORT = """# Survey about world models

## TL;DR
- World models learn a compressed simulator of the environment [1].
- Recent work scales latent dynamics to video [2].

## Background
World models have been studied for years [1].

## Trends and open problems
Evaluation is still disputed [3].

## References
[1] World Models. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)
[2] Latent Dynamics. hf-search. https://huggingface.co/papers/2503.00002 (2025-03-04)
[3] A blog post. web. https://example.org/post (2025-02-10)
"""


def problems(*args):
    return check(*args)


class HappyPathTests(unittest.TestCase):
    def test_consistent_report_has_no_problems(self):
        self.assertEqual(problems(GOOD_REPORT, SOURCES), [])

    def test_grouped_citations_are_expanded(self):
        report = GOOD_REPORT.replace("[2]", "[1, 2]", 1)
        self.assertEqual(problems(report, SOURCES), [])

    def test_range_citation_is_expanded(self):
        report = GOOD_REPORT.replace("[2]", "[1-2]", 1)
        self.assertEqual(problems(report, SOURCES), [])

    def test_citation_in_a_code_span_is_not_a_citation(self):
        report = GOOD_REPORT.replace("[1]", "`[9]`", 1)   # 9 is not a source, inside code it is ignored
        self.assertEqual(problems(report, SOURCES), [])

    def test_citation_in_a_markdown_link_is_not_a_citation(self):
        report = GOOD_REPORT.replace("[1]", "[9](https://example.org/x)", 1)
        self.assertEqual(problems(report, SOURCES), [])


class Rule1EmptySourcesTests(unittest.TestCase):
    def test_empty_sources_is_a_problem(self):
        self.assertTrue(problems(GOOD_REPORT, []))

    def test_sources_must_be_a_list(self):
        self.assertTrue(problems(GOOD_REPORT, {"n": 1}))


class Rule2SourceShapeTests(unittest.TestCase):
    def test_non_integer_n_is_a_problem(self):
        for bad in ("1", 1.0, None, True):
            with self.subTest(bad=bad):
                sources = [dict(SOURCES[0], n=bad)]
                found = problems("# T\n\nx [1].\n\n## References\n[1] T. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)\n", sources)
                self.assertTrue(any("integer" in problem for problem in found), found)

    def test_url_must_be_http(self):
        sources = [dict(SOURCES[0], url="arxiv.org/abs/2501.00001")]
        found = problems(GOOD_REPORT, sources)
        self.assertTrue(any("http" in problem for problem in found), found)

    def test_duplicate_url_is_a_problem(self):
        sources = SOURCES + [dict(SOURCES[1], n=4)]
        found = problems(GOOD_REPORT, sources)
        self.assertTrue(any("duplicate url" in problem for problem in found), found)

    def test_duplicate_number_is_a_problem(self):
        sources = SOURCES + [dict(SOURCES[1])]
        found = problems(GOOD_REPORT, sources)
        self.assertTrue(any("duplicate number" in problem for problem in found), found)


class Rule3ReferencesHeadingTests(unittest.TestCase):
    def test_missing_references_heading_is_a_problem(self):
        report = GOOD_REPORT.split("## References")[0]
        found = problems(report, SOURCES)
        self.assertTrue(any("References" in problem for problem in found), found)

    def test_two_references_headings_are_a_problem(self):
        report = GOOD_REPORT.replace("## Trends and open problems", "## References\n")
        found = problems(report, SOURCES)
        self.assertTrue(any("headings" in problem for problem in found), found)

    def test_reference_numbers_are_not_counted_as_body_citations(self):
        # [3] appears only in the reference list: it must still be reported as never cited in the body
        body = GOOD_REPORT.replace("Evaluation is still disputed [3].", "Evaluation is still disputed.")
        found = problems(body, SOURCES)
        self.assertTrue(any("never cited" in problem and "[3]" in problem for problem in found), found)


class Rule4CitationCoverageTests(unittest.TestCase):
    def test_citation_without_a_source_is_a_problem(self):
        report = GOOD_REPORT.replace("for years [1].", "for years [7].")
        found = problems(report, SOURCES)
        self.assertTrue(any("missing from sources.json" in problem and "[7]" in problem for problem in found), found)

    def test_uncited_source_is_a_problem(self):
        report = GOOD_REPORT.replace("Evaluation is still disputed [3].", "Evaluation is still disputed.")
        found = problems(report, SOURCES)
        self.assertTrue(any("never cited" in problem for problem in found), found)

    def test_extra_source_without_a_reference_line_is_a_problem(self):
        sources = SOURCES + [{"n": 4, "id": "x", "url": "https://example.org/extra", "title": "Extra", "date": "2025-05-05", "source": "web"}]
        found = problems(GOOD_REPORT, sources)
        self.assertTrue(any("reference [4] is missing" in problem for problem in found), found)


class Rule5ReferenceLineTests(unittest.TestCase):
    def test_duplicated_reference_line_is_a_problem(self):
        report = GOOD_REPORT + "[3] A blog post. web. https://example.org/post (2025-02-10)\n"
        found = problems(report, SOURCES)
        self.assertTrue(any("more than once" in problem for problem in found), found)

    def test_reference_number_that_is_not_a_source_is_a_problem(self):
        report = GOOD_REPORT + "[9] Phantom. web. https://example.org/phantom (2025-01-01)\n"
        found = problems(report, SOURCES)
        self.assertTrue(any("[9]" in problem for problem in found), found)

    def test_missing_reference_line_is_a_problem(self):
        report = "\n".join(line for line in GOOD_REPORT.splitlines() if not line.startswith("[2] ")) + "\n"
        found = problems(report, SOURCES)
        self.assertTrue(any("reference [2] is missing" in problem for problem in found), found)


class Rule6BundledUrlTests(unittest.TestCase):
    def test_two_sources_under_one_number_is_a_problem(self):
        report = GOOD_REPORT.replace(
            "[2] Latent Dynamics. hf-search. https://huggingface.co/papers/2503.00002 (2025-03-04)",
            "[2] Latent Dynamics; A blog post; another. hf-search. https://huggingface.co/papers/2503.00002 "
            "https://example.org/post (2025-03-04)")
        found = problems(report, SOURCES)
        self.assertTrue(any("exactly one URL" in problem for problem in found), found)

    def test_reference_line_without_a_url_is_a_problem(self):
        report = GOOD_REPORT.replace("https://huggingface.co/papers/2503.00002", "no-url-here")
        found = problems(report, SOURCES)
        self.assertTrue(any("exactly one URL" in problem for problem in found), found)

    def test_url_mismatch_is_a_problem(self):
        report = GOOD_REPORT.replace("https://huggingface.co/papers/2503.00002", "https://huggingface.co/papers/9999.00009")
        found = problems(report, SOURCES)
        self.assertTrue(any("!= sources.json url" in problem for problem in found), found)

    def test_sentence_punctuation_after_a_url_is_tolerated(self):
        report = GOOD_REPORT.replace("(2025-02-10)", "(2025-02-10).")
        self.assertEqual(problems(report, SOURCES), [])


class FinalizerContractTests(unittest.TestCase):
    """The provided finalizer rewrites the report; the validator must accept its exact output shape."""

    def test_finalizer_style_output_is_ok(self):
        report = ("# Title\n\nA claim [1]. Another claim [2].\n\n"
                  "## References\n"
                  "[1] Some paper. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)\n"
                  "[2] Another paper. web. https://example.org/post (n.d.)\n")
        sources = [SOURCES[0], dict(SOURCES[2], n=2)]
        self.assertEqual(problems(report, sources), [])

    def test_missing_date_in_reference_is_fine(self):
        report = ("# Title\n\nClaim [1].\n\n## References\n"
                  "[1] Some paper. arxiv. https://arxiv.org/abs/2501.00001\n")
        self.assertEqual(problems(report, [SOURCES[0]]), [])


class RegexEdgeTests(unittest.TestCase):
    def test_range_citation_helper(self):
        from check_citations import _extract_group
        self.assertEqual(_extract_group("1"), [1])
        self.assertEqual(_extract_group("1, 2"), [1, 2])
        self.assertEqual(_extract_group("1-3"), [1, 2, 3])
        self.assertEqual(_extract_group("9-2"), [9, 2])       # absurd range -> endpoints only
        self.assertEqual(_extract_group("1;2"), [1, 2])

    def test_trailing_punctuation_is_not_part_of_the_url(self):
        from check_citations import _strip_trailing_punctuation
        self.assertEqual(_strip_trailing_punctuation("https://example.org/post)."), "https://example.org/post")
        self.assertEqual(_strip_trailing_punctuation("https://example.org/post"), "https://example.org/post")

    def test_blank_helpers_keep_the_line_count(self):
        from check_citations import _blank_code, _blank_links
        self.assertEqual(_blank_code("a `[9]` b").count("\n"), 0)
        self.assertEqual(_blank_code("```\n[9]\n```").count("\n"), 2)
        self.assertEqual(_blank_links("[1](https://x) tail").count("\n"), 0)


if __name__ == "__main__":
    unittest.main()
