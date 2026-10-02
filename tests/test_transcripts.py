"""
Tests for the earnings-call transcript fetcher.
Offline: HTML and transcript text are inline samples of real phrasings.
Run with: pytest tests/test_transcripts.py -v
"""
from datetime import date

import pytest

from src.data_pipeline.fetchers import transcript_fetcher as tf
from src.data_pipeline.fetchers.transcript_fetcher import (
    ConcallLink, Transcript, TranscriptFetcher, parse_concalls, period_end_from_text,
)

CONCALL_HTML = """
<section id="documents"><div class="concalls"><ul>
 <li><div>Jul 2026</div>
     <a class="button-chip" href="https://bse/a.pdf" title="Raw Transcript">Transcript</a>
     <button data-url="/concalls/summary/1/">AI Summary</button>
     <div class="button-chip">PPT</div></li>
 <li><div>Apr 2026</div>
     <a href="https://bse/b.pdf">Transcript</a>
     <a href="https://co/ppt.pdf">PPT</a></li>
 <li><div>Mar 2026</div><button>Transcript</button></li>
</ul></div></section>
"""


class TestParsing:

    def test_concall_links_need_a_transcript_href(self):
        links = parse_concalls(CONCALL_HTML)
        assert [(l.label, l.transcript_url, l.ppt_url) for l in links] == [
            ("Jul 2026", "https://bse/a.pdf", None),
            ("Apr 2026", "https://bse/b.pdf", "https://co/ppt.pdf"),
        ]

    @pytest.mark.parametrize("text, expected", [
        # TCS
        ("Sub: Transcript of the earnings conference call for the quarter and year ended "
         "March 31, 2026", date(2026, 3, 31)),
        # TECHM — words between "quarter" and "ended", day before month
        ("Transcript of the quarterly earnings meeting for the fourth quarter and financial "
         "year ended 31st March, 2026", date(2026, 3, 31)),
        # WIPRO
        ("“Wipro Limited Q1 FY '27 Earnings Conference Call” July 16, 2026", date(2026, 6, 30)),
        ("Infosys Limited Q2 FY26 Financial Results Press Conference", date(2025, 9, 30)),
        ("Q3 FY2026 earnings call", date(2025, 12, 31)),
        # INFY — only the results URL says which quarter
        ("hosted at https://www.infosys.com/investors/reports-filings/quarterly-results/2026- "
         "2027/q1.html", date(2026, 6, 30)),
        # an investor day is not an earnings call
        ("Sub: Transcript of the Infosys Limited Investor AI Day 2026", None),
    ])
    def test_period_end(self, text, expected):
        assert period_end_from_text(text) == expected

    def test_period_label(self):
        t = Transcript("TCS", "Jul 2026", "u", date(2026, 6, 30), 1, "", "")
        assert t.period_label == "Q1 FY27"
        assert Transcript("TCS", "Apr 2026", "u", date(2026, 3, 31), 1, "", "").period_label == "Q4 FY26"


class TestLatest:

    def _fetcher(self, tmp_path, monkeypatch, docs: dict[str, str]):
        """A fetcher whose 'downloads' return the given text per URL."""
        f = TranscriptFetcher(cache_dir=tmp_path, rate_limit_delay=0)
        links = [ConcallLink(label, url) for url, (label, _) in docs.items()]
        monkeypatch.setattr(f, "list_concalls", lambda symbol: links)
        monkeypatch.setattr(f, "_download", lambda url: url.encode())
        monkeypatch.setattr(tf, "_extract_text", lambda pdf: (docs[pdf.decode()][1], 3))
        return f

    def test_mislabelled_duplicate_is_not_counted_twice(self, tmp_path, monkeypatch):
        f = self._fetcher(tmp_path, monkeypatch, {
            "u1": ("Jul 2026", "Q1 FY27 earnings call"),
            "u2": ("Jul 2026", "Q1 FY26 earnings call"),     # the label is wrong
            "u3": ("Apr 2026", "quarter and year ended March 31, 2026"),
        })
        got = f.latest("TCS", n=2)
        assert [t.period_label for t in got] == ["Q1 FY27", "Q4 FY26"]

    def test_undated_documents_are_skipped(self, tmp_path, monkeypatch):
        f = self._fetcher(tmp_path, monkeypatch, {
            "u1": ("Feb 2026", "Investor AI Day 2026"),
            "u2": ("Jan 2026", "Q3 FY26 earnings call"),
        })
        assert [t.period_label for t in f.latest("INFY", n=2)] == ["Q3 FY26"]

    def test_cached_document_is_not_downloaded_again(self, tmp_path, monkeypatch):
        f = self._fetcher(tmp_path, monkeypatch, {"u1": ("Jul 2026", "Q1 FY27 call")})
        first = f.latest("TCS", n=1)[0]

        def _fail(url):
            raise AssertionError("downloaded a cached transcript")
        monkeypatch.setattr(f, "_download", _fail)
        again = f.latest("TCS", n=1)[0]
        assert (again.text, again.period_end, again.url) == (first.text, first.period_end, first.url)
