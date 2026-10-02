"""
Tests for the management-outlook extractor.
Offline: transcripts and model replies are stubs, so no network or API quota.
Run with: pytest tests/test_outlook.py -v
"""
import json
from datetime import date

import pytest

from src.data_pipeline.fetchers.transcript_fetcher import Transcript
from src.facts.outlook import OutlookExtractor, select_passages, verify_quote

Q1 = Transcript("TCS", "Jul 2026", "https://bse/q1.pdf", date(2026, 6, 30), 20,
                "Good evening. We remain conﬁdent that TCS is well positioned to convert demand "
                "into stronger growth as client spending normalizes. Our operating margin was 24%. "
                "Going forward, we target at least 1% of our employee base in the new model. "
                "Thank you.", "")
Q4 = Transcript("TCS", "Apr 2026", "https://bse/q4.pdf", date(2026, 3, 31), 30,
                "We remain committed to returning substantial free cash flows to our shareholders "
                "while judiciously investing in growth. The weather was nice.", "")


class _Fetcher:
    def __init__(self, transcripts):
        self.transcripts = transcripts

    def latest(self, symbol, n=2):
        return self.transcripts[:n]


class _LLM:
    def __init__(self, points):
        self.reply, self.calls = json.dumps({"points": points}), 0

    def complete(self, **kwargs):
        self.calls += 1
        return "Here you go:\n" + self.reply


def _point(quote, source="Q1 FY27", category="growth", statement="Management expects growth."):
    return {"category": category, "statement": statement, "quote": quote, "source": source}


class TestVerifyQuote:

    def test_exact_and_normalised_matches(self):
        text = Q1.text
        assert verify_quote("We remain confident that TCS is well positioned to convert demand", text)
        assert verify_quote("we REMAIN  confident that TCS is well positioned", text)    # case, spacing

    def test_elided_quote_needs_every_part(self):
        assert verify_quote("We remain confident that TCS ... target at least 1% of our employee base", Q1.text)
        assert not verify_quote("We remain confident that TCS ... target at least 5% of our employee base", Q1.text)

    def test_invented_or_too_short_quotes_fail(self):
        assert not verify_quote("We will double revenue by FY30 across every market", Q1.text)
        assert not verify_quote("operating margin", Q1.text)


class TestSelectPassages:

    def test_keeps_forward_looking_sentences_with_context(self):
        text = " ".join(f"Filler sentence number {i}." for i in range(50)) + \
               " Going forward, we expect margins to improve. Filler end."
        out = select_passages(text, budget=50)
        assert "Going forward, we expect margins to improve." in out
        assert "Filler sentence number 49." in out          # neighbour kept for context
        assert "Filler sentence number 10." not in out


class TestExtractor:

    def _extractor(self, tmp_path, points, transcripts=(Q1, Q4)):
        llm = _LLM(points)
        return OutlookExtractor(fetcher=_Fetcher(list(transcripts)), llm=llm, cache_dir=tmp_path), llm

    def test_unverifiable_points_are_dropped(self, tmp_path):
        ex, _ = self._extractor(tmp_path, [
            _point("We remain confident that TCS is well positioned to convert demand into stronger growth"),
            _point("We will double revenue by FY30 across every single market we serve"),   # invented
            _point("we target at least 1% of our employee base in the new model", category="vibes"),
        ])
        o = ex.outlook("TCS")
        assert [p.statement for p in o.points] == ["Management expects growth."]
        assert o.dropped == 2
        assert o.points[0].url == Q1.url and "ﬁ" not in o.points[0].quote

    def test_misattributed_source_is_corrected(self, tmp_path):
        ex, _ = self._extractor(tmp_path, [
            _point("returning substantial free cash flows to our shareholders while judiciously investing",
                   source="Q1 FY27", category="capital_allocation"),
        ])
        p = ex.outlook("TCS").points[0]
        assert (p.source, p.url) == ("Q4 FY26", Q4.url)

    def test_second_request_uses_the_cache(self, tmp_path):
        ex, llm = self._extractor(tmp_path, [
            _point("We remain confident that TCS is well positioned to convert demand into stronger growth"),
        ])
        first = ex.outlook("TCS")
        again = ex.outlook("TCS")
        assert llm.calls == 1
        assert [p.quote for p in again.points] == [p.quote for p in first.points]

    def test_no_transcripts_means_no_model_call(self, tmp_path):
        ex, llm = self._extractor(tmp_path, [], transcripts=())
        o = ex.outlook("TCS")
        assert not o.available and "No recent earnings-call transcript" in o.note
        assert llm.calls == 0

    def test_model_failure_is_reported_and_not_cached(self, tmp_path):
        class _Broken:
            calls = 0

            def complete(self, **kwargs):
                _Broken.calls += 1
                raise RuntimeError("quota exhausted")

        ex = OutlookExtractor(fetcher=_Fetcher([Q1]), llm=_Broken(), cache_dir=tmp_path)
        assert "failed" in ex.outlook("TCS").note
        ex.outlook("TCS")
        assert _Broken.calls == 2
