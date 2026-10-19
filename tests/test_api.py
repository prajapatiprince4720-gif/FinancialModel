"""
Tests for the research service layer and the FastAPI app.

Offline by design: the live price lookup and the risk-free rate are pinned, so
these run against the local Screener cache only and never spend LLM quota.
Run with: pytest tests/test_api.py -v
"""
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from src.reports.disclosure import is_report_trustworthy
from src.services.research_service import ResearchService, jsonable, normalise_symbol
from src.utils.macro_rates import RFContext

FIXED_RF = RFContext(rate=0.068, source="manual", as_of="2026-10-01", is_live=True)
FIXED_CMP = 1500.0


@pytest.fixture(scope="module")
def service():
    """One service for the module, so universe-wide scoring runs only once."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.valuation.dcf_engine._fetch_cmp", lambda symbol: FIXED_CMP)
        svc = ResearchService()
        svc._rf_context = FIXED_RF
        yield svc


@pytest.fixture
def client(service, monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "get_service", lambda: service)
    monkeypatch.setattr(api_main, "_EXPORT_DIR", tmp_path)
    api_main._hits.clear()
    return TestClient(api_main.app)


# ── Pure helpers ──────────────────────────────────────────────────────────────

class TestHelpers:

    @pytest.mark.parametrize("raw", ["tcs", "TCS.NS", " tcs.bo ", "TCS"])
    def test_normalise_symbol(self, raw):
        assert normalise_symbol(raw) == "TCS"

    def test_normalise_symbol_empty(self):
        assert normalise_symbol(None) == ""

    def test_jsonable_converts_numpy_and_non_finite(self):
        out = jsonable({"a": np.float64(1.5), "b": [np.int64(2), float("nan")],
                        "c": float("inf"), "d": "x"})
        assert out == {"a": 1.5, "b": [2, None], "c": None, "d": "x"}
        assert type(out["a"]) is float and type(out["b"][0]) is int

    def test_trustworthy_when_live_and_fresh(self):
        dcf = SimpleNamespace(rf_is_live=True, verdict="FAIR")
        ok, warns = is_report_trustworthy(dcf, {"age_days": 10})
        assert ok and warns == []

    def test_untrustworthy_flags_each_problem(self):
        dcf = SimpleNamespace(rf_is_live=False, verdict="INSUFFICIENT_DATA")
        ok, warns = is_report_trustworthy(dcf, {"age_days": 200})
        assert not ok
        assert len(warns) == 3


# ── Service layer ─────────────────────────────────────────────────────────────

class TestResearchService:

    def test_universe_is_nifty50(self, service):
        uni = service.universe()
        assert len(uni) == 50
        assert all(u["ticker"].endswith(".NS") for u in uni)

    def test_is_covered(self, service):
        assert service.is_covered("hdfcbank.ns")
        assert not service.is_covered("NOTASTOCK")

    def test_report_unknown_symbol_raises(self, service):
        with pytest.raises(KeyError):
            service.report("NOTASTOCK")

    def test_report_hdfcbank(self, service):
        r = service.report("HDFCBANK")
        assert r["symbol"] == "HDFCBANK"
        assert r["macro"]["rf_pct"] == pytest.approx(6.8)
        assert r["valuation"]["cmp"] == FIXED_CMP
        assert r["score"] is not None and 0 <= r["score"]["composite"] <= 100
        assert r["financials"]["years"]
        # Warnings and the trust flag must agree.
        assert r["trustworthy"] == (r["warnings"] == [])

    def test_leaderboard_sorted_descending(self, service):
        rows = service.leaderboard(limit=10)
        assert len(rows) == 10
        scores = [r["composite"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_scores_cached_until_cache_changes(self, service):
        assert service.scores() is service.scores()


# ── HTTP API ──────────────────────────────────────────────────────────────────

class TestAPI:

    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and body["universe"] == 50

    def test_security_headers(self, client):
        r = client.get("/api/disclaimer")
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert "default-src 'self'" in r.headers["Content-Security-Policy"]

    def test_stocks(self, client):
        assert len(client.get("/api/stocks").json()["stocks"]) == 50

    def test_leaderboard_limit_validated(self, client):
        assert client.get("/api/leaderboard?limit=0").status_code == 422
        assert len(client.get("/api/leaderboard?limit=5").json()["rows"]) == 5

    def test_report_ok(self, client):
        r = client.get("/api/report/hdfcbank")
        assert r.status_code == 200
        assert r.json()["symbol"] == "HDFCBANK"

    def test_report_unknown_is_404(self, client):
        assert client.get("/api/report/NOTASTOCK").status_code == 404

    def test_faq_unknown_is_404(self, client):
        assert client.get("/api/faq/NOTASTOCK").status_code == 404

    def test_ask_validates_question_length(self, client):
        assert client.post("/api/ask", json={"question": "hi"}).status_code == 422

    def test_ask_unknown_symbol_is_404(self, client, monkeypatch):
        monkeypatch.setattr("src.agents.investor_qa_agent.InvestorQAAgent",
                            lambda: SimpleNamespace(ask_detailed=lambda q, ticker="": {"answer": "x", "source": "llm"}))
        r = client.post("/api/ask", json={"question": "Is it cheap?", "symbol": "NOTASTOCK"})
        assert r.status_code == 404

    def test_ask_rate_limited(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "_LLM_RATE_LIMIT", 2)
        monkeypatch.setattr("src.agents.investor_qa_agent.InvestorQAAgent",
                            lambda: SimpleNamespace(ask_detailed=lambda q, ticker="": {"answer": "stub answer", "source": "llm"}))
        payload = {"question": "What is the moat?", "symbol": "TCS"}
        first = client.post("/api/ask", json=payload)
        assert first.status_code == 200
        assert first.json()["answer"] == "stub answer"
        assert client.post("/api/ask", json=payload).status_code == 200
        blocked = client.post("/api/ask", json=payload)
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers

    def test_ask_llm_failure_is_502(self, client, monkeypatch):
        def boom(q, ticker=""):
            raise RuntimeError("quota exhausted")
        monkeypatch.setattr("src.agents.investor_qa_agent.InvestorQAAgent",
                            lambda: SimpleNamespace(ask_detailed=boom))
        r = client.post("/api/ask", json={"question": "What is the moat?"})
        assert r.status_code == 502

    def test_export_docx(self, client):
        import docx

        r = client.get("/api/export/HDFCBANK?fmt=docx")
        assert r.status_code == 200
        assert "attachment" in r.headers["Content-Disposition"]
        assert r.headers["Cache-Control"] == "no-store"
        text = "\n".join(p.text for p in docx.Document(BytesIO(r.content)).paragraphs)
        assert "SEBI" in text

    def test_export_rejects_bad_format(self, client):
        assert client.get("/api/export/HDFCBANK?fmt=xlsx").status_code == 422


class TestShareCount:

    def test_loss_year_does_not_crash(self):
        """BHARTIARTL-shaped EPS (0.74 → −55) used to raise IndexError."""
        from src.valuation.dcf_engine import DCFEngine
        e = DCFEngine(rf_context=FIXED_RF, quiet=True)
        np_ser  = [1688.0, -30664.0, -12364.0, 8305.0, 37481.0]
        eps_ser = [0.74, -55.06, -25.63, 7.23, 58.85]
        assert e._shares(np_ser, eps_ser) == pytest.approx(37481.0 / 58.85)

    def test_zero_eps_returns_zero(self):
        from src.valuation.dcf_engine import DCFEngine
        e = DCFEngine(rf_context=FIXED_RF, quiet=True)
        assert e._shares([100.0], [0.0]) == 0.0


class TestHistoricalPrice:
    """A historical view must never be valued against today's live price."""

    @pytest.fixture(autouse=True)
    def _no_live_price(self, monkeypatch):
        def _fail(symbol):
            raise AssertionError("historical valuation fetched a live price")
        monkeypatch.setattr("src.valuation.dcf_engine._fetch_cmp", _fail)

    # One company per valuation path: value-driver, bank, insurer.
    @pytest.mark.parametrize("symbol", ["TCS", "HDFCBANK", "HDFCLIFE"])
    def test_without_price_outputs_stay_empty(self, symbol):
        from datetime import date
        from src.data import load_company
        from src.valuation.dcf_engine import DCFEngine
        cd = load_company(symbol)
        if cd is None:
            pytest.skip(f"{symbol} not in local cache")
        r = DCFEngine(rf_context=FIXED_RF, quiet=True).value(symbol, data=cd.as_of(date(2022, 6, 1)))
        assert r.cmp is None
        assert r.margin_of_safety is None

    @pytest.mark.parametrize("symbol", ["TCS", "HDFCBANK", "HDFCLIFE"])
    def test_supplied_price_is_used(self, symbol):
        from datetime import date
        from src.data import load_company
        from src.valuation.dcf_engine import DCFEngine
        cd = load_company(symbol)
        if cd is None:
            pytest.skip(f"{symbol} not in local cache")
        r = DCFEngine(rf_context=FIXED_RF, quiet=True).value(symbol, data=cd.as_of(date(2022, 6, 1), price=1234.0))
        assert r.cmp == 1234.0
