"""
EquityLens web API — serves the research engine to a browser.

Run it:
    uvicorn api.main:app --reload
    # then open http://127.0.0.1:8000

Why a server rather than opening an HTML file: the valuation engine, the vector
store and the LLM clients are Python, and the API keys must stay server-side. The
browser gets JSON and rendered documents, never a key.

Design notes
------------
* Endpoint handlers are plain `def`, not `async def`. They call blocking code
  (Monte Carlo, ChromaDB, LLM HTTP calls), and FastAPI runs sync handlers in a
  worker thread — whereas blocking inside `async def` would stall the event loop
  and freeze every other request.
* The LLM endpoints are rate-limited per client. Without that, a page left open
  in a loop can burn a free-tier daily quota in minutes.
* Generated documents are streamed from a temp directory and are not left lying
  around in reports/ for every browser request.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.reports.disclosure import SEBI_DISCLAIMER, SEBI_DISCLAIMER_SHORT
from src.services.research_service import get_service, normalise_symbol
from src.utils.logger import get_logger

logger = get_logger("api")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_WEB_DIR      = _PROJECT_ROOT / "web"
_EXPORT_DIR   = _PROJECT_ROOT / "reports" / "exports"

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Deliberately in-process and simple: this protects one developer's free-tier
# quota on a locally run app. A public deployment needs a shared store (Redis).
_LLM_RATE_LIMIT   = int(os.getenv("EQUITYLENS_LLM_RATE_LIMIT", "20"))   # requests
_LLM_RATE_WINDOW  = int(os.getenv("EQUITYLENS_LLM_RATE_WINDOW", "300")) # seconds
_EXPORT_RATE_LIMIT = int(os.getenv("EQUITYLENS_EXPORT_RATE_LIMIT", "30"))

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _rate_limit(key: str, limit: int, window: int = _LLM_RATE_WINDOW) -> None:
    """Raise 429 when `key` has exceeded `limit` requests in `window` seconds."""
    now = time.time()
    with _hits_lock:
        bucket = _hits[key]
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = int(window - (now - bucket[0])) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit reached ({limit} per {window}s). Retry in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


def _client_key(request: Request, scope: str) -> str:
    host = request.client.host if request.client else "unknown"
    return f"{scope}:{host}"


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EquityLens API",
    description=(
        "AI-assisted equity research for Nifty 50 companies. "
        + SEBI_DISCLAIMER_SHORT
    ),
    version="1.0.0",
)

# Same-origin in normal use; permissive only for localhost dev tooling.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """
    Conservative headers that behave identically in Safari, Chrome, Firefox and Edge.

    The CSP allows only same-origin assets, which the UI satisfies — it ships no
    CDN dependencies, so nothing here needs a third-party allowance.
    """
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'",
    )
    return response


# ── Models ────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    symbol:   str = Field(default="", max_length=20)


class ThesisRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)


# ── Meta endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness plus everything the UI needs to describe its own data quality."""
    svc = get_service()
    return {
        "status":     "ok",
        "universe":   len(svc.universe()),
        "macro":      svc.macro(),
        "data_health": svc.data_health(),
        "disclaimer": SEBI_DISCLAIMER_SHORT,
    }


@app.get("/api/disclaimer")
def disclaimer() -> dict[str, str]:
    return {"full": SEBI_DISCLAIMER, "short": SEBI_DISCLAIMER_SHORT}


@app.get("/api/macro")
def macro() -> dict[str, Any]:
    """India 10Y GSEC risk-free rate with its provenance."""
    return get_service().macro()


# ── Universe + rankings ───────────────────────────────────────────────────────

@app.get("/api/stocks")
def stocks() -> dict[str, Any]:
    """Every covered company. Cheap — no valuation runs."""
    return {"count": 50, "stocks": get_service().universe()}


@app.get("/api/leaderboard")
def leaderboard(limit: int = Query(default=50, ge=1, le=50)) -> dict[str, Any]:
    """Universe ranked by composite quality score."""
    svc = get_service()
    return {
        "rows": svc.leaderboard(limit=limit),
        "macro": svc.macro(),
        "data_health": svc.data_health(),
    }


# ── Per-company research ──────────────────────────────────────────────────────

@app.get("/api/report/{symbol}")
def report(symbol: str) -> dict[str, Any]:
    """Full valuation + score + provenance payload for one company."""
    svc = get_service()
    try:
        return svc.report(symbol)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"'{symbol}' is not in the covered Nifty 50 universe.",
        )


@app.get("/api/facts/{symbol}")
def facts(symbol: str) -> dict[str, Any]:
    """
    Every computed fact about one company: metric histories with growth rates,
    trend slopes and z-scores, market multiples, valuation, score, peer
    percentiles and data-quality caveats. No LLM involved.
    """
    from src.services.research_service import jsonable

    sheet = get_service().fact_sheet(symbol)
    if sheet is None:
        raise HTTPException(status_code=404, detail=f"'{symbol}' is not covered.")
    return jsonable(sheet.to_dict())


@app.get("/api/faq/{symbol}")
def faq(symbol: str, limit: int = Query(default=200, ge=1, le=400)) -> dict[str, Any]:
    """
    The pre-answered investor Q&A for one company.

    This is the trained knowledge base — the UI uses it to show what can be asked
    without spending an LLM call.
    """
    from src.training.faq_trainer import company_status, load_cached_pairs

    sym = normalise_symbol(symbol)
    if not get_service().is_covered(sym):
        raise HTTPException(status_code=404, detail=f"'{symbol}' is not covered.")

    status = company_status(sym)
    return {
        "symbol":   sym,
        "company":  status["company"],
        "answered": status["answered"],
        "expected": status["expected"],
        "trained":  status["trained"],
        "coverage_pct": round(status["ratio"] * 100, 1),
        "pairs":    load_cached_pairs(sym)[:limit],
    }


@app.get("/api/training-status")
def training_status_endpoint() -> dict[str, Any]:
    """Fleet-wide FAQ training progress, so the UI can flag thin coverage."""
    from src.training.faq_trainer import TRAINED_RATIO, training_status

    st  = training_status()
    per = st["per_company"]
    return {
        "trained_count": len(st["trained"]),
        "total":         st["total"],
        "threshold_pct": round(TRAINED_RATIO * 100),
        "trained":       st["trained"],
        "partial":       st["partial"],
        "missing":       st["missing"],
        "questions_remaining": sum(len(per[s]["missing"]) for s in st["pending"]),
        "coverage": {
            sym: {"answered": v["answered"], "expected": v["expected"],
                  "trained": v["trained"]}
            for sym, v in per.items()
        },
    }


# ── LLM-backed endpoints (rate limited) ───────────────────────────────────────

@app.post("/api/ask")
def ask(payload: AskRequest, request: Request) -> dict[str, Any]:
    """
    Answer an investor question, grounded in the retrieved knowledge base.

    Rate limited: each call spends LLM quota that belongs to whoever runs the server.
    """
    _rate_limit(_client_key(request, "ask"), _LLM_RATE_LIMIT)

    from src.agents.investor_qa_agent import InvestorQAAgent

    symbol = normalise_symbol(payload.symbol) if payload.symbol else ""
    ticker = f"{symbol}.NS" if symbol else ""
    if symbol and not get_service().is_covered(symbol):
        raise HTTPException(status_code=404, detail=f"'{payload.symbol}' is not covered.")

    try:
        result = InvestorQAAgent().ask_detailed(payload.question, ticker=ticker)
    except Exception as exc:
        logger.error(f"/api/ask failed: {exc}")
        raise HTTPException(status_code=502, detail=f"The language model call failed: {exc}")

    return {
        "question":   payload.question,
        "symbol":     result.get("symbol") or symbol or None,
        "answer":     result["answer"],
        # "facts" = computed from the fact sheet, no model call; "llm" = model
        # reasoning over the fact sheet; "concept" = canned explanation.
        "source":     result.get("source", "llm"),
        "disclaimer": SEBI_DISCLAIMER_SHORT,
    }


@app.post("/api/thesis")
def thesis(payload: ThesisRequest, request: Request) -> dict[str, Any]:
    """Generate the AI investment thesis for one company."""
    _rate_limit(_client_key(request, "thesis"), _LLM_RATE_LIMIT)

    svc = get_service()
    sym = normalise_symbol(payload.symbol)
    try:
        data = svc.report(sym, include_financials=False)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"'{payload.symbol}' is not covered.")

    try:
        text = _build_thesis(sym, data)
    except Exception as exc:
        logger.error(f"/api/thesis failed for {sym}: {exc}")
        raise HTTPException(status_code=502, detail=f"The language model call failed: {exc}")

    return {"symbol": sym, "thesis": text, "disclaimer": SEBI_DISCLAIMER_SHORT}


def _build_thesis(symbol: str, data: dict[str, Any]) -> str:
    """Compose the thesis prompt from the already-computed report payload."""
    from src.llm import get_llm_client

    val   = data.get("valuation", {})
    score = data.get("score") or {}
    name  = data.get("name", symbol)

    facts = [f"Composite score: {score.get('composite', 'n/a')}/100 (grade {score.get('grade', 'n/a')})"]
    if score:
        facts += [
            f"Revenue CAGR 3Y: {score.get('revenue_cagr_3y')}%",
            f"EPS CAGR 3Y: {score.get('eps_cagr_3y')}%",
            f"ROCE percentile: {score.get('roce_percentile')}",
            f"D/E: {score.get('de_ratio')}x",
            f"Piotroski F-Score: {score.get('piotroski')}/9",
        ]
    if val.get("intrinsic_value"):
        facts.append(
            f"{val.get('model')} intrinsic value: Rs {val['intrinsic_value']:,.0f} "
            f"vs CMP Rs {val.get('cmp') or 0:,.0f} "
            f"(margin of safety {val.get('margin_of_safety_pct')}%)"
        )

    prompt = (
        f"Write a concise 3-paragraph investment thesis for {name} ({symbol}), "
        f"an Indian Nifty 50 company.\n\n"
        + "\n".join(f"- {f}" for f in facts)
        + "\n\nCover: (1) business model and competitive moat, (2) financial quality "
          "and growth drivers, (3) valuation assessment and key risks. "
          "Be factual and data-driven. Use Indian conventions (Rs Crore, NSE, SEBI)."
    )
    return get_llm_client().complete(
        system_prompt=(
            "You are a senior Indian equity analyst writing for a retail investor. "
            "Be specific and balanced, and never omit the risks."
        ),
        user_message=prompt,
        temperature=0.3,
    )


# ── Document export ───────────────────────────────────────────────────────────

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")


@app.get("/api/export/{symbol}")
def export(
    symbol: str,
    request: Request,
    fmt: Literal["pdf", "docx"] = Query(default="pdf"),
    ai: bool = Query(default=False, description="Include an AI-written thesis (spends LLM quota)"),
) -> FileResponse:
    """
    Generate a downloadable research document.

    `ai=false` by default so that clicking Download never silently spends LLM
    quota; the UI asks for the thesis explicitly.
    """
    _rate_limit(_client_key(request, "export"), _EXPORT_RATE_LIMIT)

    svc = get_service()
    sym = normalise_symbol(symbol)
    if not svc.is_covered(sym):
        raise HTTPException(status_code=404, detail=f"'{symbol}' is not covered.")

    from config.nifty50_tickers import NIFTY50_TICKERS

    company   = NIFTY50_TICKERS[sym]
    dcf       = svc.valuation(sym)
    score     = svc.score_for(sym)
    freshness = svc.freshness(sym)

    ai_thesis = ""
    if ai:
        _rate_limit(_client_key(request, "thesis"), _LLM_RATE_LIMIT)
        try:
            ai_thesis = _build_thesis(sym, svc.report(sym, include_financials=False))
        except Exception as exc:
            logger.warning(f"export: thesis skipped for {sym}: {exc}")

    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base  = _SAFE_NAME.sub("", f"{sym}_EquityResearch_{stamp}")

    try:
        if fmt == "docx":
            from src.reports.docx_report import generate_docx
            path = generate_docx(
                symbol=sym, company_name=company, dcf_result=dcf,
                score_result=score, ai_thesis=ai_thesis,
                output_dir=str(_EXPORT_DIR), freshness=freshness,
                filename=f"{base}.docx",
            )
            media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            from src.reports.pdf_report import generate_pdf
            path = generate_pdf(
                symbol=sym, company_name=company, dcf_result=dcf,
                score_result=score, ai_thesis=ai_thesis,
                output_dir=str(_EXPORT_DIR),
            )
            media = "application/pdf"
    except Exception as exc:
        logger.error(f"export {fmt} failed for {sym}: {exc}")
        raise HTTPException(status_code=500, detail=f"Could not generate the {fmt.upper()}: {exc}")

    filename = f"{sym}_EquityResearch_{stamp}.{fmt}"
    return FileResponse(
        path,
        media_type=media,
        filename=filename,
        headers={
            # Explicit attachment disposition so Safari downloads rather than
            # trying to render the file inline.
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


# ── Static front end ──────────────────────────────────────────────────────────

@app.get("/")
def index() -> Any:
    index_file = _WEB_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            {"detail": "Front end not built. The API is at /api/health and /docs."},
            status_code=404,
        )
    return FileResponse(index_file, media_type="text/html")


@app.get("/favicon.ico")
def favicon() -> Any:
    icon = _WEB_DIR / "favicon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    return RedirectResponse("/static/favicon.svg", status_code=302)


if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")
