"""
Earnings-call transcripts for NSE companies — the source for management's own
statements about targets, guidance and plans.

Screener's company page lists each concall with a link to the raw transcript,
usually the company's filing on BSE. Those PDFs are downloaded, their text is
extracted, and each is dated from its own text ("quarter ended March 31, 2026")
rather than from Screener's label: the labels are sometimes wrong (TCS's
"Jul 2026" entry once linked the July 2025 call).

Cached under data/raw/transcripts/<SYMBOL>/, one .txt per document plus an
index.json, so a document is downloaded once.

Usage:
    fetcher = TranscriptFetcher()
    for t in fetcher.latest("TCS", n=2):
        print(t.period_end, t.url, len(t.text))
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.data_pipeline.fetchers.screener_fetcher import HEADERS, SCREENER_SLUG_OVERRIDES
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MONTHS = ("January|February|March|April|May|June|July|August|September|"
           "October|November|December")
# "quarter ended March 31, 2026", "quarter and year ended March 31, 2026",
# "fourth quarter and financial year ended 31st March, 2026",
# "first quarter (Q1) ended June 30, 2026"
_DATE = (rf"(?:({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})"
         rf"|(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS}),?\s+(\d{{4}}))")
_PERIOD_END = re.compile(rf"(?:quarter|period)[\w\s,'()-]{{0,40}}?\s+ended\s+{_DATE}", re.IGNORECASE)
# "for the year ended March 31st, 2026" — a full-year (Q4) call. Only used when no
# quarter is stated, because other calls mention the previous year in passing.
_YEAR_END = re.compile(rf"\byear\s+ended\s+{_DATE}", re.IGNORECASE)
# "Q4 FY26", "Q1 FY '27", "Q3 FY2026"
_QUARTER_FY = re.compile(r"\bQ([1-4])\s*FY\s*'?\s*(\d{4}|\d{2})\b", re.IGNORECASE)
# Infosys states it only in its results URL: ".../quarterly-results/2026-2027/q1.html"
_RESULTS_URL = re.compile(r"quarterly-results/(\d{4})-\s?(\d{4})/q([1-4])", re.IGNORECASE)
_QUARTER_END = {1: (6, 30), 2: (9, 30), 3: (12, 31), 4: (3, 31)}
_SCAN_CHARS = 6000          # the period is stated on the cover letter or first page


@dataclass
class ConcallLink:
    label:          str              # Screener's month label, e.g. "Apr 2026"
    transcript_url: str
    ppt_url:        Optional[str] = None


@dataclass
class Transcript:
    symbol:      str
    label:       str                 # Screener's label (may be wrong)
    url:         str
    period_end:  Optional[date]      # quarter end stated in the document itself
    pages:       int
    text:        str
    fetched_at:  str

    @property
    def period_label(self) -> str:
        """'Q4 FY26'-style label from period_end (Indian April–March fiscal year)."""
        if not self.period_end:
            return self.label
        m, y = self.period_end.month, self.period_end.year
        q  = {6: 1, 9: 2, 12: 3, 3: 4}.get(m)
        fy = y + 1 if m > 3 else y
        return f"Q{q} FY{fy % 100:02d}" if q else self.period_end.strftime("%b %Y")


def parse_concalls(html: str) -> list[ConcallLink]:
    """Concall entries with a transcript link, in page order (newest first)."""
    soup  = BeautifulSoup(html, "html.parser")
    links = []
    for li in soup.select("#documents .concalls li"):
        label_div = li.find("div")
        anchors   = {a.get_text(strip=True): a.get("href") for a in li.find_all("a") if a.get("href")}
        if not label_div or not anchors.get("Transcript"):
            continue
        links.append(ConcallLink(label=label_div.get_text(strip=True),
                                 transcript_url=anchors["Transcript"],
                                 ppt_url=anchors.get("PPT")))
    return links


def period_end_from_text(text: str) -> Optional[date]:
    """
    The quarter end a transcript is about, read from its opening pages: an
    explicit "quarter ... ended <date>" if stated, else a "Q4 FY26"-style label,
    else a company results URL (Indian April–March fiscal year). None for
    non-earnings events such as an investor day.
    """
    head = " ".join(text[:_SCAN_CHARS].split())
    m = _PERIOD_END.search(head)
    if m and (d := _match_date(m)):
        return d
    q = _QUARTER_FY.search(head)
    if q:
        fy = int(q.group(2)) if len(q.group(2)) == 4 else 2000 + int(q.group(2))
        return _quarter_end(int(q.group(1)), fy)
    y = _YEAR_END.search(head)
    if y and (d := _match_date(y)) and d.month == 3:
        return d
    u = _RESULTS_URL.search(head.replace(" ", ""))
    if u and int(u.group(2)) == int(u.group(1)) + 1:
        return _quarter_end(int(u.group(3)), int(u.group(2)))
    return None


def _match_date(m: re.Match) -> Optional[date]:
    if m.group(1):
        month, day, year = m.group(1), m.group(2), m.group(3)
    else:
        day, month, year = m.group(4), m.group(5), m.group(6)
    try:
        return datetime.strptime(f"{month} {day} {year}", "%B %d %Y").date()
    except ValueError:
        return None


def _label_matches(label: str, period_end: date) -> bool:
    """
    Whether a listing month fits the quarter a document covers. Calls happen a
    few weeks to ~3 months after quarter end, so anything outside −1..+6 months
    is a wrong link. An unparseable label is given the benefit of the doubt.
    """
    try:
        listed = datetime.strptime(label.strip(), "%b %Y").date()
    except ValueError:
        return True
    months = (listed.year - period_end.year) * 12 + listed.month - period_end.month
    return -1 <= months <= 6


def _quarter_end(quarter: int, fiscal_year: int) -> date:
    """Last day of `quarter` of the April–March year ending in March `fiscal_year`."""
    month, day = _QUARTER_END[quarter]
    return date(fiscal_year if quarter == 4 else fiscal_year - 1, month, day)


class TranscriptFetcher:

    def __init__(self, cache_dir: str | Path = "data/raw/transcripts",
                 rate_limit_delay: float = 1.5):
        self.cache_dir = Path(cache_dir)
        self.delay     = rate_limit_delay

    def latest(self, symbol: str, n: int = 2, max_documents: int = 6) -> list[Transcript]:
        """
        The `n` most recent transcripts for distinct quarters, newest first.
        Looks at up to `max_documents` links; a duplicate or undatable document
        is skipped, so a mislabelled link cannot pass for a newer call.
        """
        links = self.list_concalls(symbol)
        found: dict[date, Transcript] = {}
        for link in links[:max_documents]:
            t = self.get(symbol, link)
            if t is None or t.period_end is None:
                continue
            if not _label_matches(link.label, t.period_end):
                logger.warning(f"[Transcripts] {symbol}: '{link.label}' links a {t.period_label} "
                               f"document — mislabelled, skipped")
                continue
            found.setdefault(t.period_end, t)
            if len(found) >= n:
                break
        return sorted(found.values(), key=lambda t: t.period_end, reverse=True)[:n]

    def list_concalls(self, symbol: str) -> list[ConcallLink]:
        slug = SCREENER_SLUG_OVERRIDES.get(symbol, symbol)
        for suffix in ("consolidated/", ""):
            try:
                resp = requests.get(f"https://www.screener.in/company/{slug}/{suffix}",
                                    headers=HEADERS, timeout=20)
                time.sleep(self.delay)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning(f"[Transcripts] {symbol}: Screener page failed: {exc}")
                return []
            links = parse_concalls(resp.text)
            if links:
                return links
        return []

    def get(self, symbol: str, link: ConcallLink) -> Optional[Transcript]:
        """One transcript, from the cache or downloaded and extracted."""
        sym_dir = self.cache_dir / symbol
        key     = hashlib.sha1(link.transcript_url.encode()).hexdigest()[:16]
        index   = self._read_index(sym_dir)
        if key in index:
            meta = index[key]
            text = (sym_dir / f"{key}.txt").read_text(encoding="utf-8")
            # Dated from the text on every load, so parser fixes reach cached documents.
            return Transcript(text=text, symbol=symbol, label=meta["label"], url=meta["url"],
                              period_end=period_end_from_text(text),
                              pages=meta["pages"], fetched_at=meta["fetched_at"])

        pdf = self._download(link.transcript_url)
        if pdf is None:
            return None
        text, pages = _extract_text(pdf)
        if not text.strip():
            logger.warning(f"[Transcripts] {symbol} {link.label}: no extractable text (scanned PDF?)")
            return None
        t = Transcript(symbol=symbol, label=link.label, url=link.transcript_url,
                       period_end=period_end_from_text(text), pages=pages, text=text,
                       fetched_at=datetime.now().isoformat(timespec="seconds"))

        sym_dir.mkdir(parents=True, exist_ok=True)
        (sym_dir / f"{key}.txt").write_text(text, encoding="utf-8")
        meta = {k: v for k, v in asdict(t).items() if k != "text"}
        meta["period_end"] = t.period_end.isoformat() if t.period_end else None
        index[key] = meta
        (sym_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
        return t

    def _download(self, url: str) -> Optional[bytes]:
        try:
            resp = requests.get(url, headers={**HEADERS, "Referer": "https://www.bseindia.com/"},
                                timeout=60)
            time.sleep(self.delay)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"[Transcripts] download failed {url}: {exc}")
            return None
        if not resp.content.startswith(b"%PDF"):
            logger.warning(f"[Transcripts] not a PDF: {url}")
            return None
        return resp.content

    @staticmethod
    def _read_index(sym_dir: Path) -> dict:
        path = sym_dir / "index.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


def _extract_text(pdf: bytes) -> tuple[str, int]:
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(pdf))
        return "\n".join(p.extract_text() or "" for p in reader.pages), len(reader.pages)
    except Exception as exc:          # malformed PDFs raise a variety of errors
        logger.warning(f"[Transcripts] text extraction failed: {exc}")
        return "", 0
