"""Download 10-K filings and XBRL facts from SEC EDGAR.

SEC fair-access rules: send a User-Agent with name + email, stay under
10 requests/second. Everything is cached in data/raw/ so reruns are free.
"""
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import config

RAW = Path("data/raw")
HEADERS = {"User-Agent": config.SEC_USER_AGENT}
_last_call = 0.0


def _get(url: str) -> requests.Response:
    global _last_call
    wait = 0.15 - (time.time() - _last_call)  # ~7 req/s
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    _last_call = time.time()
    resp.raise_for_status()
    return resp


def _cached_json(url: str, path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    data = _get(url).json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def ticker_to_cik(ticker: str) -> int:
    table = _cached_json("https://www.sec.gov/files/company_tickers.json",
                         RAW / "company_tickers.json")
    for row in table.values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"])
    raise KeyError(f"Ticker not found: {ticker}")


def company_facts(cik: int) -> dict:
    """All XBRL facts the company has reported (the ground truth source)."""
    return _cached_json(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        RAW / "facts" / f"{cik}.json")


def find_10k(cik: int, fiscal_year: int) -> dict:
    """Return the 10-K whose fiscal year matches, using the XBRL facts.

    The facts file tags each value with its fiscal year (fy) and the
    accession number (accn) of the filing it came from, so we take the
    accession from the 10-K reporting that fiscal year.
    """
    facts = company_facts(cik)
    for unit_rows in facts["facts"]["us-gaap"]["Assets"]["units"].values():
        for r in unit_rows:
            if r.get("form") == "10-K" and r.get("fy") == fiscal_year:
                accn = r["accn"]
                break
        else:
            continue
        break
    else:
        raise LookupError(f"No FY{fiscal_year} 10-K for CIK {cik}")

    subs = _cached_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                        RAW / "submissions" / f"{cik}.json")
    recent = subs["filings"]["recent"]
    i = recent["accessionNumber"].index(accn)
    return {"cik": cik, "accn": accn,
            "primary_doc": recent["primaryDocument"][i],
            "report_date": recent["reportDate"][i]}


def filing_text(filing: dict) -> str:
    """Plain text of the 10-K's primary document."""
    path = RAW / "filings" / f"{filing['accn']}.txt"
    if path.exists():
        return path.read_text()
    url = (f"https://www.sec.gov/Archives/edgar/data/{filing['cik']}/"
           f"{filing['accn'].replace('-', '')}/{filing['primary_doc']}")
    soup = BeautifulSoup(_get(url).content, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Inline XBRL filings carry a hidden block of tags, dates and IDs.
    # It is full of digits and would win the statement search, so drop it.
    hidden = soup.find_all(style=re.compile(r"display\s*:\s*none", re.I))
    for tag in hidden + soup.find_all("ix:header"):
        if not tag.decomposed:
            tag.decompose()
    text = re.sub(r"[ \t\xa0]+", " ", soup.get_text("\n"))
    text = re.sub(r"\n\s*\n+", "\n", text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# Headings that open the two statements the Z''-score needs.
# Headings are matched on a copy of the text with all whitespace removed,
# because filings often split headings across lines or even mid-word
# ("Consolidated B" / "alance Sheets"). "Consolidated" is optional because
# some firms (e.g., Microsoft) title their statements "Balance Sheets".
STATEMENT_PATTERNS = [
    r"(consolidated)?(balancesheets?|statements?offinancialposition|financialposition)",
    r"(consolidated)?(statements?of(operations|income|earnings)"
    r"|(income|earnings)statements?|resultsofoperations)",
]

# Line items that follow each heading when it is the real statement.
ANCHORS = [
    ["total current assets", "total assets", "total current liabilities",
     "retained earnings", "accumulated deficit", "total liabilities", "equity"],
    ["revenue", "net sales", "cost of", "gross profit", "operating income",
     "income from operations", "income before income taxes", "net income",
     "net earnings", "per share"],
]


def _squash(text):
    """Text with whitespace removed, plus each character's original position."""
    chars, pos = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch.lower())
            pos.append(i)
    return "".join(chars), pos


def statements_excerpt(text: str, max_chars: int = config.MAX_CHARS) -> str:
    """Cut the filing down to the balance sheet and income statement.

    Each heading appears many times (table of contents, auditor's report,
    MD&A, notes). We keep the occurrence whose next few thousand characters
    contain the most of that statement's line items, with digit count as
    tiebreak.
    """
    per_statement = max_chars // len(STATEMENT_PATTERNS)
    squashed, pos = _squash(text)
    parts = []
    for pat, anchors in zip(STATEMENT_PATTERNS, ANCHORS):
        best, best_score = None, (-1, -1)
        for m in re.finditer(pat, squashed):
            start = pos[m.start()]
            window = text[start: start + per_statement]
            head = " ".join(window[:6000].lower().split())
            score = (sum(a in head for a in anchors),
                     sum(ch.isdigit() for ch in head))
            if score > best_score:
                best, best_score = window, score
        if best:
            parts.append(best)
    return "\n\n=====\n\n".join(parts)
