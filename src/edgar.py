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
    text = re.sub(r"[ \t\xa0]+", " ", soup.get_text("\n"))
    text = re.sub(r"\n\s*\n+", "\n", text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


# Headings that open the two statements the Z''-score needs.
STATEMENT_PATTERNS = [
    r"consolidated balance sheets?|consolidated statements? of financial position",
    r"consolidated statements? of (operations|income|earnings)"
    r"|consolidated (income|earnings) statements?",
]


def statements_excerpt(text: str, max_chars: int = config.MAX_CHARS) -> str:
    """Cut the filing down to the balance sheet and income statement.

    Full 10-Ks are too long to send whole. Each heading appears several
    times (table of contents, notes), so we take the occurrence followed
    by the most digits, which is almost always the statement itself.
    """
    per_statement = max_chars // len(STATEMENT_PATTERNS)
    parts = []
    for pat in STATEMENT_PATTERNS:
        best, best_score = None, -1
        for m in re.finditer(pat, text, flags=re.I):
            window = text[m.start(): m.start() + per_statement]
            score = sum(ch.isdigit() for ch in window)
            if score > best_score:
                best, best_score = window, score
        if best:
            parts.append(best)
    return "\n\n=====\n\n".join(parts)
