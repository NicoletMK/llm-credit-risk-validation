"""Step 1: for each ticker, get the 10-K, cut the statements, record XBRL truth.

Inclusion rule: the firm's statements must print current assets, current
liabilities, and operating income. Firms that fail are listed in skipped.csv.
Run from the repo root:  python -m scripts.01_download
"""
from pathlib import Path

import pandas as pd

import config
from src import edgar
from src.ground_truth import FIELDS, xbrl_values

OUT = Path("data/processed")
(OUT / "excerpts").mkdir(parents=True, exist_ok=True)
OPINC_WORDS = ["operating income", "operating profit", "income from operations",
               "operating earnings", "operating loss", "loss from operations"]

tickers = [t.strip() for t in Path(config.TICKERS_FILE).read_text().splitlines()
           if t.strip() and not t.startswith("#")]
rows, skipped = [], []
for t in tickers:
    try:
        cik = edgar.ticker_to_cik(t)
        filing = edgar.find_10k(cik, config.FISCAL_YEAR)
        excerpt = edgar.statements_excerpt(edgar.filing_text(filing))
        truth = xbrl_values(cik, filing["accn"])
    except Exception as e:
        skipped.append({"ticker": t, "reason": f"download error: {e!r}"})
        print(f"skip {t}: download error")
        continue

    flat = " ".join(excerpt.lower().split())
    reasons = []
    if truth["current_assets"] is None or truth["current_liabilities"] is None:
        reasons.append("no current assets/liabilities (unclassified balance sheet)")
    if truth["operating_income"] is None or not any(w in flat for w in OPINC_WORDS):
        reasons.append("no operating income line on income statement")
    missing = [f for f in FIELDS if truth[f] is None
               and f not in ("current_assets", "current_liabilities", "operating_income")]
    if missing:
        reasons.append("answer key missing: " + ", ".join(missing))
    if reasons:
        skipped.append({"ticker": t, "reason": "; ".join(reasons)})
        print(f"skip {t}: {'; '.join(reasons)}")
        continue

    (OUT / "excerpts" / f"{t}.txt").write_text(excerpt)
    rows.append({"ticker": t, **filing, **truth})
    print(f"ok   {t}")

pd.DataFrame(rows).to_csv(OUT / "ground_truth.csv", index=False)
pd.DataFrame(skipped, columns=["ticker", "reason"]).to_csv(OUT / "skipped.csv", index=False)
print(f"\n{len(rows)} firms kept, {len(skipped)} excluded (reasons in data/processed/skipped.csv)")
