"""Step 1: for each ticker, get the 10-K, cut the statements, record XBRL truth.

Output: data/processed/ground_truth.csv, data/processed/excerpts/<TICKER>.txt
Run from the repo root:  python -m scripts.01_download
"""
from pathlib import Path

import pandas as pd

import config
from src import edgar
from src.ground_truth import xbrl_values

OUT = Path("data/processed")
(OUT / "excerpts").mkdir(parents=True, exist_ok=True)

tickers = [t.strip() for t in Path(config.TICKERS_FILE).read_text().splitlines()
           if t.strip() and not t.startswith("#")]
rows, skipped = [], []
for t in tickers:
    try:
        cik = edgar.ticker_to_cik(t)
        filing = edgar.find_10k(cik, config.FISCAL_YEAR)
        excerpt = edgar.statements_excerpt(edgar.filing_text(filing))
        (OUT / "excerpts" / f"{t}.txt").write_text(excerpt)
        rows.append({"ticker": t, **filing, **xbrl_values(cik, filing["accn"])})
        print(f"ok   {t}")
    except Exception as e:  # log and continue; report the exclusions later
        skipped.append({"ticker": t, "reason": repr(e)})
        print(f"skip {t}: {e}")

pd.DataFrame(rows).to_csv(OUT / "ground_truth.csv", index=False)
pd.DataFrame(skipped).to_csv(OUT / "skipped.csv", index=False)
print(f"\n{len(rows)} firms kept, {len(skipped)} skipped")
