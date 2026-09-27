"""Step 0: build the firm sample for the full study.

1. Pull every company's FY2024 values for the Z'' inputs from the SEC
   "frames" API (one file per XBRL tag, all companies at once).
2. Keep firms that tag every input, file annual reports on Form 10-K
   (foreign firms filing 20-F are out of scope), are in retail
   (SIC 5200-5999) or software and IT services (SIC 7370-7379), and
   have at least $50M in assets.
3. Compute each firm's Z''-zone from its XBRL values and sample up to
   PER_ZONE firms per zone, so the distress and grey zones are not empty.

Outputs:
  data/tickers.txt          the sample, read by step 1
  data/sample_frame.csv     every eligible firm, its zone, and whether it was sampled
Run from the repo root:  python -m scripts.00_build_sample
"""
import random
from pathlib import Path

import pandas as pd

from src import edgar
from src.zscore import z_double_prime, zone

SECTORS = {"retail": (5200, 5999), "software_it": (7370, 7379)}
PER_ZONE = 60
MIN_ASSETS = 50e6
SEED = 2026

INSTANT, ANNUAL = "CY2024Q4I", "CY2024"
TAGS = {
    "total_assets":        ("Assets", INSTANT),
    "current_assets":      ("AssetsCurrent", INSTANT),
    "current_liabilities": ("LiabilitiesCurrent", INSTANT),
    "retained_earnings":   ("RetainedEarningsAccumulatedDeficit", INSTANT),
    "equity_incl_nci":     ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", INSTANT),
    "equity_parent":       ("StockholdersEquity", INSTANT),
    "operating_income":    ("OperatingIncomeLoss", ANNUAL),
}


def frame(tag, period):
    url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json"
    data = edgar._cached_json(url, edgar.RAW / "frames" / f"{tag}_{period}.json")
    return {int(r["cik"]): r["val"] for r in data["data"]}


def sector_of(sic):
    for name, (lo, hi) in SECTORS.items():
        if lo <= sic <= hi:
            return name
    return None


print("Downloading frames...")
df = pd.DataFrame({field: pd.Series(frame(tag, period))
                   for field, (tag, period) in TAGS.items()})
df.index.name = "cik"
df["total_equity"] = df["equity_incl_nci"].fillna(df["equity_parent"])
need = ["total_assets", "current_assets", "current_liabilities",
        "retained_earnings", "total_equity", "operating_income"]
df = df.dropna(subset=need)
df = df[df.total_assets >= MIN_ASSETS]
print(f"{len(df)} firms tag every input. Looking up sectors (cached after first run)...")

rows = []
for i, cik in enumerate(df.index):
    if i and i % 250 == 0:
        print(f"  {i}/{len(df)}")
    subs = edgar._cached_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                              edgar.RAW / "submissions" / f"{cik}.json")
    sector = sector_of(int(subs.get("sic") or 0))
    tickers = subs.get("tickers") or []
    files_10k = "10-K" in subs.get("filings", {}).get("recent", {}).get("form", [])
    if sector and tickers and files_10k:
        rows.append({"cik": cik, "ticker": tickers[0], "name": subs.get("name"),
                     "sic": subs.get("sic"), "sector": sector})

pop = pd.DataFrame(rows).set_index("cik").join(df[need])
pop["z"] = [z_double_prime(r) for r in pop[need].to_dict("records")]
pop["zone"] = pop["z"].map(zone)
pop = pop.dropna(subset=["zone"])

rng = random.Random(SEED)
pop["sampled"] = False
for z, group in pop.groupby("zone"):
    picks = rng.sample(list(group.index), min(PER_ZONE, len(group)))
    pop.loc[picks, "sampled"] = True

pop.to_csv("data/sample_frame.csv")
sample = pop[pop.sampled]
Path("data/tickers.txt").write_text(
    f"# Built by scripts/00_build_sample.py (seed {SEED}, up to {PER_ZONE} per zone)\n"
    + "\n".join(f"{t} {c}" for c, t in sample.ticker.items()) + "\n")

print("\nEligible firms by sector and zone:")
print(pd.crosstab(pop.sector, pop.zone, margins=True))
print("\nSampled:")
print(pd.crosstab(sample.sector, sample.zone, margins=True))
print(f"\n{len(sample)} tickers written to data/tickers.txt")
