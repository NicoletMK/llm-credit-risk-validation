"""Check each firm's excerpt and answer key. Prints OK or what is missing."""
import pandas as pd
from pathlib import Path

gt = pd.read_csv("data/processed/ground_truth.csv")
fields = ["total_assets", "current_assets", "current_liabilities",
          "total_liabilities", "retained_earnings", "stockholders_equity",
          "operating_income"]

for _, row in gt.iterrows():
    t = row["ticker"]
    text = Path(f"data/processed/excerpts/{t}.txt").read_text().lower()
    problems = []
    if "total assets" not in text:
        problems.append("excerpt has no balance sheet")
    if not any(w in text for w in ["operating income", "operating profit",
                                   "income from operations", "operating earnings"]):
        problems.append("excerpt has no operating income")
    missing = [f for f in fields if pd.isna(row[f])]
    if missing:
        problems.append("answer key missing: " + ", ".join(missing))
    if str(row["total_liabilities_source"]).startswith("derived"):
        problems.append("total liabilities derived (not tagged)")
    print(f"{t:6} {'OK' if not problems else '; '.join(problems)}")
