"""Step 4: compare every scored method side by side (LLMs and rules).

Reads each folder in data/processed/results/ that step 3 wrote.
Writes data/processed/comparison/: summary.csv, accuracy_by_field.csv,
disagreements.csv (field values that some methods got right and others not).
Run from the repo root:  python -m scripts.04_compare
"""
from pathlib import Path

import pandas as pd

OUT = Path("data/processed")
CMP = OUT / "comparison"
CMP.mkdir(exist_ok=True)

methods = sorted(d.name for d in (OUT / "results").iterdir()
                 if (d / "field_level.csv").exists())
summary, by_field, correct = [], {}, {}
for m in methods:
    fl = pd.read_csv(OUT / "results" / m / "field_level.csv")
    fl = fl[fl.outcome != "no_ground_truth"]
    base = fl[fl.condition == "base"]
    z = pd.read_csv(OUT / "results" / m / "zones.csv")
    z = z[(z.condition == "base") & z.zone_true.notna()]
    row = {"method": m,
           "firms": base.ticker.nunique(),
           "field_accuracy": (base.outcome == "correct").mean(),
           "missing": (base.outcome == "missing").sum(),
           "wrong_value": base.outcome.isin(["wrong_value", "sign_error", "scale_error"]).sum(),
           "not_in_text": (~base.grounded & base.as_printed.notna()).sum(),
           "wrong_zone": (z.zone_llm != z.zone_true).sum()}
    stab = fl[fl.condition == "stability"]
    if len(stab):
        agree = stab.groupby(["ticker", "field"]).pred.nunique(dropna=False)
        row["unstable_firm_fields"] = (agree > 1).sum()
        terse = fl[fl.condition == "terse"]
        row["terse_prompt_accuracy"] = (terse.outcome == "correct").mean()
    summary.append(row)
    by_field[m] = base.groupby("field").outcome.apply(lambda s: (s == "correct").mean())
    correct[m] = base.set_index(["ticker", "field"]).outcome.eq("correct")

summary = pd.DataFrame(summary).set_index("method")
summary.to_csv(CMP / "summary.csv")
pd.DataFrame(by_field).to_csv(CMP / "accuracy_by_field.csv")
c = pd.DataFrame(correct)
split = c[c.nunique(axis=1) > 1]
split.to_csv(CMP / "disagreements.csv")

pd.set_option("display.width", 200)
print(summary.round(3).to_string())
print("\nAccuracy by field (main prompt):")
print(pd.DataFrame(by_field).round(3).to_string())
print(f"\n{len(split)} field values where methods disagree (see comparison/disagreements.csv)")
