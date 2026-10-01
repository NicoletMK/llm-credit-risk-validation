"""Step 3: score one method's extractions and write the tables for the report.

  python -m scripts.03_evaluate                 # the model in config.LLM_MODEL
  python -m scripts.03_evaluate openai/model-x  # another model you ran
  python -m scripts.03_evaluate rules           # the rule-based challenger

Outputs in data/processed/results/<method>/:
  field_level.csv      one row per (firm, condition, run, field)
  accuracy_by_field.csv, error_types.csv, hallucination.csv,
  stability.csv, prompt_sensitivity.csv, zones.csv, zone_confusion.csv
Run from the repo root:  python -m scripts.03_evaluate
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.ground_truth import FIELDS
from src.validate import classify, is_grounded, numbers_in, stability
from src.zscore import z_double_prime, zone

import sys

import config

METHOD = sys.argv[1] if len(sys.argv) > 1 else config.LLM_MODEL
OUT = Path("data/processed")
LOG = OUT / ("extractions_rules.jsonl" if METHOD == "rules" else "extractions.jsonl")
RES = OUT / "results" / METHOD.replace("/", "_")
RES.mkdir(parents=True, exist_ok=True)

gt = pd.read_csv(OUT / "ground_truth.csv").set_index("ticker")
runs = [json.loads(l) for l in LOG.read_text().splitlines()]
# Keep only calls made on each firm's current excerpt (drops firms no longer
# in the sample and calls made on an older, since-fixed excerpt).
cur = {t: hashlib.sha1((OUT / "excerpts" / f"{t}.txt").read_text().encode()).hexdigest()[:12]
       for t in gt.index}
runs = [r for r in runs if r["ticker"] in cur and r.get("excerpt_sha") == cur[r["ticker"]]
        and r.get("model") == METHOD]
if not runs:
    sys.exit(f"No extractions found for {METHOD}. Run step 2 for it first.")
texts = {t: numbers_in((OUT / "excerpts" / f"{t}.txt").read_text()) for t in gt.index}

rows, zrows = [], []
for r in runs:
    t, parsed = r["ticker"], r["parsed"] or {}
    pred = {}
    for f in FIELDS:
        item = parsed.get(f) or {}
        truth = gt.at[t, f]
        truth = None if pd.isna(truth) else float(truth)
        v = item.get("value_usd")
        pred[f] = None if v is None else float(v)
        rows.append({"ticker": t, "condition": r["condition"], "run": r["run"],
                     "field": f, "pred": pred[f], "truth": truth,
                     "as_printed": item.get("as_printed"), "scale": item.get("scale"),
                     "outcome": classify(pred[f], truth),
                     "grounded": is_grounded(item.get("as_printed"), texts[t]),
                     "parse_error": r["error"]})
    truth_vals = {f: (None if pd.isna(gt.at[t, f]) else float(gt.at[t, f])) for f in FIELDS}
    z_llm, z_true = z_double_prime(pred), z_double_prime(truth_vals)
    zrows.append({"ticker": t, "condition": r["condition"], "run": r["run"],
                  "z_llm": z_llm, "z_true": z_true,
                  "zone_llm": zone(z_llm), "zone_true": zone(z_true)})

fl = pd.DataFrame(rows)
fl.to_csv(RES / "field_level.csv", index=False)
scored = fl[fl.outcome != "no_ground_truth"]
base = scored[scored.condition == "base"]

# 1. Accuracy per field, main prompt.
(base.assign(correct=base.outcome.eq("correct"))
     .groupby("field").correct.agg(["mean", "count"])
     .rename(columns={"mean": "accuracy", "count": "n"})
     .to_csv(RES / "accuracy_by_field.csv"))

# 2. What kind of errors.
pd.crosstab(base.field, base.outcome).to_csv(RES / "error_types.csv")

# 3. Hallucination: reported number not found anywhere in the text.
has_val = scored[scored.as_printed.notna()]
(has_val.assign(hallucinated=~has_val.grounded)
        .groupby("condition").hallucinated.agg(["mean", "sum", "count"])
        .to_csv(RES / "hallucination.csv"))

# 4. Stability across repeated runs.
stab = fl[fl.condition == "stability"]
srows = [{"ticker": t, "field": f, **stability(g.pred.fillna(-1).tolist())}
         for (t, f), g in stab.groupby(["ticker", "field"])]
if srows:  # the rules challenger is deterministic, so it has no stability runs
    pd.DataFrame(srows).to_csv(RES / "stability.csv", index=False)

# 5. Prompt sensitivity: accuracy by prompt wording.
prompt_conds = scored[scored.condition != "stability"]
(prompt_conds.assign(correct=prompt_conds.outcome.eq("correct"))
     .groupby(["condition", "field"]).correct.mean().unstack(0)
     .to_csv(RES / "prompt_sensitivity.csv"))

# 6. Headline: does the LLM put firms in the wrong risk zone?
z = pd.DataFrame(zrows)
z.to_csv(RES / "zones.csv", index=False)
zb = z[(z.condition == "base") & z.zone_true.notna()]
pd.crosstab(zb.zone_true, zb.zone_llm.fillna("not computable"),
            rownames=["true zone"], colnames=["LLM zone"]).to_csv(RES / "zone_confusion.csv")

print(f"[{METHOD}]")
print(f"Field accuracy (main prompt): {base.outcome.eq('correct').mean():.1%}")
print(f"Hallucination rate (main prompt): "
      f"{(~has_val[has_val.condition == 'base'].grounded).mean():.1%}")
print(f"Zone misclassification: {(zb.zone_llm != zb.zone_true).mean():.1%} "
      f"of {len(zb)} firms")
