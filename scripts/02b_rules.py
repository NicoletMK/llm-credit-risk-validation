"""Step 2b: run the rule-based challenger on every excerpt.

Writes data/processed/extractions_rules.jsonl in the same format as the LLM
log, so step 3 can score it:  python -m scripts.03_evaluate rules
Run from the repo root:  python -m scripts.02b_rules
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.rules_extract import extract

OUT = Path("data/processed")
gt = pd.read_csv(OUT / "ground_truth.csv")
with (OUT / "extractions_rules.jsonl").open("w") as log:
    for t in gt["ticker"]:
        text = (OUT / "excerpts" / f"{t}.txt").read_text()
        sha = hashlib.sha1(text.encode()).hexdigest()[:12]
        log.write(json.dumps({"ticker": t, "condition": "base", "run": 0,
                              "temperature": None, "model": "rules",
                              "excerpt_sha": sha, **extract(text)}) + "\n")
print(f"Rules run on {len(gt)} firms -> extractions_rules.jsonl")
