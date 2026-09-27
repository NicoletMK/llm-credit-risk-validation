"""Step 2: run the LLM on every excerpt.

Per firm: each prompt variant once at temperature 0, plus N repeated runs of
the base prompt at the stability temperature. Every call is appended to
data/processed/extractions.jsonl; finished calls are skipped on rerun.
Run from the repo root:  python -m scripts.02_extract
"""
import json
from pathlib import Path

import pandas as pd

import config
from src.extract_llm import PROMPTS, extract

OUT = Path("data/processed")
log_path = OUT / "extractions.jsonl"
done = set()
if log_path.exists():
    for line in log_path.read_text().splitlines():
        r = json.loads(line)
        done.add((r["ticker"], r["condition"], r["run"]))

jobs = [(p, 0, 0.0) for p in PROMPTS] + \
       [("stability", i, config.STABILITY_TEMPERATURE) for i in range(config.N_STABILITY_RUNS)]

gt = pd.read_csv(OUT / "ground_truth.csv")
with log_path.open("a") as log:
    for t in gt["ticker"]:
        text = (OUT / "excerpts" / f"{t}.txt").read_text()
        for condition, run, temp in jobs:
            if (t, condition, run) in done:
                continue
            prompt = "base" if condition == "stability" else condition
            res = extract(text, prompt=prompt, temperature=temp)
            log.write(json.dumps({"ticker": t, "condition": condition, "run": run,
                                  "temperature": temp, "model": config.LLM_MODEL,
                                  **res}) + "\n")
            log.flush()
        print(f"done {t}")
