"""Step 2: run the LLM on every excerpt.

Per firm: each prompt variant once at temperature 0, plus N repeated runs of
the base prompt at the stability temperature. Every call is appended to
data/processed/extractions.jsonl with a fingerprint (hash) of the excerpt it
read. A call is skipped on rerun only if the same excerpt was already done,
so firms whose excerpt changed are re-run automatically.
Run from the repo root:
  python -u -m scripts.02_extract                 # model in config.LLM_MODEL
  python -u -m scripts.02_extract openai/model-x  # any other OpenRouter model ID
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

import config
from src.extract_llm import PROMPTS, extract

MODEL = sys.argv[1] if len(sys.argv) > 1 else config.LLM_MODEL
config.LLM_MODEL = MODEL  # extract() reads the model from config
OUT = Path("data/processed")
log_path = OUT / "extractions.jsonl"
done = set()
if log_path.exists():
    for line in log_path.read_text().splitlines():
        r = json.loads(line)
        done.add((r["ticker"], r["condition"], r["run"], r.get("excerpt_sha"), r.get("model")))

jobs = [(p, 0, 0.0) for p in PROMPTS] + \
       [("stability", i, config.STABILITY_TEMPERATURE) for i in range(config.N_STABILITY_RUNS)]

gt = pd.read_csv(OUT / "ground_truth.csv")
texts = {t: (OUT / "excerpts" / f"{t}.txt").read_text() for t in gt["ticker"]}
shas = {t: hashlib.sha1(x.encode()).hexdigest()[:12] for t, x in texts.items()}
todo = [t for t in gt["ticker"]
        if any((t, c, r, shas[t], MODEL) not in done for c, r, _ in jobs)]
print(f"[{MODEL}] {len(todo)} of {len(gt)} firms need LLM calls")

with log_path.open("a") as log:
    for t in todo:
        for condition, run, temp in jobs:
            if (t, condition, run, shas[t], MODEL) in done:
                continue
            prompt = "base" if condition == "stability" else condition
            res = extract(texts[t], prompt=prompt, temperature=temp)
            log.write(json.dumps({"ticker": t, "condition": condition, "run": run,
                                  "temperature": temp, "model": MODEL,
                                  "excerpt_sha": shas[t], **res}) + "\n")
            log.flush()
        print(f"done {t}")
