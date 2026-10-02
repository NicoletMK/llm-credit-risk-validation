"""Step 2: run the LLM on every excerpt.

Per firm: each prompt variant once at temperature 0, plus N repeated runs of
the base prompt at the stability temperature. Every call is appended to
data/processed/extractions.jsonl with the model name and a fingerprint (hash)
of the excerpt it read. Calls already done for the same model and excerpt are
skipped, so the script can be stopped and restarted at any time.

Several calls run at once (default 6) to save time.
Run from the repo root:
  python -u -m scripts.02_extract                     # model in config.LLM_MODEL
  python -u -m scripts.02_extract openai/model-x      # any other OpenRouter model ID
  python -u -m scripts.02_extract openai/model-x 10   # with 10 calls at once
"""
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import config

MODEL = sys.argv[1] if len(sys.argv) > 1 else config.LLM_MODEL
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
config.LLM_MODEL = MODEL  # extract() reads the model from config
from src.extract_llm import PROMPTS, extract  # noqa: E402

OUT = Path("data/processed")
log_path = OUT / "extractions.jsonl"
done = set()
if log_path.exists():
    for line in log_path.read_text().splitlines():
        r = json.loads(line)
        done.add((r["ticker"], r["condition"], r["run"], r.get("excerpt_sha"), r.get("model")))

conditions = [(p, 0, 0.0) for p in PROMPTS] + \
             [("stability", i, config.STABILITY_TEMPERATURE) for i in range(config.N_STABILITY_RUNS)]

gt = pd.read_csv(OUT / "ground_truth.csv")
texts = {t: (OUT / "excerpts" / f"{t}.txt").read_text() for t in gt["ticker"]}
shas = {t: hashlib.sha1(x.encode()).hexdigest()[:12] for t, x in texts.items()}
jobs = [(t, c, r, temp) for t in gt["ticker"] for c, r, temp in conditions
        if (t, c, r, shas[t], MODEL) not in done]
print(f"[{MODEL}] {len(jobs)} calls to make, {WORKERS} at a time")

lock = threading.Lock()


def run(job):
    t, condition, run_id, temp = job
    prompt = "base" if condition == "stability" else condition
    for attempt in range(6):
        try:
            res = extract(texts[t], prompt=prompt, temperature=temp)
            break
        except Exception as e:
            if "429" in repr(e) and attempt < 5:   # rate limited: wait, then retry
                time.sleep(15 * (attempt + 1))
                continue
            return job, repr(e)                     # other error: skip; a rerun retries it
    record = {"ticker": t, "condition": condition, "run": run_id, "temperature": temp,
              "model": MODEL, "excerpt_sha": shas[t], **res}
    with lock, log_path.open("a") as log:
        log.write(json.dumps(record) + "\n")
    return job, None


finished = failed = 0
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    for fut in as_completed([pool.submit(run, j) for j in jobs]):
        job, err = fut.result()
        if err:
            failed += 1
            print(f"error {job[0]} {job[1]} {job[2]}: {err[:120]}")
        else:
            finished += 1
        if (finished + failed) % 50 == 0:
            print(f"{finished + failed}/{len(jobs)} calls ({failed} errors)")
print(f"done: {finished} saved, {failed} errors" + (" (rerun to retry them)" if failed else ""))
