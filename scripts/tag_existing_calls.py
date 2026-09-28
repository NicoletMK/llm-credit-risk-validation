"""One-time: tag calls made before excerpt fingerprints existed.

Run this BEFORE re-cutting the excerpts. It stamps each old call with the
fingerprint of the excerpt currently on disk, which is the one it read.
Run from the repo root:  python -m scripts.tag_existing_calls
"""
import hashlib
import json
from pathlib import Path

OUT = Path("data/processed")
log_path = OUT / "extractions.jsonl"
shas = {p.stem: hashlib.sha1(p.read_text().encode()).hexdigest()[:12]
        for p in (OUT / "excerpts").glob("*.txt")}
rows = [json.loads(l) for l in log_path.read_text().splitlines()]
tagged = 0
for r in rows:
    if "excerpt_sha" not in r and r["ticker"] in shas:
        r["excerpt_sha"] = shas[r["ticker"]]
        tagged += 1
log_path.with_suffix(".jsonl.bak").write_text(log_path.read_text())
log_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"Tagged {tagged} of {len(rows)} calls. Backup saved as extractions.jsonl.bak")
