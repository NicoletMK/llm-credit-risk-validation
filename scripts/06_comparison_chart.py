"""Step 6: chart comparing every extraction method (main prompt).

Panel (a): share of field values matching the filed value exactly.
Panel (b): share of firms placed in the wrong Z'' zone.
Includes GPT-6 Luna's first run (before the statement-search fix) when
data/processed/results_run1/ exists, rescored with the same exact-match rule.
Output: data/processed/comparison/comparison.png
Run from the repo root:  python -m scripts.06_comparison_chart
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("data/processed")
CMP = OUT / "comparison"
CMP.mkdir(exist_ok=True)

# Folder name -> label on the chart, in display order.
METHODS = [
    ("results_run1", "GPT-6 Luna\nrun 1 (before fix)"),
    ("results/openai_gpt-6-luna", "GPT-6 Luna"),
    ("results/anthropic_claude-haiku-4.5", "Claude Haiku 4.5"),
    ("results/rules", "Rule-based\nchallenger"),
]

rows = []
for folder, label in METHODS:
    d = OUT / folder
    if not (d / "field_level.csv").exists():
        continue
    fl = pd.read_csv(d / "field_level.csv")
    fl = fl[(fl.condition == "base") & fl.truth.notna()]
    correct = fl.pred.notna() & (fl.pred == fl.truth)   # exact match for every method
    z = pd.read_csv(d / "zones.csv")
    z = z[(z.condition == "base") & z.zone_true.notna()]
    rows.append({"method": label, "firms": z.ticker.nunique(),
                 "field_accuracy": 100 * correct.mean(),
                 "wrong_zone": 100 * (z.zone_llm != z.zone_true).mean(),
                 "wrong_zone_n": int((z.zone_llm != z.zone_true).sum())})
res = pd.DataFrame(rows)
res.to_csv(CMP / "comparison_chart_data.csv", index=False)

colors = ["#bbbbbb" if "run 1" in m else ("#d98c2b" if "Rule" in m else "#2b6cb0")
          for m in res.method]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
b1 = ax1.bar(res.method, res.field_accuracy, color=colors)
ax1.set_ylim(min(80, res.field_accuracy.min() - 3), 101.5)
ax1.set_ylabel("Field values exactly correct (%)")
ax1.set_title("(a) Extraction accuracy")
for bar, v in zip(b1, res.field_accuracy):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

b2 = ax2.bar(res.method, res.wrong_zone, color=colors)
ax2.set_ylim(0, max(res.wrong_zone.max() * 1.18, 1))
ax2.set_ylabel("Firms not in their correct zone (%)")
ax2.set_title("(b) Effect on the credit risk zone")
for bar, v, n, f in zip(b2, res.wrong_zone, res.wrong_zone_n, res.firms):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + ax2.get_ylim()[1] * 0.01,
             f"{n}/{f}", ha="center", va="bottom", fontsize=9)
for ax in (ax1, ax2):
    ax.tick_params(axis="x", labelsize=9)
fig.tight_layout()
fig.savefig(CMP / "comparison.png", dpi=150)
print(res.round(2).to_string(index=False))
print(f"\nChart written to {CMP / 'comparison.png'}")
