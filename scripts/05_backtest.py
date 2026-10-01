"""Step 5: backtest the Z''-score against real bankruptcies.

Does Z'' predict which companies file for bankruptcy in the following year?

Two cohorts, each scored on its fiscal-year values and followed for the
next calendar year:
  FY2023 values -> bankruptcy filed during 2024
  FY2024 values -> bankruptcy filed during 2025
A bankruptcy is an 8-K reporting Item 1.03 (Bankruptcy or Receivership),
which companies must file with the SEC within four business days.

Population: 10-K filers with every Z'' input, at least $50M in assets,
excluding financial firms (SIC 6000-6999), for which Z'' was not designed.
Firms that already reported a bankruptcy before the follow-up year are dropped.

Tests:
  1. Bankruptcy rate by zone (safe / grey / distress).
  2. AUC of Z'' with a bootstrap 95% interval (0.5 = no better than chance).
  3. Out-of-time backtest: refit the four weights by logistic regression on
     the FY2023 cohort, test on FY2024, and compare with Altman's fixed weights.

Outputs in data/processed/backtest/. Run from the repo root:
  python -m scripts.05_backtest
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src import edgar
from src.zscore import zone

OUT = Path("data/processed/backtest")
OUT.mkdir(parents=True, exist_ok=True)
MIN_ASSETS = 50e6
N_BOOT = 1000
RNG = np.random.default_rng(2026)
COHORTS = {2023: ("CY2023Q4I", "CY2023"), 2024: ("CY2024Q4I", "CY2024")}
TAGS = {
    "total_assets":        ("Assets", 0),
    "current_assets":      ("AssetsCurrent", 0),
    "current_liabilities": ("LiabilitiesCurrent", 0),
    "retained_earnings":   ("RetainedEarningsAccumulatedDeficit", 0),
    "equity_incl_nci":     ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", 0),
    "equity_parent":       ("StockholdersEquity", 0),
    "operating_income":    ("OperatingIncomeLoss", 1),
}
WEIGHTS = np.array([6.56, 3.26, 6.72, 1.05])
XCOLS = ["x1", "x2", "x3", "x4"]


def frame(tag, period):
    url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json"
    data = edgar._cached_json(url, edgar.RAW / "frames" / f"{tag}_{period}.json")
    return {int(r["cik"]): r["val"] for r in data["data"]}


def submissions(cik):
    return edgar._cached_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                              edgar.RAW / "submissions" / f"{cik}.json")


def bankruptcy_dates(subs):
    """Filing dates of 8-Ks that report Item 1.03."""
    rec = subs.get("filings", {}).get("recent", {})
    return [d for f, d, items in zip(rec.get("form", []), rec.get("filingDate", []),
                                     rec.get("items", []))
            if f.startswith("8-K") and "1.03" in (items or "")]


def build_cohort(year):
    instant, annual = COHORTS[year]
    df = pd.DataFrame({f: pd.Series(frame(tag, (instant, annual)[dur]))
                       for f, (tag, dur) in TAGS.items()})
    df["total_equity"] = df.equity_incl_nci.fillna(df.equity_parent)
    df = df.dropna(subset=["total_assets", "current_assets", "current_liabilities",
                           "retained_earnings", "total_equity", "operating_income"])
    df = df[df.total_assets >= MIN_ASSETS]
    print(f"FY{year}: {len(df)} firms tag every input; checking filings (cached after first run)...")

    rows = []
    start, end = f"{year + 1}-01-01", f"{year + 1}-12-31"
    for i, cik in enumerate(df.index):
        if i and i % 500 == 0:
            print(f"  {i}/{len(df)}")
        subs = submissions(cik)
        sic = int(subs.get("sic") or 0)
        forms = subs.get("filings", {}).get("recent", {}).get("form", [])
        if "10-K" not in forms or 6000 <= sic <= 6999:
            continue
        dates = bankruptcy_dates(subs)
        if any(d < start for d in dates):
            continue  # already bankrupt before the follow-up year
        rows.append({"cik": cik, "name": subs.get("name"), "sic": sic,
                     "bankrupt": int(any(start <= d <= end for d in dates))})
    pop = pd.DataFrame(rows).set_index("cik").join(df)

    ta = pop.total_assets
    tl = ta - pop.total_equity
    pop = pop[tl > 0]
    ta, tl = pop.total_assets, pop.total_assets - pop.total_equity
    pop["x1"] = (pop.current_assets - pop.current_liabilities) / ta
    pop["x2"] = pop.retained_earnings / ta
    pop["x3"] = pop.operating_income / ta
    pop["x4"] = pop.total_equity / tl
    pop["z"] = pop[XCOLS].to_numpy() @ WEIGHTS
    pop["zone"] = pop.z.map(zone)
    pop["cohort"] = year
    return pop


def auc_ci(y, score):
    """AUC with a bootstrap 95% interval."""
    y, score = np.asarray(y), np.asarray(score)
    point = roc_auc_score(y, score)
    boots = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            boots.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


pop = pd.concat([build_cohort(y) for y in COHORTS])
pop.to_csv(OUT / "population.csv")

# 1. Bankruptcy rate by zone.
rates = (pop.groupby(["cohort", "zone"]).bankrupt.agg(["sum", "count", "mean"])
            .rename(columns={"sum": "bankruptcies", "count": "firms", "mean": "rate"}))
rates.to_csv(OUT / "rate_by_zone.csv")

# 2 + 3. AUC: Altman's fixed weights on both cohorts; refit weights out of time.
train, test = pop[pop.cohort == 2023], pop[pop.cohort == 2024]
lo_q, hi_q = train[XCOLS].quantile(0.01), train[XCOLS].quantile(0.99)
clip = lambda d: d[XCOLS].clip(lo_q, hi_q, axis=1)   # limit extreme ratios, using train cutoffs only
logit = LogisticRegression(C=1e6, max_iter=5000).fit(clip(train), train.bankrupt)

results = []
for name, d in [("FY2023 cohort", train), ("FY2024 cohort", test)]:
    a = auc_ci(d.bankrupt, -d.z)          # lower Z'' = riskier
    results.append({"cohort": name, "model": "Altman fixed weights", "auc": a[0],
                    "ci_low": a[1], "ci_high": a[2], "bankruptcies": d.bankrupt.sum(), "firms": len(d)})
a = auc_ci(test.bankrupt, logit.predict_proba(clip(test))[:, 1])
results.append({"cohort": "FY2024 cohort", "model": "Refit weights (trained on FY2023)",
                "auc": a[0], "ci_low": a[1], "ci_high": a[2],
                "bankruptcies": test.bankrupt.sum(), "firms": len(test)})
auc = pd.DataFrame(results)
auc.to_csv(OUT / "auc.csv", index=False)

# Is the refit model really better? Paired bootstrap of the AUC difference
# on the same resampled firms (refit minus Altman, FY2024 cohort).
y = test.bankrupt.to_numpy()
s_alt, s_fit = -test.z.to_numpy(), logit.predict_proba(clip(test))[:, 1]
diffs = []
for _ in range(N_BOOT):
    idx = RNG.integers(0, len(y), len(y))
    if y[idx].min() != y[idx].max():
        diffs.append(roc_auc_score(y[idx], s_fit[idx]) - roc_auc_score(y[idx], s_alt[idx]))
diff = {"auc_difference": roc_auc_score(y, s_fit) - roc_auc_score(y, s_alt),
        "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
        "share_of_resamples_refit_better": np.mean(np.array(diffs) > 0)}
pd.DataFrame([diff]).to_csv(OUT / "auc_difference.csv", index=False)
pd.DataFrame({"feature": XCOLS, "altman_weight": WEIGHTS,
              "refit_coefficient": logit.coef_[0]}).to_csv(OUT / "coefficients.csv", index=False)

# Figure: bankruptcy rate by zone, and ROC curves on the FY2024 cohort.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
zones = ["safe", "grey", "distress"]
width = 0.38
for k, (year, color) in enumerate([(2023, "tab:blue"), (2024, "tab:orange")]):
    sub = rates.loc[year].reindex(zones)
    xs = np.arange(len(zones)) + (k - 0.5) * width
    ax1.bar(xs, sub["rate"] * 100, width, color=color,
            label=f"FY{year} values, bankruptcies in {year + 1}")
    for x, (_, row) in zip(xs, sub.iterrows()):   # label every bar, including zeros
        ax1.text(x, row["rate"] * 100 + 0.04, f"{int(row.bankruptcies)}/{int(row.firms)}",
                 ha="center", va="bottom", fontsize=8)
ax1.set_xticks(range(len(zones)), zones)
ax1.set_ylabel("Firms filing for bankruptcy (%)")
ax1.set_xlabel("Z'' zone")
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.12)
ax1.legend(fontsize=8, loc="upper left")
s_fit_all = logit.predict_proba(clip(test))[:, 1]
for label, sc in [("Altman fixed weights", -test.z), ("Refit weights", s_fit_all)]:
    fpr, tpr, _ = roc_curve(test.bankrupt, sc)
    ax2.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(test.bankrupt, sc):.2f})")
ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
ax2.set_xlabel("False positive rate")
ax2.set_ylabel("True positive rate")
ax2.set_title(f"FY2024 cohort, out of time ({int(test.bankrupt.sum())} bankruptcies)")
ax2.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "backtest.png", dpi=150)

pd.set_option("display.width", 200)
print("\nBankruptcy rate by zone:")
print(rates.round(4).to_string())
print("\nAUC (0.5 = chance, 1.0 = perfect ranking):")
print(auc.round(3).to_string(index=False))
print("\nRefit minus Altman AUC on FY2024 (paired bootstrap):")
print(pd.DataFrame([diff]).round(3).to_string(index=False))
print("An interval that includes 0 means the data cannot show the refit model is better.")
if min(train.bankrupt.sum(), test.bankrupt.sum()) < 20:
    print("\nNote: fewer than 20 bankruptcies in a cohort; treat AUCs as rough (see the intervals).")
print(f"\nTables and backtest.png written to {OUT}")
