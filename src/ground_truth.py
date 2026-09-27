"""Answer key: the company's own XBRL-tagged values for each field."""
from datetime import date

from src.edgar import company_facts

# Field -> (XBRL tags to try in order, "instant" balance-sheet or "duration" full-year).
# Only items printed on the face of the statements. Total liabilities is
# computed in zscore.py as total assets minus total equity.
FIELDS = {
    "total_assets":        (["Assets"], "instant"),
    "current_assets":      (["AssetsCurrent"], "instant"),
    "current_liabilities": (["LiabilitiesCurrent"], "instant"),
    "retained_earnings":   (["RetainedEarningsAccumulatedDeficit"], "instant"),
    # Total equity including noncontrolling interests. Firms with no
    # noncontrolling interests tag only StockholdersEquity.
    "total_equity":        (["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                             "StockholdersEquity"], "instant"),
    "operating_income":    (["OperatingIncomeLoss"], "duration"),
}


def _pick(rows, accn, kind):
    """Current-year value from this filing (filings also repeat prior years)."""
    rows = [r for r in rows if r.get("accn") == accn]
    if kind == "instant":
        rows = [r for r in rows if "start" not in r]
    else:
        def days(r):
            return (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days
        rows = [r for r in rows if "start" in r and 330 <= days(r) <= 400]
    if not rows:
        return None
    return max(rows, key=lambda r: r["end"])["val"]


def xbrl_values(cik, accn):
    """Return {field: value in USD or None} and {field_source: tag used}."""
    gaap = company_facts(cik)["facts"]["us-gaap"]
    out = {}
    for field, (tags, kind) in FIELDS.items():
        out[field], out[f"{field}_source"] = None, None
        for tag in tags:
            rows = gaap.get(tag, {}).get("units", {}).get("USD", [])
            val = _pick(rows, accn, kind)
            if val is not None:
                out[field], out[f"{field}_source"] = val, tag
                break
    return out
