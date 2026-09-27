"""Ground-truth financial values from the company's own XBRL tags.

XBRL values are the numbers the company itself tagged and filed with the
SEC, so they serve as the answer key for the LLM's extraction.
"""
from datetime import date

from src.edgar import company_facts

# Field -> XBRL tags to try, in order. "instant" = balance-sheet date,
# "duration" = full-year flow (income statement).
FIELDS = {
    "total_assets":        (["Assets"], "instant"),
    "current_assets":      (["AssetsCurrent"], "instant"),
    "current_liabilities": (["LiabilitiesCurrent"], "instant"),
    "total_liabilities":   (["Liabilities"], "instant"),
    "retained_earnings":   (["RetainedEarningsAccumulatedDeficit"], "instant"),
    "stockholders_equity": (["StockholdersEquity"], "instant"),
    "operating_income":    (["OperatingIncomeLoss"], "duration"),
}


def _pick(rows: list, accn: str, kind: str):
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


def xbrl_values(cik: int, accn: str) -> dict:
    """Return {field: value in USD or None} plus {field_source: tag used}."""
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

    # Many firms never tag total liabilities. Derive it and record that we did,
    # since a derived answer key is weaker than a tagged one.
    if out["total_liabilities"] is None:
        tle = _pick(gaap.get("LiabilitiesAndStockholdersEquity", {})
                    .get("units", {}).get("USD", []), accn, "instant")
        eq_incl_nci = _pick(gaap.get(
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            {}).get("units", {}).get("USD", []), accn, "instant")
        eq = eq_incl_nci if eq_incl_nci is not None else out["stockholders_equity"]
        if tle is not None and eq is not None:
            out["total_liabilities"] = tle - eq
            out["total_liabilities_source"] = "derived: LiabilitiesAndStockholdersEquity - equity"
    return out
