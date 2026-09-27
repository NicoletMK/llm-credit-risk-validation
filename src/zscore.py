"""Altman Z''-score (Altman 1995 non-manufacturer version).

Uses book equity, so every input comes straight from the filing:
  Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
  X1 = working capital / total assets
  X2 = retained earnings / total assets
  X3 = EBIT / total assets        (operating income as the EBIT proxy)
  X4 = book equity / total liabilities
Zones: safe > 2.60, grey 1.10-2.60, distress < 1.10.
"""

SAFE, DISTRESS = 2.60, 1.10


def z_double_prime(v: dict):
    """v: dict with the ground_truth.FIELDS keys. Returns None if an input is missing."""
    need = ["total_assets", "current_assets", "current_liabilities",
            "total_liabilities", "retained_earnings", "stockholders_equity",
            "operating_income"]
    if any(v.get(k) is None for k in need) or not v["total_assets"] or not v["total_liabilities"]:
        return None
    ta = v["total_assets"]
    x1 = (v["current_assets"] - v["current_liabilities"]) / ta
    x2 = v["retained_earnings"] / ta
    x3 = v["operating_income"] / ta
    x4 = v["stockholders_equity"] / v["total_liabilities"]
    return 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4


def zone(z):
    if z is None:
        return None
    if z > SAFE:
        return "safe"
    if z < DISTRESS:
        return "distress"
    return "grey"
