"""Altman Z''-score (Altman 1995, non-manufacturer version).

  Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
  X1 = (current assets - current liabilities) / total assets
  X2 = retained earnings / total assets
  X3 = operating income / total assets
  X4 = total equity / total liabilities
Total liabilities = total assets - total equity.
Zones: safe > 2.60, grey 1.10-2.60, distress < 1.10.
"""

SAFE, DISTRESS = 2.60, 1.10
INPUTS = ["total_assets", "current_assets", "current_liabilities",
          "retained_earnings", "total_equity", "operating_income"]


def z_double_prime(v):
    if any(v.get(k) is None for k in INPUTS):
        return None
    ta = v["total_assets"]
    tl = ta - v["total_equity"]
    if not ta or tl <= 0:
        return None
    x1 = (v["current_assets"] - v["current_liabilities"]) / ta
    x2 = v["retained_earnings"] / ta
    x3 = v["operating_income"] / ta
    x4 = v["total_equity"] / tl
    return 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4


def zone(z):
    if z is None:
        return None
    if z > SAFE:
        return "safe"
    if z < DISTRESS:
        return "distress"
    return "grey"
