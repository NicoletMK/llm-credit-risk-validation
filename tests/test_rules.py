from src.rules_extract import extract

BS = """Consolidated Balance Sheets
(In thousands)
2025
2024
Total current assets
1,354,231
1,433,350
Total assets
$
3,830,775
$
3,557,900
Total current liabilities
900,000
800,000
Accumulated deficit
(
44,212
)
(40,000)
Total stockholders' equity
100,000
90,000
Noncontrolling interests
5,000
Total equity
105,000
Total liabilities and equity
3,830,775"""
IS = """Consolidated Statements of Operations
(In thousands)
Revenue
500
Loss from operations
(12,345)
(1,000)"""


def test_rules():
    r = extract(BS + "\n\n=====\n\n" + IS)["parsed"]
    assert r["total_assets"]["value_usd"] == 3_830_775_000
    assert r["retained_earnings"]["value_usd"] == -44_212_000   # parentheses on separate lines
    assert r["total_equity"]["value_usd"] == 105_000_000        # prefers total incl. NCI
    assert r["operating_income"]["value_usd"] == -12_345_000


def test_layout_variants():
    zw = "\u200b"
    bs = f"""Consolidated Balance Sheets
(In
thousands)
September 30,
2023
2024
{zw}
Total assets
{zw}
$
2,421,305
$
2,605,068
Total current assets
100
200"""
    r = extract(bs + "\n\n=====\n\nIncome\n(in millions)\nOperating income\n5\n")["parsed"]
    assert r["total_assets"]["value_usd"] == 2_605_068_000   # latest year is the 2nd column; unit split across lines
    assert r["current_assets"]["value_usd"] == 200_000
    assert r["operating_income"]["value_usd"] == 5_000_000
