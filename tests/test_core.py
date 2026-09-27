from src.validate import classify, is_grounded, numbers_in, stability
from src.zscore import z_double_prime, zone


def test_classify():
    assert classify(100.2, 100) == "correct"
    assert classify(-100, 100) == "sign_error"
    assert classify(100_000, 100_000_000) == "scale_error"
    assert classify(None, 5) == "missing"
    assert classify(80, 100) == "wrong_value"


def test_grounding():
    nums = numbers_in("Total assets $ 87,764 and (1,234.5)")
    assert is_grounded("87,764", nums)
    assert is_grounded("(1,234.5)", nums)
    assert not is_grounded("88,000", nums)


def test_zscore():
    v = dict(total_assets=100, current_assets=40, current_liabilities=20,
             retained_earnings=30, total_equity=50, operating_income=10)
    z = z_double_prime(v)
    # 6.56*.2 + 3.26*.3 + 6.72*.1 + 1.05*1 = 1.312+.978+.672+1.05
    assert abs(z - 4.012) < 1e-9
    assert zone(z) == "safe" and zone(1.5) == "grey" and zone(0.5) == "distress"


def test_stability():
    s = stability([1, 1, 1, 2])
    assert s["agreement"] == 0.75 and s["n_distinct"] == 2
