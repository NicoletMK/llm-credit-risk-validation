"""Validation tests: accuracy, error types, hallucination, stability, zone flips."""
import re
from collections import Counter

import config

SCALE_RATIOS = (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)


def classify(pred, truth, tol=config.TOLERANCE) -> str:
    """Label one extracted value against the XBRL answer key."""
    if truth is None:
        return "no_ground_truth"
    if pred is None:
        return "missing"
    if truth == 0:
        return "correct" if pred == 0 else "wrong_value"
    if abs(pred - truth) / abs(truth) <= tol:
        return "correct"
    if abs(pred + truth) / abs(truth) <= tol:
        return "sign_error"
    if pred != 0 and any(abs(pred * r - truth) / abs(truth) <= tol for r in SCALE_RATIOS):
        return "scale_error"
    return "wrong_value"


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set:
    return {m.group().replace(",", "") for m in _NUM.finditer(text)}


def is_grounded(as_printed, text_numbers: set) -> bool:
    """True if the number the LLM says it read actually appears in the text.

    A value that is wrong but grounded is a misread (wrong line or column).
    A value that appears nowhere in the text is a hallucination.
    """
    if as_printed is None:
        return True
    digits = re.sub(r"[^\d.]", "", str(as_printed)).rstrip(".")
    return digits in text_numbers


def stability(values: list) -> dict:
    """Run-to-run agreement for one field across repeated extractions."""
    counts = Counter(values)
    modal, n_modal = counts.most_common(1)[0]
    return {"n_runs": len(values), "n_distinct": len(counts),
            "agreement": n_modal / len(values), "modal_value": modal}
