"""Rule-based challenger: extract the same six values without an LLM.

For each field:
  1. Find the line whose label matches a list of common wordings.
  2. Read the numbers printed after it, one per year column.
  3. Pick the most recent year's column, using the year headers above the
     label (most filings put the latest year first, some put it last).
  4. Apply the unit from the nearest "(In thousands)" / "(In millions)" note
     above the label.
Parentheses mean a negative number. Deterministic, free, and unable to
hallucinate, but it only knows the labels and layouts written here.
"""
import re

# (statement part, label patterns in order of preference).
# Part 0 = balance sheet, 1 = income statement.
LABELS = {
    "total_assets":        (0, [r"total assets"]),
    "current_assets":      (0, [r"total current assets"]),
    "current_liabilities": (0, [r"total current liabilities"]),
    "retained_earnings":   (0, [r"retained earnings.*", r"accumulated deficit.*",
                                r"retained deficit.*", r"(accumulated|retained) (deficit|earnings).*"]),
    "total_equity":        (0, [r"total equity.*",
                                r"total (stockholders|shareholders)'? equity.*"]),
    "operating_income":    (1, [r"operating income.*", r"operating \(?loss\)?.*",
                                r"operating profit.*",
                                r"(income|loss|\(loss\) income|income \(loss\)) from operations.*",
                                r"operating (income|loss) \((loss|income)\).*"]),
}

INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00a0"), " ")
NUM = re.compile(r"^\(?\$?\s*(\d[\d,]*(?:\.\d+)?)\s*\)?$")
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
YEAR_LINE = re.compile(rf"^(?:(?:{MONTHS})\s+\d{{1,2}},?\s+)?(?:fiscal\s+)?((?:19|20)\d\d)$")
SCALES = [("billions", 1e9), ("millions", 1e6), ("thousands", 1e3)]


def _norm(line):
    """Lowercase, remove invisible characters, collapse spaces."""
    line = line.translate(INVISIBLE).lower().replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(line.split()).rstrip(":").strip()


def _is_filler(s):
    """Lines with no letters or digits: blanks, '$', '(', ')'."""
    return not re.search(r"[a-z0-9]", s)


def _numbers_after(lines, i, max_numbers=3, lookahead=14):
    """Numbers printed after line i, up to the next label. Parentheses = negative."""
    found = []
    for j in range(i + 1, min(i + 1 + lookahead, len(lines))):
        s = lines[j]
        if _is_filler(s):
            continue
        if s in ("—", "–", "-"):
            found.append(("—", 0.0))
        else:
            m = NUM.match(s)
            if not m:
                break  # reached the next label
            value = float(m.group(1).replace(",", ""))
            prev = next((lines[k] for k in range(j - 1, i, -1) if lines[k]), "")
            nxt = next((lines[k] for k in range(j + 1, min(j + 3, len(lines))) if lines[k]), "")
            neg = s.startswith("(") or s.endswith(")") or prev.endswith("(") or nxt.startswith(")")
            found.append((s, -value if neg else value))
        if len(found) == max_numbers:
            break
    return found


def _latest_column(lines, i, lookback=80):
    """0 if the most recent year is the first column, else 1."""
    years = [int(m.group(1)) for s in lines[max(0, i - lookback): i]
             if (m := YEAR_LINE.match(s))]
    # Use the last header block above the label: its first two distinct years.
    for k in range(len(years) - 1, 0, -1):
        if years[k] != years[k - 1]:
            a, b = years[k - 1], years[k]
            return 1 if a < b else 0
    return 0


def _scale(text_before):
    """Unit from the nearest unit note above the label (whitespace ignored)."""
    squashed = re.sub(r"\s+", "", text_before.lower())
    best, best_pos = ("ones", 1.0), -1
    for name, factor in SCALES:
        for pat in (f"in{name}", f"{name}ofdollars", f"dollarsin{name}"):
            pos = squashed.rfind(pat)
            if pos > best_pos:
                best, best_pos = (name, factor), pos
    if best_pos < 0 and ("$000" in squashed or "000somitted" in squashed):
        return "thousands", 1e3
    return best


def extract(statement_text):
    parts = statement_text.split("\n\n=====\n\n")
    out = {}
    for field, (part_idx, patterns) in LABELS.items():
        part = parts[part_idx] if part_idx < len(parts) else statement_text
        raw_lines = part.split("\n")
        lines = [_norm(l) for l in raw_lines]
        item = {"as_printed": None, "scale": None, "value_usd": None}
        for pat in patterns:
            for i, label in enumerate(lines):
                if "liabilities and" in label or not re.fullmatch(pat, label):
                    continue
                nums = _numbers_after(lines, i)
                if not nums:
                    continue
                col = _latest_column(lines, i)
                printed, value = nums[min(col, len(nums) - 1)]
                scale_name, factor = _scale("\n".join(raw_lines[max(0, i - 400): i]))
                item = {"as_printed": printed, "scale": scale_name,
                        "value_usd": int(round(value * factor))}
                break
            if item["value_usd"] is not None:
                break
        out[field] = item
    return {"parsed": out, "raw": "", "error": None}
