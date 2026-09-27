"""Ask an LLM to extract the Z''-score inputs from 10-K statement text."""
import json
import re

import anthropic

import config
from src.ground_truth import FIELDS

FIELD_NAMES = list(FIELDS)

OUTPUT_SPEC = f"""Return only a JSON object, no other text. Keys: {FIELD_NAMES}.
Each value is an object with:
  "as_printed": the number exactly as it appears in the statement (string), or null
  "scale": the unit the statement states, one of "ones", "thousands", "millions", "billions"
  "value_usd": the number converted to whole US dollars (integer), or null
Use the most recent fiscal year column. Use null if the item is not in the text.
Write negative numbers (including those shown in parentheses) with a minus sign."""

# Prompt variants for the sensitivity test. "base" is the main prompt;
# the others ask for the same thing in different words.
PROMPTS = {
    "base": (
        "You are extracting figures from a company's annual report (Form 10-K).\n"
        "From the balance sheet and income statement below, extract: total assets, "
        "total current assets, total current liabilities, total liabilities, "
        "retained earnings (accumulated deficit), total stockholders' equity "
        "attributable to the parent, and operating income (loss).\n\n"
    ),
    "terse": (
        "Extract these values from the 10-K financial statements below: "
        "total assets, current assets, current liabilities, total liabilities, "
        "retained earnings, stockholders' equity, operating income.\n\n"
    ),
    "analyst": (
        "You are a credit analyst computing an Altman Z''-score. Read the "
        "financial statements below and pull the inputs you need: total assets, "
        "current assets, current liabilities, total liabilities, retained earnings, "
        "shareholders' equity, and operating income (EBIT proxy).\n\n"
    ),
}

_client = None


def _client_():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def parse_json(text: str) -> dict:
    text = re.sub(r"```(json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def extract(statement_text: str, prompt: str = "base",
            temperature: float = 0.0) -> dict:
    """One extraction call. Returns {field: {...}} plus raw text and errors."""
    msg = _client_().messages.create(
        model=config.LLM_MODEL,
        max_tokens=1500,
        temperature=temperature,
        messages=[{"role": "user", "content":
                   PROMPTS[prompt] + OUTPUT_SPEC + "\n\n---\n" + statement_text}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    try:
        return {"parsed": parse_json(raw), "raw": raw, "error": None}
    except (json.JSONDecodeError, ValueError) as e:
        return {"parsed": None, "raw": raw, "error": f"parse failure: {e}"}
