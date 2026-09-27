# Validating an LLM for Credit Risk Inputs

This project tests whether a large language model (LLM) can reliably extract the
financial figures a credit risk score needs from company annual reports (Form 10-K).
It then measures how extraction errors change each company's credit risk zone.

The answer key is the company's own XBRL data, the machine-readable numbers
every public company files with the SEC alongside the 10-K.

## Credit risk score

The project uses the Altman Z''-score (Altman, 1995), which needs only
balance-sheet and income-statement items:

    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    X1 = working capital / total assets
    X2 = retained earnings / total assets
    X3 = operating income / total assets
    X4 = stockholders' equity / total liabilities

Zones: safe above 2.60, grey from 1.10 to 2.60, distress below 1.10.

## Validation tests

1. **Accuracy.** Share of extracted values within 0.5% of the XBRL value, per field.
2. **Error types.** Each wrong value is labeled as missing, sign error,
   scale error (e.g., millions read as thousands), or wrong value.
3. **Hallucination.** The LLM reports the number as printed. If that number
   appears nowhere in the text it was given, the value is counted as hallucinated.
4. **Stability.** The same filing is run 10 times at default temperature;
   agreement is the share of runs matching the most common answer.
5. **Prompt sensitivity.** Accuracy under three wordings of the same request.
6. **Risk zone impact.** Share of firms whose Z''-zone from LLM values differs
   from the zone computed with XBRL values.

## Setup

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...

Edit `config.py`: put your name and email in `SEC_USER_AGENT` (the SEC requires it),
and set the fiscal year and model. List tickers in `data/tickers.txt`.

## Run

From the repo root:

    python -m scripts.01_download   # 10-Ks, statement excerpts, XBRL answer key
    python -m scripts.02_extract    # LLM calls, logged to extractions.jsonl
    python -m scripts.03_evaluate   # result tables in data/processed/results/
    python -m pytest tests

Step 2 skips calls already logged, so it can be stopped and resumed.
Firms that fail in step 1 are listed with the reason in `skipped.csv`.

## Known limitations of the setup

- Total liabilities is untagged for many firms. It is then derived as
  total liabilities and equity minus total equity, and the derivation is recorded
  in `total_liabilities_source`. Report accuracy separately for tagged and derived cases.
- Operating income stands in for EBIT.
- The statement excerpt is found by heading search. Spot-check a sample of
  `data/processed/excerpts/` to confirm the right tables were captured.
