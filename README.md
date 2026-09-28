# Validating an LLM That Extracts Credit Risk Inputs from 10-K Filings

Can a large language model (LLM) read a company's annual report accurately enough to feed a credit risk score? I tested one on 153 U.S. companies and scored every number it extracted against the company's own filed data.

**Result.** On the final run, the LLM matched the filed value for [FINAL]% of fields and put [FINAL] of 153 firms in the wrong credit risk zone. The first run looked worse (8.1% of firms in the wrong zone), but 54 of the 56 missing values were never in the text the LLM received. The weak point was the step that finds the financial statements inside the filing, not the model.

Full write-up: [validation report (PDF)](report/validation_report.pdf)

## What I tested

The LLM reads the balance sheet and income statement from each company's FY2024 10-K and returns six numbers. These feed the Altman Z''-score, a standard credit risk score that sorts firms into safe, grey, and distress zones.

The answer key is each company's XBRL data. These are the machine-readable numbers every public company files with the SEC alongside its 10-K, so the LLM's output can be checked value by value.

The validation covers six tests:

1. Accuracy: share of values within 0.5% of the filed value.
2. Error types: missing, sign error, scale error (millions read as thousands), or wrong value.
3. Hallucination: whether each reported number appears anywhere in the text the LLM was given.
4. Stability: agreement across 10 repeated runs of the same filing.
5. Prompt sensitivity: accuracy under three wordings of the same request.
6. Risk zone impact: share of firms whose zone changes when LLM values replace filed values.

## Findings

[FINAL: 3–5 sentences in your own words. Candidates from the runs so far:]

- The LLM did not invent numbers. When the statement was missing from its input, it returned null.
- Retrieval errors drove almost all failures in run 1. Fixing the statement search raised field accuracy from 93.7% to [FINAL]%.
- The same bug had also excluded 5 eligible firms, so retrieval errors biased the sample as well as the scores.
- Stability: [FINAL]. Prompt sensitivity: [FINAL].

## Sample

Retail (SIC 5200–5999) and software and IT services (SIC 7370–7379) firms that file a 10-K, report every Z'' input, and hold at least $50M in assets. I sampled up to 60 firms per zone so the distress and grey zones are well represented. Of 180 sampled firms, 27 were excluded: 21 print no operating income line, and the rest had no FY2024 10-K. Exclusions are listed with reasons in `data/processed/skipped.csv`.

## Design decisions

- Z'' (non-manufacturer version) fits the two sectors, since both are non-manufacturing.
- The LLM extracts only numbers printed on the statements. Total liabilities is computed as total assets minus total equity, because many firms never print that line.
- Operating income stands in for EBIT.
- Foreign firms filing Form 20-F are out of scope.

## Limitations

- Z'' was estimated on older firm samples. Young software firms with large accumulated deficits score as distressed even when they hold plenty of cash.
- The answer key is the company's own tagging, which can differ from the printed statement.
- Results are for one model ([MODEL]) at one point in time.

## Reproduce

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=...        # and set SEC_USER_AGENT in config.py

python -m scripts.00_build_sample    # sample from SEC bulk data
python -m scripts.01_download        # 10-Ks, statement excerpts, answer key
python -m scripts.02_extract         # LLM calls (resumable)
python -m scripts.03_evaluate        # result tables in data/processed/results/
python -m pytest tests
```

Full run: about 2,000 LLM calls, roughly $4 at [MODEL] prices.
