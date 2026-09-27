# Model Validation Report: LLM Extraction of Credit Risk Inputs

Target length: 4-6 pages. Structured after the model validation elements in
the Federal Reserve's SR 11-7 guidance (conceptual soundness, outcomes analysis,
ongoing monitoring).

## 1. Model Overview
Purpose and intended use: extracting Z''-score inputs from 10-K filings for credit screening.
Model: which LLM and version, prompt, temperature. Out of scope: autonomous credit decisions.

## 2. Data
Sample: sectors, fiscal year, number of firms kept and excluded (with reasons).
Answer key: XBRL company facts. Fields where the answer key is derived.

## 3. Conceptual Soundness
Assumptions and where they can fail:
- The excerpt contains the right statements.
- The LLM reads the stated unit scale correctly.
- Operating income is a fair EBIT proxy.
- Z'' was built on older firm samples; state its limits for current firms.

## 4. Outcomes Analysis
### Accuracy by field
### Error types
### Hallucination
### Stability across runs
### Prompt sensitivity
### Risk zone impact (headline result)
Zone confusion table; examples of firms that changed zone and which error caused it.

## 5. Limitations
Where the model should not be used or trusted.

## 6. Recommendations and Controls
Controls that would catch the observed errors before use,
e.g., check that total assets equals total liabilities plus equity,
reconcile against XBRL where available, flag low run-to-run agreement.

## 7. Ongoing Monitoring
What to track if this were in production: accuracy on a labeled sample each
quarter, zone flip rate, behavior after a model version change.
