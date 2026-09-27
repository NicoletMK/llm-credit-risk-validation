"""Project settings. Edit these before running."""

# SEC requires a User-Agent with your name and email on every request.
SEC_USER_AGENT = "Nicole Tian nicole.tian@my.utsa.edu"

# Sample: tickers to include. Keep to 1-2 sectors, ~100-200 firms.
TICKERS_FILE = "data/tickers.txt"

# Fiscal year of the 10-K to use for every firm.
FISCAL_YEAR = 2024

# LLM settings. Set OPENROUTER_API_KEY in your environment.
LLM_MODEL = "MODEL_ID"
N_STABILITY_RUNS = 10       # repeated runs per filing for the stability test
STABILITY_TEMPERATURE = 1.0 # default sampling, to measure run-to-run variation

# Relative error below which an extracted value counts as correct.
TOLERANCE = 0.005  # 0.5%

# Max characters of filing text sent to the LLM.
MAX_CHARS = 60_000
