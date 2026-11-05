# Data, accounting and assumptions

## Reported source data

The annual historical table uses consolidated ASML US GAAP XBRL facts from SEC-filed 2025 and 2024 Form 20-F documents. The parser excludes dimensional contexts, including country and product breakdowns, and prefers the most precise repeated fact. It does not select the first number with a matching label.

Financial figures are stored in EUR million. Share counts are in millions; the source's unit and scaling are applied before conversion. Costs are positive in the processed analytical table; calculations subtract them explicitly.

Q3 2026 values are transcribed from the SEC-filed US GAAP quarterly statements and are unaudited. The 2026 revenue and Q4 gross-margin, R&D and SG&A guidance come from the 14 October release. These are company forecasts, not actuals.

The dated prices are unadjusted daily closing values from Yahoo's chart endpoint for 4 November 2026. They are not live quotes or execution prices. Raw responses are retained. Peer-company financial facts come from SEC company-facts API snapshots.

## Definitions

| Item | Definition | Main limitation |
|---|---|---|
| Reported FCF | Operating cash flow less cash PPE and intangible purchases | Cash timing and financing-related items differ from model FCFF |
| Core NWC including advances | Trade receivables + inventory + current contract assets - payables - current/noncurrent contract liabilities | Excludes other operating assets/liabilities |
| Normalised forecast NWC | 10% of forecast revenue | Independent assumption; not a balance-by-balance roll-forward |
| FCFF | EBIT after assumed operating tax + D&A - cash capex - change in normalised NWC | Simplified taxes and reinvestment |
| Debt proxy | 2025 current and noncurrent debt less nine-month debt/finance-lease repayments | Mixes debt balances and debt/lease cash movements |
| Share proxy | Q3 2026 basic weighted-average shares, 383.1 million | Not period-end diluted shares |
| Non-operating assets | Q3 equity holdings, associates and current/noncurrent loans at book value | Fair value and restrictions not separately assessed |
| Cash reserve | EUR2 billion retained as operating cash | Analyst assumption |

## Forecast assumptions

Revenue is not modelled by shipments multiplied by equipment selling prices. The model uses aggregate growth and margin drivers because detailed order conversion, product mix and customer-level delivery schedules are not available in this project. No booking or backlog value is counted as revenue automatically.

The base growth assumptions reach EUR58.9 billion of revenue in 2030, within the older EUR44-60 billion Investor Day range. The stronger upside extends beyond that range and assumes better execution and demand. Downside margins fall below the older 56-60% margin scenarios.

The 9.5% WACC and 3% terminal growth rate are illustrative. A full market calibration is an extension, not a completed part of this model. The terminal year's NOPAT, net capex and working-capital investment are rebuilt from the stable operating drivers. The model does not simply grow final-year FCFF.

Q4 FCFF uses company revenue/margin/expense guidance and analyst estimates of D&A, capex, tax and working capital. The remaining 57 days are prorated from a 95-day fiscal quarter, assuming even cash generation. Actual cash timing may be uneven. It is discounted to 4 November 2026 with year-end timing for subsequent annual cash flows.

## Comparables

FY2025 financial denominators are deliberately labelled. The table does not claim to be current TTM. Peer year-ends differ, later buybacks are not reconstructed, cash may exclude short-term investments, and restated split-adjusted shares can have later filing dates than the underlying fiscal year. Fiscal-year share counts are used to avoid silently mixing weighted-average earnings denominators with period-end market-cap calculations. Even then, the market-cap proxy is not an exact November share count.

Multiples use native reporting currencies, so currency cancels within each ratio. The absolute financial amounts are not aggregated. The DCF equity bridge uses the latest disclosed ASML snapshot and different proxies, and therefore should not be expected to match the historical comparables bridge.

## Source provenance

`data/raw/manifest.json` contains download URLs, hashes and byte counts. Large filings and company-facts files are excluded from version control; processed source extracts are included. Source APIs may be restated later. The checked-in workbook, data and reports preserve the delivered analytical snapshot.
