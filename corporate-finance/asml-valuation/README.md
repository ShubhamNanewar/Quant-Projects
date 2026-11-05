# ASML: financial analysis and valuation

An independent company-analysis project connecting ASML's business, financial statements, operating assumptions and valuation. Prepared for learning and interview discussion, using information available by **4 November 2026**.

## Start here

- [Research report](reports/RESEARCH_REPORT.md): question, method, results and limitations.
- [Six-page brief](reports/ASML_Investment_Brief.pdf): a short presentation of the analysis.
- [Excel valuation model](model/ASML_Valuation.xlsx): editable operating assumptions, three cases and a native sensitivity table.
- [Browser report](reports/REPORT.html): local HTML with a vector chart.
- [Interview guide](docs/INTERVIEW_GUIDE.md): how to explain the model without overstating it.
- [Source and assumption notes](docs/DATA_AND_ASSUMPTIONS.md).

## What the project covers

1. Extract consolidated US GAAP facts from SEC-filed ASML annual reports for 2023-2025. Dimensional facts are excluded; repeated facts are resolved by reported precision.
2. Reconcile historical gross profit, EBIT and the balance-sheet equation. Calculate reported free cash flow and a clearly defined core working-capital measure.
3. Incorporate Q3 2026 financials and the October 2026 revenue guidance.
4. Forecast revenue, gross profit, R&D, SG&A, tax, D&A, capex and normalised working capital for 2027-2031.
5. Calculate unlevered free cash flow, remaining-2026 cash flow, terminal value and an enterprise-to-equity bridge.
6. Compare base, downside and upside cases. Test a 5-by-5 WACC / terminal-growth grid.
7. Compare historical valuation multiples with Applied Materials, Lam Research and KLA, with fiscal-period and share-count limitations stated.

## Main finding

With an illustrative 9.5% discount rate and 3% terminal growth, the base model produces approximately **EUR765 per share**, compared with a dated observed close of approximately **EUR1,338**. Downside and upside operating cases produce approximately EUR558 and EUR943. The base terminal value is roughly three quarters of enterprise value.

This is not evidence by itself that the market is wrong. It shows how much the conclusion depends on future margins, growth, reinvestment and the discount rate. The operating upside case holds WACC constant; it is not a probability-weighted price target or a comprehensive bull case.

## Using the workbook

- Change `Assumptions!D4`: **1 Base, 2 Downside, 3 Upside**. All three use the same forecast and DCF calculations.
- Change the blue case assumptions in the rows below each active driver.
- `DCF!D6` and `DCF!D7` control the discount rate and terminal growth.
- `DCF!F48:K53` is the native two-variable sensitivity table, linked to those inputs and the calculated per-share output.
- `Summary` shows the selected case, valuation, dated price and reported financial history.
- `Actuals` retains reported inputs and source references. Green working cells link across sheets; blue numbers are inputs. The output summary uses dark text.

The frozen research report and PDF do not update when Excel assumptions change. Rebuild them after changing the Python assumptions if you want all outputs to agree again. In Excel, enable automatic calculation; data-table recalculation may depend on the application's calculation settings.

## Important boundaries

This is an **operating forecast and DCF**, not a full three-statement model. It does not include a complete debt, retained-earnings or projected balance-sheet schedule.

The WACC, normalised working capital, minimum cash reserve and quarter-stub allocations are analyst assumptions. Ending shares and debt use disclosed proxies and an estimated bridge. Non-operating assets are added at book value. Those choices are visible and can materially affect equity value.

The peer table uses November 2026 prices and **FY2025 denominators**, including FY2025 share counts and balance sheets. It is a historical diagnostic, not matched TTM or forward trading comparables. Differences in fiscal year-end, subsequent buybacks, cash definitions and business mix matter. No peer-multiple price target is claimed.

The model has been checked in the bundled calculation engine and against Python. It has not been opened and tested in Microsoft Excel desktop.

## Reproduce the analysis

Python 3.11+:

```bash
pip install -r requirements.txt
python src/fetch_sources.py
python src/analyse.py
python -m unittest discover -s tests -v
python src/build_report.py
```

Large raw filings are excluded from Git; the downloader retrieves them and checks cached files. The manifest records hashes and source URLs. The downloaded SEC company-facts API can be revised over time. The committed processed extracts and result files preserve the delivered snapshot. A later rebuild from refreshed sources may differ.

The Excel builder uses the Codex bundled `@oai/artifact-tool` runtime:

```bash
node src/build_workbook.mjs
```

It requires that package to be available in the Node environment. The workbook is usable independently of the builder. The Python calculation and report do not require the Excel authoring package.

## Checks

- Six Python tests cover source controls, statement identities, cash-flow identities, case ordering, discount/growth sensitivity and the terminal-growth guard.
- All three Excel case values agree with Python to within EUR0.01 per share.
- All 25 native sensitivity-table values agree with Python to the same tolerance.
- A change to the final-year growth input changes the valuation, then is restored.
- The workbook is scanned for formula errors and rendered for layout review.

See `reports/workbook_checks.json` for the recorded case checks.

## Possible CV description

> Built an ASML operating forecast and DCF valuation using SEC-filed financial statements; assessed growth, margins, reinvestment and terminal-value sensitivity, with a documented historical peer comparison.

Describe this as independent project work and be prepared to explain the assumptions and approximations. Do not describe it as transaction execution, professional equity research or a full three-statement model.
