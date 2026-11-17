# Ford issuer credit research: cash resilience versus refinancing risk

A fundamental credit research case for interviewing and learning. **Historical FY2025 filing-based analysis, not a current investment recommendation.** Prepared 17 November 2026; no subsequent 2026 events or prices enter the model.

## Read first

- [Six-page research memo](Ford_Credit_Research.pdf)
- [Source-labelled reported inputs and explicit forecast assumptions](data/inputs.json)
- [Scenario results](data/scenario_results.csv)
- [EBIT / working-capital sensitivity grid](data/sensitivity.csv)
- [Illustrative bond-yield sensitivity](data/bond_sensitivity.csv)
- [Interview guide](INTERVIEW_GUIDE.md)

## Question and conclusion

Can Ford's industrial business repay the scheduled 2026 maturities while preserving a cash buffer if bond-market refinancing is unavailable?

Under the analyst assumptions, pre-funding cash is $23.46bn in the base case, $16.76bn in the downside and $13.26bn in the severe case. To preserve the $20bn management cash target, downside/severe draws are $3.52bn/$7.33bn, including an illustrative full-year 8% cost on new draws. The model assumes access to $15.5bn of corporate/supplemental facilities and excludes short-dated 2026 facility capacity. These results support conditional near-term repayment resilience, not a prediction of spreads or a finding that bonds are cheap.

## What the work demonstrates

1. Separating industrial and finance-subsidiary financial statements.
2. Reconciling consolidated operating cash flow to company adjusted free cash flow, then subtracting cash obligations excluded from the adjusted measure.
3. Distinguishing contractual maturities from carrying debt and avoiding double-counted lease principal.
4. Stressing earnings, cash conversion, capital spending and distributions.
5. Separating management targets from legal covenants and cash from conditional facility access.
6. Connecting an issuer view to bond-price sensitivity without inventing live prices or credit spreads.

## Model mechanics

All amounts are USD billions except per-100 bond prices and rates.

```
adjusted FCF = industrial adjusted EBIT + depreciation - capex
             + working-capital cash effect + Ford Credit distributions
             - cash interest - cash tax + other/timing

cash before new funding = opening cash/securities + adjusted FCF
                        - restructuring - funded pensions - dividends
                        - other non-operating cash - scheduled principal

required draw = max(0, (cash target - cash before funding) / (1 - draw interest rate))
actual draw = min(required draw, available facility capacity)
```

The base case is anchored to the midpoint of Ford's original 2026 adjusted FCF guidance. The component bridge is an analyst decomposition, not company guidance for those line items. In particular, other/timing of 4.2 is a material balancing assumption: removing it brings base pre-funding cash below the target. EBITDA proxies and interest coverage are not covenant or agency-adjusted measures. The exported EBIT/interest ratio is **before incremental draw interest**, not post-funding coverage.

No bond refinancing, asset sales or liquidity support to Ford Credit is assumed. Severe-case dividends are suspended; downside dividends are initially retained and separately sensitised. The model does not assign scenario probabilities. Annual closing cash cannot establish intrayear liquidity. Cash fungibility, facility access, actual covenant headroom and issuer-level recovery waterfalls have not been independently established.

## Source control and provenance

Primary source: [Ford 2025 Annual Report](https://s205.q4cdn.com/882619693/files/doc_financials/2025/ar/__Ford-2025-Annual-Report__.pdf), particularly printed pages 62-66, 74, 79-82 and Note18 pages 156-159. PDF page numbers are printed page + 6. `data/inputs.json` maps each input group to its source; `data/source_manifest.json` records the downloaded file hash. The source report is retained locally and excluded from the portable project ZIP. Download it from the link for independent verification.

## Reproduce

Python 3.10+; calculations and tests use the standard library only.

```bash
python src/model.py
python -m unittest discover -s src -p test_model.py -v
# Optional PDF regeneration:
pip install reportlab
python src/report.py
```

Nine tests cover the base cash bridge, debt reconciliation, subsidiary perimeter, interest-inclusive funding, funding limits, sensitivities and bond pricing identities. The exact GAAP cash-flow reconciliation is also asserted when running the model. No external API is needed to recalculate the supplied inputs.

## Limits and next research step

This project is intentionally narrower than a fully underwritten trade. Before making a live recommendation, update the financial statements, verify security-specific terms and prices, build a monthly cash forecast, compare peer spreads and analyse legal recovery priorities. The six-year coupon-date bond example is a generic fixed-cash-flow sensitivity, not the actual 2032 issue's current dirty or clean price.

AI assisted implementation and drafting. Reproduce the analysis, investigate assumptions and form your own view before using it as evidence of personal credit-research proficiency. No affiliation with Susquehanna or Ford is implied.
