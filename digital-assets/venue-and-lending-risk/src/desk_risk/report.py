"""Generate a readable report from the saved analysis tables."""
import base64
from html import escape
import json
from pathlib import Path

import pandas as pd

from .analyze import ROOT


def markdown_table(frame):
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    rule = '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'
    rows = ['| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |'
            for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, rule] + rows)


def main():
    run = (ROOT / 'data/LATEST').read_text().strip()
    out = ROOT / 'outputs' / run
    s = json.loads((out / 'summary.json').read_text())
    books = pd.read_csv(out/'books.csv')
    candles = pd.read_csv(out/'candles.csv')
    coverage = pd.read_csv(out/'position_coverage.csv')
    stress = pd.read_csv(out/'borrower_stress.csv')
    vaults = pd.read_csv(out/'vaults.csv')
    oracles = pd.read_csv(out/'oracles.csv')
    positions = pd.read_csv(out/'positions.csv')
    alerts = pd.read_csv(out/'alerts.csv')
    opportunities = pd.read_csv(out/'opportunities.csv')
    rejected_pairs = int((~opportunities.groupby('round').timing_gate.first()).sum())
    date_label = pd.Timestamp(s['retrieved_at']).strftime('%d %B %Y')
    stress10 = stress[stress.shock == .1]
    stress_sentence = '; '.join(f"{row.collateral}: {row.breached_debt_fraction:.2%}" for row in stress10.itertuples())
    exit_low = vaults.loc[vaults.immediate_withdrawal_fraction.idxmin()]
    concentrated = vaults.loc[vaults.largest_collateral_weight.idxmax()]
    sections = []
    def paragraph(title, text):
        sections.append(('paragraph', title, text))
    def table(title, frame):
        sections.append(('table', title, frame))
    def figure(name, caption):
        sections.append(('figure', caption, name))

    paragraph('Research question',
        'Can public exchange and lending-protocol data support a useful desk screen for liquidity, pricing gaps and collateral risk? '
        'This project builds two read-only exchange adapters and a Morpho data integration, then follows the observations through validation, '
        'calculations and a desk decision. It uses no private accounts, wallet keys or trading permissions. '
        'The purpose is to understand what the data can support and where it stops being sufficient.')
    paragraph('Main findings',
        f'The collection contains {s["books"]} ETH/USD order books across {s["snapshot_seconds"]:.1f} seconds, '
        f'{s["markets"]} Ethereum lending markets, {s["positions"]:,} borrower positions and {s["vaults"]} USDC vaults. '
        f'The largest gross cross-venue difference was {s["max_gross_bps"]:.2f} basis points. '
        f'With an assumed fee of 10 basis points per leg, the best net difference was {s["max_net_bps_at_10bps_per_leg"]:.2f} basis points. '
        'The sampled price differences therefore do not support a trade at that cost assumption. '
        'Borrower stress and vault withdrawal capacity vary substantially even where the loan token is the same.')
    table('Collection coverage', pd.DataFrame([
        ['Collection ID', run], ['Final retrieval time (UTC)', s['retrieved_at']],
        ['Archived HTTP responses', str(s['requests'])],
        ['Paired exchange observations', str(s['books']//2)],
        ['Coinbase completed hourly bars', str(s['candles_by_venue']['coinbase'])],
        ['Kraken completed hourly bars', str(s['candles_by_venue']['kraken'])],
        ['Maximum book receipt skew', f'{s["max_receipt_skew_seconds"]:.3f} seconds'],
        ['Paired Morpho oracle observations', str(s['oracle_observations_paired'])],
    ], columns=['Measure','Value']))
    paragraph('Data provenance',
        'Coinbase Exchange and Kraken supply public ETH/USD books, product metadata and recent hourly candles. '
        'Morpho supplies indexed market, position, vault and oracle configuration data. '
        'Each raw response has a request URL, retrieval times and SHA-256 checksum in the manifest. '
        'The archive is a reproducibility record, not a guarantee that an external provider is always correct. '
        'The appendix links the exact queries, data dictionary, methods, source references and contract review.')
    paragraph('Exchange liquidity',
        f'The median displayed spread was {s["spread_median_bps"]["coinbase"]:.3f} bps on Coinbase and '
        f'{s["spread_median_bps"]["kraken"]:.3f} bps on Kraken. '
        f'{rejected_pairs} paired observations fail the preliminary timing gate because a source quote is too old under the two-second rule. '
        'They remain in the descriptive tables and charts, but are flagged against execution use. '
        'A narrow best spread is only the first level of the book. The depth calculation measures how much displayed USD value is available '
        'within 10 bps of the midpoint. The sizing screen walks the book for the same ETH quantity on each leg. '
        'It rejects an incomplete fill rather than assuming the remaining quantity trades at the last observed price.')
    figure('exchange_liquidity.png', 'Observed spreads and ask-side depth. The window is short and descriptive.')
    paragraph('Costs and trade size',
        'Four target notionals, from $1,000 to $100,000, are evaluated in both directions. '
        'Six fee assumptions are applied independently to both legs. Fees are not taken from an authenticated account schedule. '
        'Inventory is assumed to be pre-funded. Leg risk, borrowing, transfers, custody limits and adverse selection remain outside the calculation. '
        'A candidate must survive those checks before it could become an executable proposal.')
    figure('fee_sensitivity.png', 'Median net difference and observed range at a $10,000 target. The zero line separates positive and negative calculations, not verified executable trades.')
    paragraph('Historical context',
        f'The available hourly data span {candles.utc.min()} to {candles.utc.max()}. '
        'Kraken provides a longer sample than Coinbase in this capture. Incomplete candles are removed; missing hours are not filled. '
        'The rolling volatility chart describes recent market conditions. It is not a fitted forecast or evidence of a trading signal.')
    figure('price_and_volatility.png', 'Hourly prices and a trailing 24-hour realised-volatility estimate.')
    paragraph('Lending markets',
        'Market utilisation is recomputed from the raw supply and borrow balances. '
        f'The maximum discrepancy against the provider value is {s["max_utilization_reconciliation_error"]:.2g}. '
        f'{s["high_utilization_markets"]} of the 30 sampled markets exceed the 95% research review threshold. '
        'That does not establish low overall risk. High collateral concentration, liquidation sensitivity, issuer risk and limited vault exits can exist below that threshold. '
        'The selection contains large listed markets, which is a deliberate selection bias.')
    figure('market_utilization.png', 'The 15 largest sampled markets by supplied USD. Market ID fragments distinguish similar pairs.')
    table('Borrower coverage', coverage.assign(
        market_id=coverage.market_id.str[:12],
        sampled_debt=coverage.sampled_debt.map(lambda v: f'{v:,.2f}'),
        market_borrow=coverage.market_borrow.map(lambda v: f'{v:,.2f}'),
        debt_coverage=coverage.debt_coverage.map(lambda v: f'{100*v:.6f}%')).rename(columns={
            'market_id':'Market ID prefix','sampled_debt':'Sampled USDC debt','positions':'Positions',
            'market_borrow':'Reported USDC debt','debt_coverage':'Coverage'}))
    paragraph('Borrower stress',
        'Debt is derived from borrow shares with the contract\'s upward rounding convention. Collateral value and supported debt are rounded down. '
        'The stress test then reduces the oracle price while holding debt and collateral quantity constant. '
        f'At a 10% shock, the shares of sampled debt beyond the threshold are {stress_sentence}. '
        'These are conditional sensitivities, not comparative loss probabilities. A 10% principal-token price decline and a 10% BTC decline have different economic meanings. '
        'No liquidation process, collateral sale or recovery is simulated.')
    chosen = stress[stress.shock.isin([.1,.2,.3])][['collateral','shock','debt_beyond_threshold','breached_debt_fraction']].copy()
    chosen['shock'] = chosen.shock.map(lambda v: f'{v:.0%}')
    chosen['debt_beyond_threshold'] = chosen.debt_beyond_threshold.map(lambda v:f'{v:,.0f}')
    chosen['breached_debt_fraction'] = chosen.breached_debt_fraction.map(lambda v:f'{v:.2%}')
    table('Static shock results', chosen.rename(columns={'collateral':'Collateral','shock':'Price decline',
          'debt_beyond_threshold':'USDC debt beyond threshold','breached_debt_fraction':'Share of sampled debt'}))
    figure('borrower_stress.png', 'Threshold exposure increases as the collateral-price shock increases. Exposure is not expected loss.')
    paragraph('Vault concentration and withdrawals',
        'The five sampled vaults allocate the same loan token across different markets. '
        f'{exit_low.vault} has the lowest estimated immediate withdrawal capacity at {exit_low.immediate_withdrawal_fraction:.2%} of saved assets. '
        f'The highest estimate is {vaults.immediate_withdrawal_fraction.max():.2%}. '
        f'{concentrated.vault} has the largest collateral concentration at {concentrated.largest_collateral_weight:.2%}. '
        'These figures describe the archived configuration. They are not a judgment of a curator\'s strategy or a current recommendation. '
        'The withdrawal estimate assumes no competing withdrawals and excludes reallocation. Capacity across vaults cannot be added when they share markets.')
    v = vaults[['vault','market_count','largest_collateral_weight','immediate_withdrawal_fraction','timelock_hours']].copy()
    for col in ['largest_collateral_weight','immediate_withdrawal_fraction']:
        v[col] = v[col].map(lambda x:f'{x:.2%}')
    table('Vault observations', v.rename(columns={'vault':'Vault','market_count':'Active markets',
          'largest_collateral_weight':'Largest collateral weight','immediate_withdrawal_fraction':'Withdrawal estimate',
          'timelock_hours':'Timelock hours'}))
    figure('vault_risk.png', 'Concentration and static withdrawal capacity answer different risk questions.')
    paragraph('Oracle observations',
        f'All {len(oracles)} markets were matched across the two observations. '
        f'{int(oracles.state_advanced.sum())} indexed state timestamps advanced. '
        f'The largest absolute oracle-price change was {oracles.price_change_bps.abs().max():.4f} bps. '
        'Feed and conversion-vault dependencies are recorded separately. A small price change or an unchanged timestamp does not establish feed freshness. '
        'The API response does not supply the underlying feed round timestamps needed for that conclusion. '
        'Several prices include conversion-vault dependencies, which need additional review beyond a single market-price comparison.')
    paragraph('Data limitations that affect interpretation',
        f'Borrower and market-state timestamps differ by {positions.state_time_gap_seconds.min():.0f} to '
        f'{positions.state_time_gap_seconds.max():.0f} seconds in the saved sample. '
        'Pagination is not block-pinned. Debt coverage is close to 100%, but that alone cannot prove an atomic snapshot. '
        'Health factors for dust debts can differ materially because a tiny integer debt change has a large ratio effect. '
        'The exchange window covers only about three minutes, the candle history is short, and there is no live execution validation. '
        'The top-market and top-vault selection omits smaller venues and protocols. All of these constraints remain visible in the project rather than being absorbed into a performance claim.')
    counts = alerts.groupby('rule').size().reset_index(name='Count')
    table('Research alerts', counts.rename(columns={'rule':'Rule'}))
    paragraph('Desk decision',
        'Proceed with read-only research and further observation. Do not approve live execution or an allocation from this evidence. '
        'The sampled exchange gap does not survive the stated fee scenario. The lending risk screen is useful, but feed freshness, '
        'same-block reconciliation, deployed contract verification and collateral-specific liquidation costs remain open. '
        'The technical backlog turns each open issue into an owner and an acceptance criterion.')
    paragraph('What this project demonstrates',
        'Public API integration, explicit handling of data types and time, depth-aware pricing calculations, protocol-specific integer arithmetic, '
        'borrower stress analysis, vault diagnostics and reproducible reporting. It is a personal research implementation. '
        'It does not demonstrate professional production connectivity, a Solidity security audit or a live trading track record.')

    links = [('Data dictionary','docs/DATA_DICTIONARY.md'),('Methods','docs/METHODOLOGY.md'),
             ('Contract reading','docs/CONTRACT_REVIEW.md'),('Desk decision and backlog','docs/DESK_DECISION.md'),
             ('Source references','docs/SOURCES.md'),('Learning guide','docs/LEARNING_GUIDE.md')]
    md = ['# Crypto Venue Data and Morpho Lending Risk', '', f'Research report | {date_label}', '']
    html = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Crypto Venue Data and Morpho Lending Risk</title><style>',
        'body{margin:0;background:#f3f5f7;color:#172635;font:17px/1.7 system-ui,sans-serif}main{max-width:1050px;margin:40px auto;background:white;padding:48px 56px}h1{font-size:38px;line-height:1.15;max-width:850px}h2{font-size:24px;margin-top:42px;color:#174b65}p{max-width:900px}img{max-width:100%;height:auto}figure{margin:28px 0}figcaption{font-size:14px;color:#536574}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid #dbe3e8;text-align:left}th{background:#edf4f7}a{color:#146080}.kicker{font-size:14px;letter-spacing:1px;text-transform:uppercase;color:#517080}.table{overflow-x:auto}@media(max-width:700px){main{margin:0;padding:24px}h1{font-size:30px}}@media print{body{background:white}main{margin:0;padding:0}figure,table{break-inside:avoid}}',
        f'</style><main><p class="kicker">Public-data research | {date_label}</p><h1>Crypto Venue Data and Morpho Lending Risk</h1>']
    for kind, title, content in sections:
        if kind == 'paragraph':
            md += ['## '+title, '', content, '']
            html += ['<h2>'+escape(title)+'</h2><p>'+escape(content)+'</p>']
        elif kind == 'table':
            md += ['### '+title, '', markdown_table(content), '']
            html += ['<h2>'+escape(title)+'</h2><div class="table">'+content.to_html(index=False, border=0, escape=True)+'</div>']
        else:
            rel = f'figures/{run}/{content}'
            md += [f'![{title}]({rel})','',title,'']
            encoded = base64.b64encode((ROOT/rel).read_bytes()).decode()
            html += [f'<figure><img alt="{escape(title)}" src="data:image/png;base64,{encoded}"><figcaption>{escape(title)}</figcaption></figure>']
    md += ['## Appendices', ''] + [f'- [{label}]({path})' for label,path in links]
    html += ['<h2>Appendices</h2><ul>'] + [f'<li><a href="{path}">{label}</a></li>' for label,path in links] + ['</ul></main></html>']
    (ROOT/'REPORT.md').write_text('\n'.join(md)+'\n')
    (ROOT/'REPORT.html').write_text('\n'.join(html))
    print(ROOT/'REPORT.html')


if __name__ == '__main__':
    main()
