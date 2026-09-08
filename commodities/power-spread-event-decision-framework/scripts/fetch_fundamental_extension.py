"""Fetch public data in restartable 28-day chunks; no credentials required."""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import io
import time
import zipfile
from pathlib import Path
import pandas as pd
import requests
from caiso_fetch import oasis_timestamp, standardize_lmp

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/raw/fundamental_extension'
CACHE.mkdir(parents=True, exist_ok=True)
URL = 'https://oasis.caiso.com/oasisapi/SingleZip'
REQUEST_LOCK = Lock()
LAST_REQUEST = 0.0


def fetch(query, version, market, start, end):
    path = CACHE / f'{query}_{start.strftime("%Y%m%d")}.csv'
    if path.exists():
        return pd.read_csv(path)
    params = dict(queryname=query, version=str(version), market_run_id=market,
                  resultformat='6', startdatetime=oasis_timestamp(start), enddatetime=oasis_timestamp(end))
    if query.startswith('PRC'):
        params['node'] = 'TH_NP15_GEN-APND'
    global LAST_REQUEST
    for attempt in range(3):
        try:
            # At most two in flight and one new request every ten seconds.
            with REQUEST_LOCK:
                time.sleep(max(0.0, 10.0 - (time.monotonic() - LAST_REQUEST)))
                LAST_REQUEST = time.monotonic()
            response = requests.get(URL, params=params, timeout=60)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                names = [n for n in archive.namelist() if n.lower().endswith('.csv')]
                if not names:
                    raise ValueError('No CSV: ' + archive.read(archive.namelist()[0]).decode()[:200])
                data = pd.concat([pd.read_csv(archive.open(n)) for n in names], ignore_index=True)
            data.to_csv(path, index=False)
            print(f'{query}: cached {start.date()} to {end.date()} ({len(data)} rows)', flush=True)
            time.sleep(5)
            return data
        except (requests.RequestException, zipfile.BadZipFile, ValueError) as exc:
            if attempt == 2:
                raise
            print(f'Retrying {query}: {exc}', flush=True)
            time.sleep(10 * (attempt + 1))


def windows(start, end):
    cursor = pd.Timestamp(start, tz='UTC')
    stop = pd.Timestamp(end, tz='UTC')
    while cursor < stop:
        nxt = min(cursor + pd.Timedelta(days=28), stop)
        yield cursor, nxt
        cursor = nxt


def main():
    forecast = []
    for start, end in windows('2024-01-01 08:00', '2026-01-01 08:00'):
        data = fetch('SLD_FCST', 1, 'DAM', start, end)
        forecast.append(data[data.TAC_AREA_NAME.isin(['CA ISO-TAC', 'PGE-TAC'])])
    f = pd.concat(forecast, ignore_index=True)
    f['datetime_utc'] = pd.to_datetime(f.INTERVALSTARTTIME_GMT, utc=True)
    # Chunk endpoints can overlap; reject conflicting duplicates rather than average them.
    keys = ['datetime_utc', 'TAC_AREA_NAME']
    if f.groupby(keys).MW.nunique().gt(1).any():
        raise ValueError('Conflicting demand forecasts at the same delivery timestamp')
    f = f.drop_duplicates(keys).pivot(index='datetime_utc', columns='TAC_AREA_NAME', values='MW')
    f.rename(columns={'CA ISO-TAC': 'load_da_system', 'PGE-TAC': 'load_da_pge'}).to_csv(CACHE/'demand_forecasts.csv')
    da, rt = [], []
    def price_window(pair):
        start, end = pair
        return fetch('PRC_LMP', 1, 'DAM', start, end), fetch('PRC_INTVL_LMP', 3, 'RTM', start, end)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for day_ahead, real_time in pool.map(price_window, windows('2025-01-01 08:00', '2026-01-01 08:00')):
            da.append(day_ahead)
            rt.append(real_time)
    d = standardize_lmp(pd.concat(da, ignore_index=True), 'da')
    r = standardize_lmp(pd.concat(rt, ignore_index=True), 'rt')
    r['hour'] = r.datetime_utc.dt.floor('h')
    counts = r.groupby('hour').size()
    counts.rename('rt_interval_count').to_csv(CACHE/'rt_interval_counts.csv')
    if not counts.eq(12).all():
        raise ValueError('Expected 12 distinct five-minute RT intervals per hour; inspect counts')
    r = r.drop(columns='datetime_utc').groupby(['hour','node']).mean(numeric_only=True).reset_index().rename(columns={'hour':'datetime_utc'})
    result = d.merge(r, on=['datetime_utc','node'], validate='one_to_one')
    result = result[(result.datetime_utc >= pd.Timestamp('2025-01-01 08:00',tz='UTC')) &
                    (result.datetime_utc < pd.Timestamp('2026-01-01 08:00',tz='UTC'))].copy()
    result['spread'] = result.total_lmp_rt - result.total_lmp_da
    for short, column in [('energy','system_energy_price'),('congestion','congestion_price'),('loss','marginal_loss_price')]:
        result[short+'_spread'] = result[column+'_rt'] - result[column+'_da']
    result.to_csv(CACHE/'prices_2025.csv',index=False)
    print(f'Prepared 2025 panel: {len(result)} hours; forecast panel: {len(f)} hours',flush=True)

if __name__ == '__main__':
    main()
