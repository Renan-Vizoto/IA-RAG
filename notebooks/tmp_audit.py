from pathlib import Path
import pandas as pd, numpy as np, re

base = Path('notebooks')
data_dir = base / 'data'
files = sorted(data_dir.glob('*electr*'))
print('files', len(files))
use_cols = [
    'net_manager','purchase_area','city','num_connections','delivery_perc',
    'perc_of_active_connections','type_of_connection','type_conn_perc',
    'annual_consume','annual_consume_lowtarif_perc','smartmeter_perc','consume_per_conn'
]
num_cols = ['num_connections','delivery_perc','perc_of_active_connections','type_conn_perc','annual_consume','annual_consume_lowtarif_perc','smartmeter_perc','consume_per_conn']
frames=[]
for fp in files:
    header = pd.read_csv(fp, nrows=0).columns.tolist()
    cols = [c for c in use_cols if c in header]
    df = pd.read_csv(fp, usecols=cols, dtype=str, low_memory=False)
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['net_manager','purchase_area','city','type_of_connection']:
        if c in df.columns:
            df[c] = df[c].astype('category')
    m = re.search(r'(20\d{2})', fp.stem)
    if m:
        df['year'] = int(m.group(1))
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
print('shape', raw.shape)
print('year range', raw['year'].min(), raw['year'].max(), 'n_years', raw['year'].nunique())
print('\nrows per year:')
print(raw['year'].value_counts().sort_index().to_string())
print('\nmissing %:')
print((raw.isna().mean().sort_values(ascending=False)*100).round(2).to_string())
print('\nunique counts:')
for c in ['net_manager','purchase_area','city','type_of_connection']:
    print(c, raw[c].nunique(dropna=True))
print('\nannual_consume describe raw:')
print(raw['annual_consume'].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99,.995]).to_string())
clean = raw[raw['annual_consume'].notna() & (raw['annual_consume'] > 0)].copy()
q_high = clean['annual_consume'].quantile(0.995)
clean = clean[clean['annual_consume'] <= q_high].copy()
clean['consume_per_conn_calc'] = clean['annual_consume'] / clean['num_connections'].replace(0, np.nan)
q_cpc = clean['consume_per_conn_calc'].quantile(0.995)
clean_cpc = clean[clean['consume_per_conn_calc'] <= q_cpc].copy()
print('\nafter annual_consume filter rows', len(clean), 'q_high', round(q_high,2))
print('after consume_per_conn filter rows', len(clean_cpc), 'q_cpc', round(q_cpc,2))
print('\nconsume_per_conn describe after filters:')
print(clean_cpc['consume_per_conn_calc'].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99,.995]).to_string())
print('\nshare duplicated by manager/city/year/num_connections/annual_consume:', clean_cpc.duplicated(subset=['net_manager','city','year','num_connections','annual_consume']).mean().round(6))
city_year = clean_cpc.groupby(['city','year']).size().reset_index(name='n')
city_counts = city_year.groupby('city')['year'].nunique()
print('cities appearing in >1 year:', int((city_counts>1).sum()), 'of', int(city_counts.shape[0]))
print('median years per city:', float(city_counts.median()))
print('p95 years per city:', float(city_counts.quantile(0.95)))
if 'consume_per_conn' in raw.columns:
    calc = raw['annual_consume'] / raw['num_connections'].replace(0,np.nan)
    diff = (raw['consume_per_conn'] - calc).abs()
    print('\nprovided consume_per_conn non-null ratio', round(raw['consume_per_conn'].notna().mean(),4))
    print('provided consume_per_conn matches computed ratio on non-null rows:', float((diff.dropna() < 1e-3).mean()))
