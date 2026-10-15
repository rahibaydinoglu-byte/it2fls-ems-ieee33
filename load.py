import glob, sys, pandas as pd, numpy as np
root = sys.argv[1]
fs = sorted(glob.glob(root + '/Dataset/*/*.csv'))
dfs = []
for f in fs:
    d = pd.read_csv(f, encoding='utf-8-sig'); d = d.rename(columns={d.columns[0]: 'TIME'})[['TIME','P_PV','LOAD','P_BAT','P_GRID','SOC_REAL','ESS_MAX']]
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df['TIME'] = pd.to_datetime(df['TIME'])
df = df.sort_values('TIME').reset_index(drop=True)
df.to_pickle(sys.argv[2])
print(len(fs), 'days', len(df), 'rows', df.TIME.min(), df.TIME.max())
print(df.describe().T.round(3))
print('ESS_MAX values', df.ESS_MAX.dropna().unique()[:5])
print('NaN counts', df.isna().sum().to_dict())
# per-day completeness
c = df.groupby(df.TIME.dt.date).size(); print('rows/day min/max', c.min(), c.max())
print('balance check P_PV+P_GRID-P_BAT-LOAD:', (df.P_PV + df.P_GRID - df.P_BAT - df.LOAD).describe().round(3).to_dict())
print('balance alt P_PV+P_GRID+P_BAT-LOAD:', (df.P_PV + df.P_GRID + df.P_BAT - df.LOAD).describe().round(3).to_dict())
