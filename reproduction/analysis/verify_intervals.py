"""Recompute every archived absolute PV-speed and paired PV-speed CI plotted or tabulated."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import global_stats as stats
p=Path(sys.argv[1])
raw=pd.read_csv(p/'episodes.csv.gz',dtype={'model':str})
points=pd.read_csv(p/'global_points.csv.gz')
dct={(m,d,f):a.pivot(index='model',columns='road',values='mean_speed_priority').sort_index().sort_index(axis=1)
    for (m,d,f),a in raw.groupby(['method','d','f'])}
def frame(m,d,f,seeds=None):
    z=dct[m,d,100 if m=='B1' else f];return z.loc[seeds] if seeds else z
n=0
for t in points.itertuples():
    if t.view=='G01_speed':
        z=stats.absolute(frame(t.method,t.d,t.f).values)
    elif t.view in {'G02_role_contrasts','G02_vs_B2','G07_PV','G08_PV_JL_vs_D2_L'}:
        left,right=t.series.split(':')[0].split('-')
        seeds=t.series.split(':')[1:] or (['2026102804','2026102805'] if t.view=='G07_PV' else None)
        a,b=frame(left,t.d,t.f,seeds),frame(right,t.d,t.f,seeds)
        cols=sorted(set(a.columns)|set(b.columns))
        z=stats.paired_contrast(a.reindex(columns=cols).values,b.reindex(columns=cols).values)
    else:continue
    assert np.allclose([z['mean'],*z['ci95']],[t.value,t.low,t.high],rtol=0,atol=1e-10),(t.view,t.series,t.d,t.f,z)
    n+=1
print(f'{n} fixed-cohort speed estimates and both endpoints reproduced to absolute tolerance 1e-10.')
