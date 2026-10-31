"""Recompute all main-grid lineage flow means and 95% intervals from episodes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from table_verification import verify_table


def rebuild(raw):
    keys=['method','model','d','f','road']
    if raw[keys+['flow_background']].isna().any().any() or raw.duplicated(keys).any():
        raise ValueError('episodes: missing values or duplicate keys')
    if not np.isfinite(raw[['d','f','road','flow_background']].to_numpy()).all():
        raise ValueError('episodes: nonfinite values')
    # B1 is a single canonical all-native panel, copied only for display.
    b1=raw[raw.method=='B1'][['d','road','flow_background']].drop_duplicates()
    if b1.duplicated(['d','road']).any():
        raise ValueError('conflicting canonical B1 roads')
    results=[]
    groups=raw[raw.method!='D2_L'].groupby(['method','model','d','f'],sort=True)
    for (m,s,d,f),g in groups:
        t=g.merge(b1,on=['d','road'],suffixes=('','_B1'),validate='one_to_one').sort_values('road')
        if not len(t) or t.flow_background_B1.min()<=0:
            raise ValueError(f'{(m,s,d,f)}: no matched roads or nonpositive B1 denominator')
        a=(t.flow_background/t.flow_background_B1).to_numpy()
        seed=np.random.SeedSequence([910019,int(d),int(f or 0),int(s) if s!='rule' else 0,sum(map(ord,m))])
        rng=np.random.default_rng(seed)
        means=a[rng.integers(len(a),size=(10000,len(a)))].mean(axis=1)
        lo,hi=np.quantile(means,[.025,.975],method='linear')
        results.append([m,s,d,f,len(a),a.mean(),lo,hi])
    return pd.DataFrame(results,columns=['method','model','d','f','pairs','R','low','high'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    args=parser.parse_args(argv)
    raw=pd.read_csv(args.data/'episodes.csv.gz',dtype={'model':str})
    reference=pd.read_csv(args.data/'lineage_flow.csv',dtype={'model':str})
    verify_table(rebuild(raw),reference,['method','model','d','f'],['pairs'],['R','low','high'],3042)
    print('3042 lineage flow means and both endpoints reproduced to absolute tolerance 1e-10.')


if __name__=='__main__':
    main()
