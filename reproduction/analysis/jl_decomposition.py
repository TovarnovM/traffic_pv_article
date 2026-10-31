"""Speed decomposition on identical roads with frozen AV coordinator weights. No new runs."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from table_verification import verify_table

def summarize(x):
    # Last two dimensions index role and metric; leading dimensions, if any, are resamples.
    v=x[...,0];p=x[...,1];u=v/(1-p)
    stop=(p[...,1]-p[...,0])*(u[...,0]+u[...,1])/2
    moving=(u[...,0]-u[...,1])*(1-(p[...,0]+p[...,1])/2)
    return np.stack([v[...,0],v[...,1],p[...,0],p[...,1],u[...,0],u[...,1],
        v[...,0]-v[...,1],p[...,0]-p[...,1],stop,moving],axis=-1)
def rebuild(rows):
    d=rows[(rows.method.isin(['J_L','D2_L']))&(rows.f==100)]
    out=[]
    for density in sorted(d.loc[d.method=='D2_L','d'].unique()):
        z=d[d.d==density]
        models=sorted(z.model.unique());assert len(models)==3
        cols=[z[(z.model==s)&(z.method==m)].set_index('road')[['mean_speed_priority','stopped_fraction_priority']] for m in ['J_L','D2_L'] for s in models]
        common=sorted(set.intersection(*(set(f.index) for f in cols)));assert len(common)==22
        # road × role × training × metric -> common-road means across the three fixed lineages.
        x=np.stack([f.loc[common].to_numpy() for f in cols],axis=1).reshape(len(common),2,3,2).mean(axis=2)
        theta=summarize(x.mean(axis=0))
        rng=np.random.default_rng(np.random.SeedSequence([910019,int(density),100,25]))
        ids=rng.integers(len(common),size=(10000,len(common)))
        boot=summarize(x[ids].mean(axis=1));assert np.allclose(boot[:,6],boot[:,8]+boot[:,9])
        for name,est,lo,hi in zip(['v_learned','v_native','p_learned','p_native','u_learned','u_native','delta_v','delta_p','stop_component','moving_component'],theta,*np.quantile(boot,[.025,.975],axis=0)):
            out.append([density,name,est,lo,hi,len(common),len(models)])
    t=pd.DataFrame(out,columns=['d','metric','value','low','high','roads','models'])
    return t


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path)
    parser.add_argument('--output',type=Path,help='write rebuilt table to a new file')
    args=parser.parse_args(argv)
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        parser.error('output must be a new file')
    rows=pd.read_csv(args.data/'episodes.csv.gz',dtype={'model':str})
    t=rebuild(rows)
    reference=pd.read_csv(args.data/'JL_substitution.csv')
    verify_table(t,reference,['d','metric'],['roads','models'],['value','low','high'],60)
    if args.output is not None:
        with args.output.open('x') as output:
            t.to_csv(output,index=False)
    print('60 JL decomposition estimates and both endpoints reproduced to absolute tolerance 1e-10.')


if __name__=='__main__':
    main()
