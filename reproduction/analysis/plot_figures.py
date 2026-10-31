"""Redraw the six manuscript figures and the supporting figures from the distributed tables."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('data',type=Path);parser.add_argument('out',type=Path)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
g=pd.read_csv(args.data/'global_points.csv.gz')
r=pd.read_csv(args.data/'lineage_flow.csv',dtype={'model':str})
local=pd.read_csv(args.data/'local_points.csv')
palette={'A':'#0072B2','P_L':'#2C8EBA','P_G':'#6F4C9B','J_L':'#B35E95','J_G':'#00876C','J_NoMP':'#A97818','B1':'#202A33','B2':'#58636C','N':'#7A838B','L':'#8C6D4F'}
mk=dict(zip(palette,['s','^','v','D','o','P','h','p','+','<']))
label={'A':'A','P_L':r'$P_\mathrm{L}$','P_G':r'$P_\mathrm{G}$','J_L':r'$J_\mathrm{L}$','J_G':r'$J_\mathrm{G}$','J_NoMP':r'$J_\mathrm{NoMP}$','B1':'B1','B2':'B2','N':'N','L':'L'}
seeds=['20260914','20260915','20260916','2026102804','2026102805']
dash=['solid',(0,(3,2)),(0,(1,1.5)),(0,(5,2,1,2)),(0,(7,2))]
pale='#CCD4D9'; ink='#1D2E39'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#829099','axes.labelcolor':ink,
    'xtick.color':ink,'ytick.color':ink,'text.color':ink,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.titlelocation':'left','axes.titlepad':6,'lines.solid_capstyle':'round','axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6})
def panel(ax,letter,title): ax.set_title('('+letter+')  '+title,loc='left',weight='bold')
def setup(ax):
    ax.grid(axis='y',color='#DDE3E6',linewidth=.6);ax.set_axisbelow(True);ax.tick_params(length=3)
def dc(ax):
    ax.set_xlim(.025,.925);ax.set_xticks([.1,.3,.5,.7,.9]);ax.set_xlabel(r'Road density $\rho$')
def line(a,z,m,ci=False):
    x=z.d/100;y=z.value; c=palette[m]
    if ci:a.fill_between(x,z.low,z.high,color=c,alpha=.12,linewidth=0)
    a.plot(x,y,color=c,marker=mk[m],lw=1.55,ms=3.5,markevery=2,label=label[m])
def save(f,name):
    tmp=args.out/(name+'.partial.pdf');f.savefig(tmp,bbox_inches='tight',pad_inches=.035,metadata={'Creator':'Matplotlib','Title':name});plt.close(f);assert tmp.stat().st_size>1000;tmp.replace(args.out/(name+'.pdf'))
def series(v,m,f=None):
    a=g[(g.view==v)&(g.series==m)];return a[a.f==f].sort_values('d') if f is not None else a
def heat(ax,v,m,lim,letter,title):
    z=series(v,m).pivot(index='f',columns='d',values='value')
    signs=series(v,m).pivot(index='f',columns='d',values='sign')
    im=ax.imshow(z.to_numpy(),origin='lower',aspect='auto',cmap='PuOr_r',vmin=-lim,vmax=lim,interpolation='none')
    y,x=np.where(signs.to_numpy()==0)
    ax.scatter(x,y,s=6,facecolors='none',edgecolors='#444',lw=.55)
    ax.set_xticks([0,3,6,9,12,15,17],['.05','.20','.35','.50','.65','.80','.90'])
    ax.set_yticks(range(len(z)),z.index);ax.set_xlabel(r'Road density $\rho$');ax.set_ylabel('AVs (%)');panel(ax,letter,title)
    return im
# Figure 1: absolute speeds, three policies per panel, identical axes.
f,ax=plt.subplots(1,3,figsize=(7.05,2.75),sharey=True)
for a,x,k in zip(ax,[10,50,95],'abc'):
    for m in ['B2','A','J_G']:line(a,series('G01_speed',m,x),m,True)
    setup(a);dc(a);a.set_ylim(-.12,6.15);panel(a,k,str(x)+'% AVs')
ax[0].set_ylabel('Mean PV speed (cells/step)');h,l=ax[0].get_legend_handles_labels();f.legend(h,l,ncol=3,loc='upper center',bbox_to_anchor=(.52,1.03));f.subplots_adjust(left=.095,right=.993,top=.79,bottom=.20,wspace=.13);save(f,'fig1_speed')
# Figure 2: grid of direct paired comparisons (separate clearly marked range for each).
f,ax=plt.subplots(2,1,figsize=(6.35,4.2))
for a,v,m,k,t in [(ax[0],'G02_role_contrasts','J_G-A','a',r'Joint graph minus coordinator only ($J_\mathrm{G}-A$)'),(ax[1],'G02_vs_B2','J_G-B2','b',r'Joint graph minus reactive yielding ($J_\mathrm{G}-\mathrm{B2}$)')]:
    z=series(v,m);lim=np.ceil(np.abs(z.value).max()*10)/10;im=heat(a,v,m,lim,k,t)
    cb=f.colorbar(im,ax=a,pad=.025,fraction=.055,aspect=13);cb.set_ticks([-lim,0,lim]);cb.set_label(r'$\Delta v$ (cells/step)')
f.subplots_adjust(left=.11,right=.98,top=.93,bottom=.10,hspace=.68);save(f,'fig2_contrasts')
# Figure 3: separate method panels, lineages drawn even when underneath the family mean.
f,ax=plt.subplots(1,3,figsize=(7.2,2.95),sharey=True)
for a,m,k in zip(ax,['A','J_L','J_G'],'abc'):
    for i,s in enumerate(seeds):
        z=r[(r.method==m)&(r.model==s)&(r.f==100)].sort_values('d')
        if z.empty:continue
        a.plot(z.d/100,z.R,color=palette[m],ls=dash[i],alpha=.75,lw=.95,marker=mk[m],markerfacecolor='white',markevery=(i%3,4),ms=2.6)
        bad=z[z.R<.95];a.scatter(bad.d/100,bad.R,marker='x',s=26,c='#B82033',lw=1.35,zorder=5)
    z=series('G03_R_bg',m,100);a.plot(z.d/100,z.value,color=palette[m],lw=1.9,marker=mk[m],markevery=3,ms=3.3)
    a.axhline(.95,color='#242B30',ls=(0,(4,2)),lw=.9);a.set_ylim(.908,1.045);dc(a);setup(a);panel(a,k,label[m])
ax[0].set_ylabel(r'Flow ratio to B1, $R$')
hs=[Line2D([],[],ls=dash[i],c='#53646D',lw=1,label=s[-4:] if i>=3 else s[-2:]) for i,s in enumerate(seeds)]
hs += [Line2D([],[],c=ink,lw=2,marker='o',ms=3.4,label='family mean'),Line2D([],[],c='#B82033',marker='x',ls='',label='below 0.95')]
f.legend(handles=hs,loc='upper center',bbox_to_anchor=(.52,1),ncol=7,columnspacing=.90,handlelength=1.7,fontsize=7.4);f.subplots_adjust(left=.096,right=.99,top=.76,bottom=.185,wspace=.14);save(f,'fig3_flow')
# Figure 4: two matched encoder families only plus separate contrasts for the two initializations.
f,ax=plt.subplots(1,2,figsize=(7.25,2.80),gridspec_kw={'width_ratios':[1.37,1]})
im=heat(ax[0],'G07_PV','J_G-J_NoMP',.22,'a','Two matched training seeds')
cb=f.colorbar(im,ax=ax[0],orientation='horizontal',pad=.35,fraction=.09,aspect=22);cb.set_ticks([-.2,0,.2]);cb.set_label(r'$\Delta v$ (cells/step)',labelpad=1)
a=ax[1];a.axhline(0,c='#69777F',lw=.8)
for i,s in enumerate(seeds[-2:]):
    z=series('G07_PV','J_G-J_NoMP:'+s,95);a.plot(z.d/100,z.value,c=palette['J_G'],marker=['o','s'][i],fillstyle='none',ms=3.5,lw=1.2,ls=['-',(0,(3,2))][i],label='Seed '+s[-4:])
z=series('G07_PV','J_G-J_NoMP',95);a.fill_between(z.d/100,z.low,z.high,color=palette['J_G'],alpha=.13,lw=0);a.plot(z.d/100,z.value,c=palette['J_G'],lw=2,label='Mean');panel(a,'b','95% AVs');dc(a);setup(a);a.set_ylabel(r'$\Delta v$ (cells/step)');a.legend(loc='lower right',fontsize=7,ncol=1,handlelength=1.5)
f.subplots_adjust(left=.084,right=.995,top=.88,bottom=.23,wspace=.45);save(f,'fig4_encoder')
# Figure 5: diagnostic under unchanged coordinator weights.
f,axes=plt.subplots(1,2,figsize=(7.25,2.85));a=axes[0];a.axhline(0,color='#63717A',lw=.85)
for i,s in enumerate(seeds[:3]):
    z=series('G08_PV_JL_vs_D2_L','J_L-D2_L:'+s,100);a.scatter(z.d/100+(i-1)*.005,z.value,s=22,marker=['o','s','^'][i],facecolors='none',edgecolors=palette['J_L'],lw=.8,label='Seed '+s[-2:])
z=series('G08_PV_JL_vs_D2_L','J_L-D2_L',100);a.errorbar(z.d/100,z.value,[z.value-z.low,z.high-z.value],capsize=3,fmt='D',c=palette['J_L'],ms=5,label='Mean and 95% interval')
a.set_xlabel(r'Road density $\rho$');a.set_ylabel('Difference (cells/step)');a.set_xticks(z.d/100);a.tick_params(axis='x',labelsize=7);setup(a);panel(a,'a','PV speed')
h,l=a.get_legend_handles_labels()
a=axes[1];d=pd.read_csv(args.data/'JL_substitution.csv');z=d[d.metric=='delta_p'].sort_values('d');a.axhline(0,color='#63717A',lw=.85)
a.errorbar(z.d/100,z.value*100,[(z.value-z.low)*100,(z.high-z.value)*100],capsize=3,fmt='D',c=palette['J_L'],ms=5)
a.set_xlabel(r'Road density $\rho$');a.set_ylabel('Difference (percentage points)');a.set_xticks(z.d/100);a.tick_params(axis='x',labelsize=7);setup(a);panel(a,'b','Time stopped')
f.legend(h,l,loc='upper center',bbox_to_anchor=(.53,1.01),ncol=4,fontsize=7.6);f.subplots_adjust(left=.10,right=.99,top=.77,bottom=.20,wspace=.38);save(f,'fig5_substitution')
# Figure 6: local restart dynamics and count still contributing to the curve.
f,ax=plt.subplots(2,2,figsize=(6.85,4.2));lm=local[(local.view=='L09_curves_01')&(local.method=='J_G')]
ll=pd.read_csv(args.data/'local_lineages.csv.gz');ll=ll[(ll.method=='J_G')&(ll.view=='L09_curves_01')]
spec=[('time_all_followup_0','time_moving_survivor_0','value','PV speed','Cells/step'),
('time_all_followup_6',None,'value','Local flow','Vehicles/(step·lane)'),
('time_all_followup_5',None,'value','Local density','Vehicles/site'),
('time_all_followup_0','time_moving_survivor_0','support_mean','Contributing restarts','Events per road')]
for a,(p,q,col,t,u),k in zip(ax.flat,spec,'abcd'):
    if col=='value':
        bound=ll[ll.panel==p].groupby('x').value.agg(['min','max']);a.fill_between(bound.index,bound['min'],bound['max'],color=palette['J_G'],alpha=.16,lw=0)
    for j,ss in enumerate([p,q]):
        if ss is None:continue
        z=lm[lm.panel==ss].sort_values('x');a.plot(z.x,z[col],color=palette['J_G'],ls=['solid',(0,(4,2))][j],lw=[1.8,1.1][j],marker=['o','o'][j],mfc=['#00876C','white'][j],ms=3.2,markevery=(j*3,7),label=['All observed','Still moving'][j])
    a.set_ylabel(u);a.set_xlim(0,40);a.set_xticks([0,10,20,30,40]);setup(a);panel(a,k,t)
    if k in 'cd': a.set_xlabel('Steps since restart')
    if k=='a':a.legend(fontsize=7.2,loc='lower right')
f.subplots_adjust(left=.12,right=.985,top=.93,bottom=.12,wspace=.43,hspace=.52);save(f,'fig6_restart')
# Supplement grid. Separate sheets for fixed cohorts and metrics avoid crowding.
for view,axis,stem in [('G01_speed','PV speed (cells/step)','s1_speed'),('G04_stops','Fraction of time stopped','s2_stops'),('G05_completion','Probability of a completed lap','s3_completion'),('G03_q_bg','Background flow (vehicles/(step·lane))','s4_bgflow')]:
    f,ax=plt.subplots(2,4,figsize=(8.2,4.7),sharey=True)
    for a,fr,k in zip(ax.flat,[10,25,50,75,90,95,100],'abcdefg'):
        for m in ['B2','A','J_G','J_L']:line(a,series(view,m,fr),m,False)
        panel(a,k,str(fr)+'% AVs');dc(a);setup(a)
    ax.flat[-1].axis('off');h,l=ax.flat[0].get_legend_handles_labels();ax.flat[-1].legend(h,l,loc='center left');f.supylabel(axis,x=.012,fontsize=9);f.subplots_adjust(left=.075,right=.99,top=.93,bottom=.11,wspace=.17,hspace=.48);save(f,stem)
# Supplementary Figure S5, raw iteration summaries. Missing values break lines.
tr=pd.read_csv(args.data/'training.csv.gz',dtype={'seed':str})
f,ax=plt.subplots(5,2,figsize=(7.5,7.5),sharex=True,sharey='col')
for row,m in enumerate(['A','P_L','P_G','J_L','J_G']):
    for col,metric in enumerate(['pv_speed_mean','team_return_mean']):
        a=ax[row,col]
        for j,s in enumerate(seeds):
            z=tr[(tr.family==m)&(tr.seed==s)].sort_values('iteration')
            if z.empty:continue
            for seg,zz in z.groupby('segment'):
                a.plot(zz.env_steps/1e6,zz[metric],c=palette[m],alpha=.78,lw=.7,ls=dash[j])
            sel=z[z.selected_checkpoint & z[metric].notna()]
            a.scatter(sel.env_steps/1e6,sel[metric],c=palette[m],edgecolors='white',lw=.5,marker=mk[m],s=18,zorder=5)
        setup(a);a.set_xticks([0,2,4,6,8,10])
        if col==0:a.set_ylabel(label[m],rotation=0,labelpad=15,weight='bold')
        if row==4:a.set_xlabel('Environment steps (millions)')
panel(ax[0,0],'a','PV speed (cells/step)');panel(ax[0,1],'b','Mean team episode return')
f.subplots_adjust(left=.105,right=.99,top=.955,bottom=.077,hspace=.24,wspace=.19);save(f,'s5_training')
