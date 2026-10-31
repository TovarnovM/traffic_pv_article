"""Independent paper-level checks from the distributed data. Uses the standard library."""
import csv,gzip,math,statistics,collections,sys,re
from pathlib import Path
root=Path(sys.argv[1])
def read(name):
    with (gzip.open(root/name,'rt') if name.endswith('.gz') else open(root/name)) as f:return list(csv.DictReader(f))
def close(x,y,d=5e-5):assert abs(float(x)-y)<=d,(x,y)
episodes=read('episodes.csv.gz');g=read('global_points.csv.gz');rr=read('lineage_flow.csv')
assert len(episodes)==116856 and sum(r['observer']=='True' for r in episodes)==41760
assert len({tuple(r[k] for k in ('d','f','model','method')) for r in episodes})==3060
# Check all main-table speeds from raw roads, weighting every trained model and cell equally.
cell=collections.defaultdict(list)
for row in episodes:
 if row['method']!='D2_L':cell[(row['method'],row['model'],int(row['d']),int(row['f']))].append(float(row['mean_speed_priority']))
group=collections.defaultdict(list)
for (m,s,d,f),v in cell.items():group[m,'high' if f>=90 else 'low'].append(statistics.mean(v))
expect={'B1':(1.870,1.870),'B2':(2.004,2.385),'N':(1.875,1.881),'L':(1.922,2.570),'A':(1.980,2.684),'P_L':(1.851,1.853),'P_G':(1.845,1.859),'J_L':(1.888,2.472),'J_G':(1.958,2.783),'J_NoMP':(1.951,2.808)}
for m,vals in expect.items():
 for h,v in zip(('low','high'),vals):close(statistics.mean(group[m,h] if m!='B1' else group[m,'high']),v,.00050)
# Original (3-model) effect size and new (5-model) effect size, same 54 cells.
def original(m):
 return statistics.mean(statistics.mean(v) for (mm,s,d,f),v in cell.items() if mm==m and f>=90 and s in ('20260914','20260915','20260916'))
old=(original('J_G')/original('A')-1)*100;new=(statistics.mean(group['J_G','high'])/statistics.mean(group['A','high'])-1)*100
close(old,2.4,.06)
print('Old-three high-AV speeds:',original('J_G'),original('A'),'relative difference (%)',old,'five-lineage difference (%)',new)
expected={('G02_role_contrasts','J_G-A'):((8,6,58),(44,10,0)),('G02_role_contrasts','J_L-A'):((7,8,57),(12,10,32)),('G02_vs_B2','J_G-B2'):((4,3,65),(49,5,0)),('G02_vs_B2','A-B2'):((4,8,60),(45,6,3)),('G02_role_contrasts','J_G-P_G'):((72,0,0),(54,0,0)),('G07_PV','J_G-J_NoMP'):((24,34,14),(1,33,20))}
for (v,m),hh in expected.items():
 for upper,target in zip((False,True),hh):
  z=[x for x in g if x['view']==v and x['series']==m and (int(x['f'])>=90)==upper];signs=collections.Counter((1 if float(x['low'])>1e-12 else -1 if float(x['high'])< -1e-12 else 0) for x in z)
  assert tuple(signs[i] for i in (1,0,-1))==target
# Rebuild every failed screen and each joint win, using B1 matched by registered road key.
baseline={(int(x['d']),int(x['road'])):float(x['flow_background']) for x in episodes if x['method']=='B1'}
pairs=collections.defaultdict(list)
for x in episodes:
 m=x['method']; d,f,j=int(x['d']),int(x['f']),int(x['road'])
 if m=='D2_L' or (d,j) not in baseline:continue
 pairs[m,x['model'],d,f].append(float(x['flow_background'])/baseline[d,j])
assert len(pairs)==len(rr)
failed=[]
for x in rr:
 k=x['method'],x['model'],int(x['d']),int(x['f'])
 pv=statistics.mean(pairs[k]);close(pv,float(x['R']),1e-12);assert int(x['pairs'])==len(pairs[k])
 if pv<.95:failed.append(k);assert float(x['low'])<.95<float(x['high'])
assert set(failed)=={('A','2026102804',25,100),('J_L','20260916',55,100),('J_L','20260916',60,100),('J_L','20260916',65,100),('L','rule',20,100),('L','rule',25,95),('L','rule',25,100)}
for m,expected_wins in [('A',(4,44)),('J_G',(4,49))]:
 for high,target in zip((False,True),expected_wins):
  z=[x for x in g if x['view']=='G02_vs_B2' and x['series']==m+'-B2' and (int(x['f'])>=90)==high]; wins=sum(float(x['low'])>0 and not any(k[0]==m and k[2]==int(x['d']) and k[3]==int(x['f']) for k in failed) for x in z);assert wins==target
# Native/learned PV paired with identical coordinator lineage and the same 22 roads.
dd=read('JL_substitution.csv');frozen=[x for x in g if x['view']=='G08_PV_JL_vs_D2_L' and x['series']=='J_L-D2_L']
for z in frozen:
 d=int(z['d']);out={x['metric']:float(x['value']) for x in dd if int(x['d'])==d}
 close(out['stop_component']+out['moving_component'],out['delta_v'],1e-12);close(out['delta_v'],float(z['value']),1e-12)
# Controls, timing, attrition, and completion claim.
local=read('local_points.csv')
def loc(panel,x,metric='value',m='J_G'):
 return float(next(r[metric] for r in local if r['method']==m and r['panel']==panel and int(r['x'])==x))
for panel,values in [('time_all_followup_0',(1,2.3544,1.9364)),('time_all_followup_5',(.7701,.4592,.5154)),('time_all_followup_6',(.1919,.4939,.4466))]:
 for j,v in zip((0,20,40),values):close(loc(panel,j),v)
close(loc('time_moving_survivor_0',40,'support_mean'),6.68,.005);close(loc('time_all_followup_0',40,'support_mean'),36.59,.005)
close(loc('controls_1',0),.3786);close(loc('controls_1',1),.6047);close(loc('controls_1',5),.6049)
for m,expected_c in [('J_L',(31,90)),('A',(174,174)),('J_G',(174,174))]:
 z=[x for x in episodes if x['method']==m and x['d']=='70' and x['f']=='100']
 assert (sum(float(x['completed_priority_laps'])>0 for x in z),len(z))==expected_c
assert all(int(x['completed_priority_laps'])==0 for x in episodes if x['d']=='90')
print('All table means, two-cohort comparisons, point classification, paired flow ratios, seven failures, combined wins, decomposition, local examples and completion checks passed.')
