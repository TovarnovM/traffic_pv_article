"""Run only in the approved numerical environment (NumPy and pandas required)."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
import jl_decomposition as jl
import verify_flow_intervals as flow
from table_verification import verify_table


class FlowIntervals(unittest.TestCase):
    def setUp(self):
        self.raw=pd.DataFrame([
            ['B1','rule',20,100,0,1.],['B1','rule',20,100,1,4.],
            ['B1','rule',20,100,3,2.],
            ['A','20260914',20,50,0,2.],['A','20260914',20,50,1,4.],
            ['A','20260914',20,50,2,100.],
            ['D2_L','20260914',20,100,0,100.]],
            columns=['method','model','d','f','road','flow_background'])

    def test_ratios_common_roads_and_exact_seed(self):
        with mock.patch.object(np.random,'SeedSequence',wraps=np.random.SeedSequence) as seed:
            result=flow.rebuild(self.raw)
        self.assertEqual(seed.call_args_list,[
            mock.call([910019,20,50,20260914,ord('A')]),
            mock.call([910019,20,100,0,sum(map(ord,'B1'))])])
        a=result[result.method=='A'].iloc[0]
        self.assertEqual(a.pairs,2)
        self.assertEqual(a.R,1.5)
        self.assertNotEqual(a.R,(2+4)/(1+4))
        self.assertNotIn('D2_L',set(result.method))
        rng=np.random.default_rng(np.random.SeedSequence([910019,20,50,20260914,ord('A')]))
        samples=np.array([2.,1.])[rng.integers(2,size=(10000,2))].mean(axis=1)
        np.testing.assert_array_equal([a.low,a.high],np.quantile(samples,[.025,.975],method='linear'))
        pd.testing.assert_frame_equal(result,flow.rebuild(self.raw.iloc[::-1]))
        b=result[result.method=='B1'].iloc[0]
        self.assertEqual((b.R,b.low,b.high,b.pairs),(1.,1.,1.,3))

    def test_invalid_episodes(self):
        bad=[pd.concat([self.raw,self.raw.iloc[:1]],ignore_index=True),
             self.raw[self.raw.method!='B1']]
        for value in (np.nan,np.inf,0.,-1.):
            raw=self.raw.copy();raw.loc[0,'flow_background']=value;bad.append(raw)
        raw=self.raw.copy();raw.loc[3,'model']=None;bad.append(raw)
        for raw in bad:
            with self.subTest(), self.assertRaises(ValueError):flow.rebuild(raw)

    def test_keyed_comparison_rejects_missing_duplicate_and_nonfinite(self):
        table=flow.rebuild(self.raw)
        def check(other):
            verify_table(table,other,['method','model','d','f'],['pairs'],['R','low','high'],2)
        check(table.iloc[::-1])
        bad=[table.iloc[:1],pd.concat([table.iloc[:1]]*2,ignore_index=True)]
        for column,value in [('model','other'),('pairs',99),('R',np.nan),('low',np.inf),('high',3.),('R',1.500001)]:
            other=table.copy();other.loc[0,column]=value;bad.append(other)
        for other in bad:
            with self.subTest(),self.assertRaises(ValueError):check(other)


class JLDecomposition(unittest.TestCase):
    def test_fixed_model_and_common_road_counts_required(self):
        rows=pd.DataFrame([
            [method,model,20,100,road,2.,.1]
            for method in ['J_L','D2_L'] for model in ['1','2','3'] for road in range(22)],
            columns=['method','model','d','f','road','mean_speed_priority','stopped_fraction_priority'])
        for bad in (rows[rows.model!='3'],rows[rows.road!=21]):
            with self.subTest(),self.assertRaises(AssertionError):jl.rebuild(bad)

    def test_cli_verifies_all_keys_and_writes_only_new_file(self):
        rows=pd.DataFrame([
            [method,model,d,100,road,2.+role*.1+road*.01,.1+role*.02]
            for d in [20,30,40,50,55,60] for role,method in enumerate(['J_L','D2_L'])
            for model in ['20260914','20260915','20260916'] for road in range(22)],
            columns=['method','model','d','f','road','mean_speed_priority','stopped_fraction_priority'])
        table=jl.rebuild(rows)
        self.assertEqual(len(table),60)
        self.assertTrue((table.roads==22).all())
        self.assertTrue((table.models==3).all())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'episodes.csv.gz';reference=root/'JL_substitution.csv'
            rows.to_csv(source,index=False);table.to_csv(reference,index=False)
            before={p:p.read_bytes() for p in (source,reference)}
            jl.main([str(root)])
            output=root/'rebuilt.csv'
            jl.main([str(root),'--output',str(output)])
            pd.testing.assert_frame_equal(pd.read_csv(output),pd.read_csv(reference))
            for target in (source,reference,output):
                old=target.read_bytes()
                with self.assertRaises(SystemExit):jl.main([str(root),'--output',str(target)])
                self.assertEqual(target.read_bytes(),old)
            for p,content in before.items():self.assertEqual(p.read_bytes(),content)
            for bad in (table.iloc[:-1],table.assign(metric='missing')):
                bad.to_csv(reference,index=False)
                with self.assertRaises(ValueError):jl.main([str(root)])


if __name__=='__main__':
    unittest.main()
