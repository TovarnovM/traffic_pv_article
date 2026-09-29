"""Focused offline statistical regressions; no simulation or model loading."""
import unittest
from unittest.mock import patch

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import publication as main
import publication_inputs as inputs


class MainSeriesTest(unittest.TestCase):
    def test_b1_ratio_not_last_control(self):
        rows = {}
        for m in main.METHODS:
            for s in main.SEEDS if m in main.FAMILIES else (None,):
                for run in range(30):
                    rows[m, s, 5, 90, run] = {**{k: 1.0 for k in main.METRICS},
                        'flow_background': 2.0 if m == 'B1' else 10.0 if m == 'L' else 1.0,
                        main.LAP: None}
        with patch.object(main, 'DENSITIES', (5,)), patch.object(main, 'FRACTIONS', (90,)):
            conditions, contrasts, background, laps = main.calculate(rows)
        a = next(r for r in background if r['method'] == 'A')
        self.assertEqual(a['mean'], .5)
        self.assertFalse(a['all_seeds_pass'])
        self.assertEqual(next(r for r in background if r['method'] == 'L')['mean'], 5)
        self.assertEqual(set(r['method'] for r in conditions), set(main.METHODS))
        self.assertEqual(set(r['method'] for r in contrasts), set(main.METHODS))
        self.assertIsNone(laps[0]['mean'])

    def test_shared_control_once_and_joint_roads(self):
        road = np.arange(30.)
        models = np.array([road+1, road+2, road+6])
        delta = main.road_difference(models, road[None, :])
        np.testing.assert_allclose(delta, 3)
        self.assertEqual(main.interval(delta, main.indices()), [3., 3.])
        varying = models + np.arange(30.)
        draws = main.indices()
        expected = varying[:, draws].mean(axis=(0, 2)) - road[draws].mean(axis=1)
        np.testing.assert_allclose(main.road_difference(varying, road[None, :])[draws].mean(axis=1), expected)
        np.testing.assert_allclose(main.interval(main.road_difference(varying, road[None, :]), draws), np.percentile(expected, [2.5, 97.5], method='linear'))
        self.assertEqual(draws.shape, (10000, 30))

    def test_missing_laps_and_conditional_pairs(self):
        self.assertEqual(main.lap_line([None, 12, 18])['mean'], 15)
        self.assertIsNone(main.lap_line([None, None])['mean'])
        self.assertIsNone(main.seed_stats([12, None, 18])['mean'])
        zero = main.lap_line([None, 12], [10, None])
        self.assertEqual(zero['n'], 0)
        self.assertIsNone(zero['mean'])
        one = main.lap_line([None, 12], [10, 10])
        self.assertEqual(one['mean'], 2)
        self.assertIsNone(one['ci95'])
        two = main.lap_line([None, 12, 18], [10, 10, 10])
        self.assertEqual(two['successful_run_ids'], [1, 2])
        self.assertIsNotNone(two['ci95'])

    def test_zero_and_missing_denominators(self):
        for bad in (0, None, float('nan'), -1):
            r = main.background_line([1, 1], [1, bad])
            self.assertEqual(r['valid_denominators'], 1)
            self.assertIsNone(r['mean'])
            self.assertFalse(r['screen_ge_095'])
        self.assertEqual(main.background_line([1, 3], [2, 2])['mean'], 1)

    def test_duplicate_missing_extra_rows_rejected(self):
        rows = [dict(method=m, training_seed=s, density_token=5, fraction_token=90, run_index=0)
                for m in ('A',)+main.CONTROLS for s in ((1,) if m == 'A' else (None,))]
        kwargs = dict(families=('A',), seeds=(1,), densities=(5,), fractions=(90,), runs=1)
        self.assertEqual(len(main.index_rows(rows, **kwargs)), 5)
        for invalid in (rows+rows[:1], rows[1:], rows+[dict(rows[0], run_index=1)]):
            with self.assertRaises(ValueError):
                main.index_rows(invalid, **kwargs)

    def test_styles_retain_all_nine_methods(self):
        styles = main.styles()
        self.assertEqual(set(styles), set(main.FAMILIES+main.CONTROLS))
        self.assertEqual(styles['P_L']['marker'], '*')
        self.assertEqual(styles['J_G']['marker'], 'D')
        self.assertEqual(len({str(v['linestyle']) for v in styles.values()}), 9)

    def test_interval_sign_is_not_point_estimate_sign(self):
        self.assertEqual(main.interval_sign([-.1, .3]), 0)
        self.assertEqual(main.interval_sign([.1, .3]), 1)
        self.assertEqual(main.interval_sign([-.3, -.1]), -1)
        self.assertEqual(main.interval_sign([1e-12, .3]), 0)
        self.assertEqual(main.interval_sign([-.3, -1e-12]), 0)

    def test_sample_sd_not_sem(self):
        r = main.seed_stats([1, 2, 3])
        self.assertEqual(r['mean'], 2)
        self.assertEqual(r['seed_sd'], 1)


    def test_figures_have_nine_methods_and_clear_legend(self):
        import tempfile
        from pathlib import Path
        from matplotlib.figure import Figure
        conditions, contrasts, background, laps = [], [], [], []
        for m in main.METHODS:
            identity = dict(method=m, density_token=5, fraction_token=90)
            for metric in main.METRICS:
                conditions.append(dict(identity, metric=metric, mean=1., ci95=[.9, 1.1]))
            for b in ('B1', 'B2'):
                contrasts.append(dict(identity, comparator=b, metric=main.METRICS[0], mean=0., ci95=[-.1, .1]))
            background.append(dict(identity, comparator='B1', metric='background_ratio', mean=1.))
            laps.append(dict(identity, mean=100., completion_rate=.5, seed_means={str(s): 100. for s in main.SEEDS}))
        checked = []
        def inspect_figure(fig, *args, **kwargs):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend = fig.legends[0]
            self.assertEqual(len(legend.get_texts()), 9)
            self.assertAlmostEqual(fig.get_figwidth(), 5.65)
            for ax in fig.axes:
                self.assertEqual(len(ax.get_legend_handles_labels()[0]), 9)
                if ax.get_xlabel():
                    self.assertLess(legend.get_window_extent(renderer).y1,
                                    ax.xaxis.label.get_window_extent(renderer).y0)
            checked.append(True)
        with tempfile.TemporaryDirectory() as tmp, patch.object(main, 'DENSITIES', (5,)), patch.object(main, 'FRACTIONS', (90,)), \
                patch.object(Figure, 'savefig', inspect_figure):
            # The renderer intentionally has three fraction columns; use all three in this test.
            for records in (conditions, contrasts, background, laps):
                records.extend([dict(r, fraction_token=f) for r in list(records) for f in (95, 100)])
            with patch.object(main, 'FRACTIONS', (90, 95, 100)):
                audit = main.plot_figures(conditions, contrasts, background, laps, Path(tmp))
        self.assertEqual(len(checked), 20)
        self.assertEqual(len(audit), 5)


class PublishedInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gzip
        import json
        cls.indexed, cls.receipt, cls.metadata, cls.reference = inputs.load_inputs()
        with gzip.open(inputs.REVIEW / 'data/evaluation/physical_rows.json.gz', 'rt') as stream:
            cls.rows = json.load(stream)

    def test_physical_and_logical_coverage(self):
        self.assertEqual(len(self.rows), 29700)
        self.assertEqual(len(self.indexed), 30780)
        self.assertEqual(sum(r['method'] == 'B1' for r in self.rows), 540)
        self.assertEqual(len(self.receipt['winners']), 15)
        for d in main.DENSITIES:
            for run in range(30):
                copies = [{k: v for k, v in self.indexed['B1', None, d, f, run].items()
                           if k != 'fraction_token'} for f in main.FRACTIONS]
                self.assertEqual(copies[0], copies[1])
                self.assertEqual(copies[1], copies[2])

    def test_negative_projection_contracts(self):
        changes = ({'joint_checkpoint_id': 'corrupt'}, {'physical_state_hash': 'corrupt'},
                   {'mean_speed_priority': None}, {'flow_all': 100.},
                   {main.LAP: None, 'completed_priority_laps': 1}, {'split': 'validation'})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                inputs.validate([dict(self.rows[0], **change), *self.rows[1:]], self.receipt['winners'])
        missing = dict(self.rows[0])
        del missing['measurement_rng_seed']
        for rows in ([missing, *self.rows[1:]], self.rows[1:], [self.rows[0], *self.rows[:-1]]):
            with self.assertRaises(ValueError):
                inputs.validate(rows, self.receipt['winners'])
        winners = [dict(r, candidate=dict(r['candidate'])) for r in self.receipt['winners']]
        winners[0]['candidate']['joint_checkpoint_id'] = 'corrupt'
        with self.assertRaises(ValueError):
            inputs.validate(self.rows, winners)

    def test_corrupted_compressed_input_rejected(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'provenance').mkdir()
            main.write_json(root / 'provenance/publication_reference.json', self.reference)
            for name in self.reference['input_sha256']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((inputs.REVIEW / name).read_bytes())
            with (root / 'data/evaluation/physical_rows.json.gz').open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'input hash changed'):
                inputs.load_inputs(root)

    def test_immutable_original_quantities_and_tables(self):
        import tempfile
        import replay_publication as replay
        conditions, contrasts, background, laps = main.calculate(self.indexed)
        summary = main.summarize(conditions, contrasts, background, laps)
        payloads = dict(conditions=conditions, contrasts=contrasts, background=background,
                        laps=laps, summary=summary, selection=self.receipt['winners'])
        replay.verify_payloads(payloads, self.reference)
        self.assertEqual(len([r for r in summary['family_seed_summaries']
                              if r['metric'] == main.METRICS[0] and r['scope'] == 'overall'
                              and len(r['seed_means']) == 3]), 5)
        self.assertEqual(summary['background_all_seeds_failures']['J_L'], 2)
        self.assertIn('lap_missing_physical_by_method', summary)
        for row in laps:
            if any(v is None for v in row['seed_means'].values()):
                self.assertIsNone(row['mean'])
        with tempfile.TemporaryDirectory() as tmp:
            main.tables(summary, self.receipt['winners'], Path(tmp))
            for name, digest in self.reference['table_sha256'].items():
                self.assertEqual(main.sha(Path(tmp) / name), digest)
        # A wrong CI sign is rejected even if the point estimate is unchanged.
        contrasts[0]['sign'] += 1
        with self.assertRaisesRegex(ValueError, 'metric mismatch: contrasts'):
            replay.verify_payloads(payloads, self.reference)


if __name__ == '__main__':
    unittest.main()
