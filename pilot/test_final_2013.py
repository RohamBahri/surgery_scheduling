"""Safety and statistics tests; no historical test labels are loaded."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pilot import run_final_2013 as f


class Final2013Tests(unittest.TestCase):
    def test_final_policy_grid_and_methods_are_fixed(self):
        self.assertEqual(f.METHODS[0],'BOOKED')
        self.assertEqual(f.METHODS[3],'ANCHORED_0p001')
        self.assertEqual(f.RIDGES,(.001,.003))
        self.assertEqual(f.SEEDS,(0,1,2))
        self.assertEqual(f.BUDGET['anchored_inner'],20)
        self.assertGreater(f.BUDGET['case_inner'],500)

    def test_clean_separate_final_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            loc=f.method_paths(root)
            self.assertEqual(len(loc),6)
            self.assertTrue(all('validation' not in str(p) for p in loc.values()))
            self.assertIn('cohort/sensitivity/ridge_0p001',str(loc['ANCHORED_0p001']))

    def test_training_only_operational_flag_needs_explicit_ack(self):
        with tempfile.TemporaryDirectory() as td:
            args=SimpleNamespace(root=Path(td),acknowledge_audit=False)
            manifest={'policy_sha256':{},'acknowledged_operational_risks':['long_room_loads']}
            with patch.object(f,'frozen_payload',return_value=manifest):
                with self.assertRaisesRegex(RuntimeError,'--acknowledge-audit'):
                    f.freeze(args)
                self.assertFalse((args.root/'FINAL_FREEZE.json').exists())
                args.acknowledge_audit=True
                f.freeze(args)
                self.assertTrue((args.root/'FINAL_FREEZE.json').exists())

    def test_training_manifest_sha_check_refuses_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            files=['training_data.npz','training_weeks.pkl','feature_encoder.pkl']
            for name in files:(root/name).write_bytes(b'right')
            f.save(root/'training_manifest.json',dict(cases=24506,calendar_weeks=79,
                training_npz_sha256=f.sha(root/files[0]),
                train_weeks_sha256=f.sha(root/files[1]),
                encoder_sha256=f.sha(root/files[2])))
            args=SimpleNamespace(root=root)
            self.assertEqual(f.verify_prepared(args)['cases'],24506)
            (root/files[1]).write_bytes(b'wrong')
            with self.assertRaisesRegex(RuntimeError,'changed'):
                f.verify_prepared(args)

    def test_calendar_bootstrap_pairs_sites_and_reuses_each_seed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'test').mkdir()
            rows=[];n=0
            for week in ('2013-01-07','2013-01-14'):
                for site in ('TGH','TWH-main'):
                    n+=1
                    for seed in f.SEEDS:
                        for name in f.METHODS:
                            rows.append(dict(calendar_week=week,site=site,seed=seed,
                                method=name,cases=1,
                                regret=(8. if name=='ANCHORED_0p001' else 10.),
                                realized_cost=100.,hindsight_cost=90.))
            f.save(root/'test'/'test_scores.json',dict(test_cases=n,per_site_week=rows))
            f.summarize(SimpleNamespace(root=root))
            summary=json.loads((root/'test'/'test_summary.json').read_text())
            self.assertEqual(summary['calendar_weeks'],2)
            self.assertEqual(summary['means']['ANCHORED_0p001'],8.)
            self.assertAlmostEqual(summary['paired_versus_cohort_case']
                                   ['ANCHORED_0p001']['difference_per_case'],-2.)

    def test_code_does_not_reference_validation_cohort_or_old_18_week_split(self):
        src=Path(f.__file__).read_text()
        self.assertNotIn("read_split(args,'validation')",src)
        self.assertNotIn("validation.npz",src)
        self.assertIn("split_rows(cohort,'test')",src)
        self.assertIn("frozen_sha=validate_freeze(args)",src)


if __name__=='__main__':unittest.main()