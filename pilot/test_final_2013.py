"""Fast orchestration tests; no workbook or Gurobi needed."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pilot import run_final_2013 as run


class Final2013Tests(unittest.TestCase):
    def test_budget_priority_and_gap_optional(self):
        self.assertEqual(run.BUDGET['gap_outer'],1)
        self.assertEqual(run.BUDGET['gap_inner'],25)
        self.assertLess(run.BUDGET['gap_seconds'],run.BUDGET['seconds'])
        self.assertEqual(run.BUDGET['anchored_inner'],20)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            self.assertNotIn('GAP_COHORT',run.available_methods(root))
            self.assertIn('CASE_COHORT',run.method_paths(root))

    def test_runner_changes_preserve_prior_training_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args=SimpleNamespace(root=root)
            original=dict(workbook_sha256='w',alpha=.8,h=30.,gamma=2.,
                          lambdas=list(run.LAMBDAS),ridges=list(run.RIDGES),
                          training_cutoff='2013-01-01',
                          feature_rule='train leave-date-out, test transform with all pre-2013 training only',
                          budget={'gap_outer':3},source_sha256={'old':'yes'})
            run.save(root/'experiment_settings.json',original)
            updated=dict(original,budget={'gap_outer':1},source_sha256={'new':'yes'})
            with patch.object(run,'identity',return_value=updated):
                run.settings(args)
            self.assertEqual(json.loads((root/'experiment_settings.json').read_text()),original)
            updated['alpha']=.5
            with patch.object(run,'identity',return_value=updated):
                with self.assertRaisesRegex(RuntimeError,'alpha'):
                    run.settings(args)

    def test_policy_versions_receive_distinct_test_folders(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            workbook=root/'workbook.xlsx';workbook.write_bytes(b'example')
            file=root/'model.json';file.write_text('{"w":[1]}')
            args=SimpleNamespace(root=root,workbook=workbook)
            a=run.test_directory(args,{'CASE_COHORT':file})
            file.write_text('{"w":[2]}')
            b=run.test_directory(args,{'CASE_COHORT':file})
            self.assertNotEqual(a,b)

    def test_training_manifest_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            files=['training_data.npz','training_weeks.pkl','feature_encoder.pkl']
            for name in files:(root/name).write_bytes(b'correct')
            run.save(root/'training_manifest.json',dict(cases=24506,calendar_weeks=79,
                training_npz_sha256=run.sha(root/files[0]),
                train_weeks_sha256=run.sha(root/files[1]),
                encoder_sha256=run.sha(root/files[2])))
            args=SimpleNamespace(root=root)
            run.verify_prepared(args)
            (root/files[1]).write_bytes(b'altered')
            with self.assertRaisesRegex(RuntimeError,'changed'):
                run.verify_prepared(args)

    def test_report_allows_missing_gap(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);test=root/'test';test.mkdir()
            rows=[]
            methods=['BOOKED','CASE_COHORT','ANCHORED_0p001','LASSO_COHORT']
            for monday in ('2013-01-07','2013-01-14'):
                for site in ('TGH','TWH-main'):
                    for seed in run.SEEDS:
                        for method in methods:
                            rows.append(dict(calendar_week=monday,site=site,seed=seed,
                                method=method,cases=1,
                                regret=8. if method=='ANCHORED_0p001' else 10.))
            run.save(test/'test_scores.json',dict(methods=methods,test_cases=4,per_site_week=rows))
            run.summarize(SimpleNamespace(test_dir=test))
            out=json.loads((test/'test_summary.json').read_text())
            self.assertEqual(out['means']['ANCHORED_0p001'],8.)
            self.assertAlmostEqual(out['paired_versus_cohort_case']['ANCHORED_0p001']['difference_per_case'],-2.)
            self.assertAlmostEqual(out['primary_vs_lasso_cohort']['difference_per_case'],-2.)


if __name__=='__main__':unittest.main()