"""Regression tests for the full-training CASE metadata checkpoint fix."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pilot import run_frozen_validation as frozen


class FrozenCheckpointTests(unittest.TestCase):
    def test_case_selection_does_not_collide_with_experiment_selection(self):
        inputs={'selection':'ridge 0.001 primary', 'context':'cohort'}
        mapped=frozen.case_settings_for_core(inputs)
        self.assertNotIn('selection',mapped)
        self.assertEqual(mapped['experiment_selection'],inputs['selection'])
        self.assertEqual(inputs['selection'],'ridge 0.001 primary')

    def test_exact_complete_bic_path_is_recovered(self):
        with tempfile.TemporaryDirectory() as work:
            path=Path(work)/'case_bic_path.json'
            settings={'selection':'ridge 0.001 primary','context':'none'}
            self.assertIsNone(frozen.recover_complete_case_path(path,settings,3))
            rows=[dict(lambda_l1=lam, coefficients=[1.,2.,3.],
                       bic_proxy=(12 if lam==0.002 else 20+lam),
                       asymmetric_loss_per_case=42.,
                       bic_formula='2*n*log(mean asymmetric loss)+df*log(n)')
                  for lam in frozen.LAMBDA_GRID]
            path.write_text(json.dumps(rows))
            recovered=frozen.recover_complete_case_path(path,settings,3)
            self.assertEqual(recovered['lambda_l1'],0.002)
            self.assertEqual(recovered['experiment_selection'],'ridge 0.001 primary')
            self.assertIn('quasi-BIC',recovered['selection'])
            path.write_text(json.dumps(rows[:-1]))
            self.assertIsNone(frozen.recover_complete_case_path(path,settings,3))

    def test_original_runner_with_trailing_newline_can_resume(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work)
            args=SimpleNamespace(root=root)
            new={'experiment':'frozen','runner_sha256':'NEW_SHA',
                 'source_sha256':{'pilot/run_pilot.py':'UNCHANGED'},
                 'selection':'ridge 0.001 primary'}
            recorded=dict(new,runner_sha256=
                frozen.PRE_FIX_RUNNER_SHA256_TRAILING_NEWLINE)
            (root/'full_training_settings.json').write_text(json.dumps(recorded))
            with patch.object(frozen,'initial_identity',return_value=new):
                frozen.assert_identity(args)
            self.assertEqual(json.loads(
                (root/'full_training_settings.json').read_text()),new)
            migration=json.loads((root/'metadata_fix_migration.json').read_text())
            self.assertEqual(migration['previous_runner_sha256'],
                             frozen.PRE_FIX_RUNNER_SHA256_TRAILING_NEWLINE)

    def test_full_identity_round_trips_through_json(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work)
            artifacts=root/'inputs'
            artifacts.mkdir()
            (artifacts/'train.npz').write_bytes(b'fake-hashed-matrix')
            (root/'workbook.xlsx').write_bytes(b'fake-hashed-workbook')
            meta={'workbook_sha256':'SAME',
                  'matrix_sha256':{'train':'SAME'},
                  'train_cases':18705,'validation_cases':5779,
                  'training_keys':[['SITE',f'train{i}'] for i in range(180)],
                  'validation_keys':[['SITE',f'valid{i}'] for i in range(54)]}
            (artifacts/'manifest.json').write_text(json.dumps(meta))
            args=SimpleNamespace(artifacts=artifacts,workbook=root/'workbook.xlsx',
                                 workers=12)
            with patch.object(frozen,'sha',return_value='SAME'):
                identity=frozen.initial_identity(args)
            self.assertEqual(identity['context'],['none','cohort'])
            self.assertEqual(identity,json.loads(json.dumps(identity)))

    def test_precise_identity_migration_only_before_frozen_policies(self):
        with tempfile.TemporaryDirectory() as work:
            root=Path(work)
            args=SimpleNamespace(root=root)
            new={'experiment':'frozen','runner_sha256':'NEW_SHA',
                 'source_sha256':{'pilot/run_pilot.py':'UNCHANGED'},
                 'selection':'ridge 0.001 primary'}
            recorded=dict(new,runner_sha256=frozen.PRE_FIX_RUNNER_SHA256)
            (root/'full_training_settings.json').write_text(json.dumps(recorded))
            with patch.object(frozen,'initial_identity',return_value=new):
                frozen.assert_identity(args)
            self.assertEqual(json.loads(
                (root/'full_training_settings.json').read_text()),new)
            self.assertTrue((root/'metadata_fix_migration.json').is_file())
            # Exact same old code hash must NOT be migrated after freeze exists.
            (root/'full_training_settings.json').write_text(json.dumps(recorded))
            (root/'frozen_policies.json').write_text('{}')
            with patch.object(frozen,'initial_identity',return_value=new):
                with self.assertRaisesRegex(RuntimeError,'refusing to overwrite'):
                    frozen.assert_identity(args)


if __name__=='__main__':
    unittest.main()
