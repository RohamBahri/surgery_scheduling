"""Fast fold-design tests; no solver and no UHN data required."""
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from pilot.run_fold_pilot import split_calendar_weeks
from pilot.run_pilot import add_context


@dataclass
class FakeWeek:
    monday: str
    case_ids: np.ndarray
    booked: np.ndarray
    k: int
    surgeon: np.ndarray
    surgeon_ids: tuple
    allowed: tuple


class FoldTests(unittest.TestCase):
    def test_disjoint_site_week_folds(self):
        first=date(2011,7,4)
        weeks=[]
        case=0
        for j in range(60):
            monday=str(first+timedelta(weeks=j))
            for site in range(3):
                case+=1
                weeks.append(FakeWeek(monday,np.array([case]),
                    np.array([200.+j]),1,np.array([0]),('A',),((0,),)))
        fit0,eval0,ids_fit0,ids_eval0,mon0=split_calendar_weeks(weeks,0)
        fit1,eval1,ids_fit1,ids_eval1,mon1=split_calendar_weeks(weeks,1)
        self.assertEqual((len(fit0),len(eval0)),(90,90))
        self.assertEqual(ids_fit0 & ids_eval0,set())
        self.assertEqual(ids_fit1 & ids_eval1,set())
        self.assertEqual(ids_fit0,ids_eval1)
        self.assertEqual(ids_fit1,ids_eval0)
        self.assertFalse(mon0 & mon1)

    def test_split_rejects_case_overlap(self):
        first=date(2011,7,4)
        weeks=[]
        k=0
        for j in range(60):
            for site in range(3):
                k+=1
                cid=1 if j in (0,1) and site==0 else k+1
                weeks.append(FakeWeek(str(first+timedelta(weeks=j)),
                    np.array([cid]),np.array([250.]),1,
                    np.array([0]),('A',),((0,),)))
        with self.assertRaisesRegex(RuntimeError,'Held-out case'):
            split_calendar_weeks(weeks,0)

    def test_context_scaler_uses_fitting_distribution(self):
        w1=FakeWeek('2011-07-04',np.array([10,11,12]),
             np.array([100.,200.,300.]),2,np.array([0,0,1]),
             ('A','B'),((0,1),(1,)))
        w2=FakeWeek('2011-07-11',np.array([13,14]),
             np.array([50.,70.]),1,np.array([0,0]),('C',),((0,),))
        base=np.ones((5,2))
        idx={cid:i for i,cid in enumerate([10,11,12,13,14])}
        x,names,mu,sd=add_context(base,[w1,w2],idx,variant='cohort')
        self.assertEqual(x.shape,(5,4))
        self.assertTrue(np.allclose(x[:,2:].mean(axis=0),0))
        evalweeks=[FakeWeek('2011-07-18',np.array([21,22]),
                    np.array([500.,100.]),1,np.array([0,0]),('D',),((0,),))]
        xe,_,mue,sde=add_context(np.ones((2,2)),evalweeks,
                    {21:0,22:1},variant='cohort',scaler=(mu,sd))
        self.assertEqual(mue,mu)
        self.assertEqual(sde,sd)
        self.assertGreater(abs(float(xe[0,2])),1.)

    def test_help_supports_case_budget(self):
        path=Path(__file__).with_name('run_fold_pilot.py')
        result=subprocess.run([sys.executable,str(path),'--help'],
                              capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--case-inner',result.stdout)
        self.assertIn('cohort',result.stdout)
        self.assertIn('--ridge-grid',result.stdout)


if __name__=='__main__':
    unittest.main()
