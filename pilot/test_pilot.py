"""No-Gurobi numerical smoke tests for the pilot's context and link."""
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('pilot_run',
                                            Path(__file__).parent/'run_pilot.py')
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

@dataclass
class Week:
    booked: np.ndarray
    k: int
    surgeon: np.ndarray
    surgeon_ids: tuple
    allowed: tuple
    case_ids: np.ndarray

class PilotTests(unittest.TestCase):
    def test_context(self):
        a=Week(np.array([100.,200.,300.]),2,np.array([0,0,1]),
               ('a','b'),((0,1),(1,)),np.array([21,22,23]))
        b=Week(np.array([50.,70.]),1,np.array([0,0]),
               ('c',),((0,),),np.array([24,25]))
        X,names,means,stds=pilot.add_context(np.ones((5,2)),[a,b],
                                                  {21:0,22:1,23:2,24:3,25:4})
        self.assertEqual(X.shape,(5,6))
        self.assertEqual(len(names),4)
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue(np.allclose(X[:,2:].mean(axis=0),0))
        self.assertTrue(np.allclose(X[:,2:].std(axis=0),1))

    def test_inverse_correction(self):
        a,h=.8,30.
        scores=np.linspace(-200.,200.,401)
        booked=200.
        shown=np.clip(scores/a,max(-h,1-booked),h)
        corrected=np.sign(shown)*np.minimum(a*np.abs(shown),
                    np.maximum(h-(1-a)*np.abs(shown),0))
        expected=np.clip(scores,-a*h,a*h)
        self.assertTrue(np.allclose(corrected,expected))
        self.assertTrue(np.all(np.diff(corrected)>=-1e-12))

    def test_positive_duration_floor(self):
        a,h=.8,30.;booked=10.
        shown=np.clip(-100/a,max(-h,1-booked),h)
        response=np.sign(shown)*min(a*abs(shown),max(h-(1-a)*abs(shown),0))
        self.assertGreaterEqual(booked+response,1)
        self.assertAlmostEqual(booked+response,2.8)

if __name__=='__main__':
    unittest.main()
