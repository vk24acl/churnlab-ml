import unittest
import numpy as np
from churnlab.pipeline import make_data,split_indices,targeting,choose_threshold,score_logit
from sklearn.dummy import DummyClassifier

class PipelineTests(unittest.TestCase):
    def test_chronological_disjoint_splits(self):
        parts=split_indices(6000)
        self.assertEqual(len(np.unique(np.concatenate(parts))),6000)
        for a,b in zip(parts,parts[1:]):self.assertLess(a.max(),b.min())
    def test_deterministic_generator(self):
        x,y=make_data(500);a,b=make_data(500)
        np.testing.assert_array_equal(x,a);np.testing.assert_array_equal(y,b)
        self.assertEqual(x.shape,(500,6));self.assertEqual(set(y),{0,1})
    def test_targeting_uses_ranking(self):
        report=targeting(np.array([0,1,1,0]),np.array([.2,.9,.8,.1]),.5)
        self.assertEqual(report['precision'],1);self.assertEqual(report['recall'],1);self.assertEqual(report['lift'],2)
    def test_capacity_rounds_up(self):
        self.assertEqual(targeting(np.array([1,0,0]),np.array([.9,.2,.1]),.1)['capacity'],1)
        with self.assertRaises(ValueError):targeting(np.array([1]),np.array([.5]),0)
    def test_threshold_separates_known_labels(self):
        threshold=choose_threshold(np.array([0,0,1,1]),np.array([.1,.2,.7,.9]))
        self.assertEqual(threshold,.7)
    def test_constant_baseline_logit_finite(self):
        x=np.zeros((10,6));m=DummyClassifier(strategy='prior').fit(x,[0]*8+[1]*2)
        self.assertTrue(np.isfinite(score_logit(m,x)).all())

if __name__=='__main__':unittest.main()
