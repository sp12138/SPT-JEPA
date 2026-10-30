"""Small CPU contracts for the new joint-case, fixed-seed summary only."""
import unittest

from experiments.carrier_progress_20261031.collect_learning import paired_outcomes, paired_summary


class SharedCaseContracts(unittest.TestCase):
    def test_opposite_seed_effects_cancel_for_every_shared_case(self):
        first=[i % 2 for i in range(50)];second=[1-x for x in first]
        rows=[dict(seed=1,**paired_outcomes(first,second,100)),
              dict(seed=2,**paired_outcomes(second,first,100))]
        result=paired_summary(rows,100)
        self.assertEqual(result['mean_difference_pp'],0)
        self.assertEqual(result['shared_case_bootstrap_ci95_pp'],[0,0])

    def test_requires_both_independent_training_seeds(self):
        row=dict(seed=1,**paired_outcomes([1]*50,[0]*50,100))
        with self.assertRaises(AssertionError):
            paired_summary([row,row],100)


if __name__=='__main__':
    unittest.main()
