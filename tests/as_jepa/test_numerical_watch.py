import unittest
from as_jepa_tools.numerical_watch import bad_tail


class NumericalWatchContracts(unittest.TestCase):
    def test_requires_two_consecutive_nonfinite_logged_training_losses(self):
        self.assertFalse(bad_tail([dict(step=1,loss=1.),dict(step=2,loss=float("nan"))]))
        self.assertFalse(bad_tail([dict(step=1,loss=float("nan"))]))
        self.assertTrue(bad_tail([dict(step=1,loss=float("nan")),dict(step=2,loss=float("inf"))]))
        self.assertFalse(bad_tail([dict(step=1,loss=float("nan")),dict(step=2,loss=2.)]))
        self.assertFalse(bad_tail([dict(state="running"),dict(state="failed")]))


if __name__=="__main__":
    unittest.main()
