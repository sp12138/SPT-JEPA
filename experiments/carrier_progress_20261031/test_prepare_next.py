"""CPU contracts for allocation and the cross-epoch sampler audit."""
import unittest
from types import SimpleNamespace

from experiments.carrier_progress_20261031.prepare_next import allocations,resume_sequence


class NextStageContracts(unittest.TestCase):
    def test_reviewed_allocation(self):
        rows=allocations()
        self.assertEqual(len(rows),12)
        self.assertEqual([r['gpu'] for r in rows[:8]],list(range(8)))
        self.assertTrue(all(r['host']=='dev2' and r['family']=='carrier' for r in rows[:8]))
        self.assertEqual([r['gpu'] for r in rows[8:]],list(range(4)))
        self.assertTrue(all(r['host']=='dev13' and r['family']=='temporal-forecast' for r in rows[8:]))
        self.assertEqual([(r['training_seed'],r['arm']) for r in rows if r['task']=='cube' and r['family']=='carrier'],
                         [(1,'identity'),(2,'identity')])

    def test_native_sampler_across_multiple_loader_epochs(self):
        dataset=SimpleNamespace(natural_length=131,encode_epoch_index=True)
        # Native loader length128; cursor400 is beyond three complete loaders.
        report=resume_sequence(dataset,cursor=400,seed=2,batch_size=8)
        self.assertEqual(report['native_loader_length'],128)
        self.assertEqual(len(report['next_indices']),32)
        self.assertTrue(report['passed'])


if __name__=='__main__':unittest.main()
