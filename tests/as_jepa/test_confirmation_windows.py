import unittest
import numpy as np
from as_jepa_tools.evaluate import select_windows,excluded_window_indices


class ToyWindows:
    window=16
    offsets=np.array([0])
    def __init__(self):
        self.states=np.zeros((2048,7),np.float32)
        self.states[:,2]=((np.arange(2048)//32)%2)*30
    def __len__(self):return 2000
    def open(self):pass
    def locate(self,index):return 0,int(index)+8


class ConfirmationWindowContracts(unittest.TestCase):
    def test_confirmation_ids_are_deterministic_and_exclude_every_diagnostic_id(self):
        data=ToyWindows()
        original=select_windows(data,256,seed=0)
        confirmation=select_windows(data,256,seed=20261028,exclude=original)
        self.assertEqual(len(original),256)
        self.assertEqual(len(confirmation),256)
        self.assertFalse(set(original).intersection(confirmation))
        self.assertEqual(confirmation,select_windows(data,256,seed=20261028,exclude=original))
        moving=[]
        for i in confirmation:
            _,start=data.locate(i);state=data.states[start:start+16,2:4]
            moving.append(np.linalg.norm(state-state[0],axis=1).max()>10)
        self.assertEqual(sum(moving),128)


    def test_exclusion_removes_entire_overlapping_clips_in_the_same_episode(self):
        from types import SimpleNamespace
        data=SimpleNamespace(episodes=np.array([2,8]),counts=np.array([100,80]),ends=np.array([100,180]))
        blocked=excluded_window_indices(data,[dict(episode=2,start=50),dict(episode=8,start=0)])
        self.assertEqual(set(range(35,66))|set(range(100,116)),blocked)
        self.assertNotIn(34,blocked);self.assertNotIn(66,blocked);self.assertNotIn(116,blocked)


if __name__=="__main__":
    unittest.main()
