import unittest
import numpy as np
import torch
from as_jepa_tools.geometry_planning import GeometryGoal
class Toy:
 def __init__(self,jacobian,transform=None):self.jacobian=jacobian;self.transform=transform
 def rollout(self,info,actions):
  z=actions.flatten(2)@self.jacobian.T
  if self.transform is not None:z=z@self.transform.T
  info["predicted_emb"]=z[:,:,None];return info
class GeometryPlanningContracts(unittest.TestCase):
 def test_finite_difference_cost_is_coordinate_covariant(self):
  torch.manual_seed(10);d=6;n=8;j=torch.randn(d,n);matrix=torch.randn(d,d);cov=matrix@matrix.T+torch.eye(d)
  transform=torch.linalg.qr(torch.randn(d,d)).Q@torch.diag(torch.linspace(.4,3,d))
  actions=torch.randn(2,16,4,2);info=dict(pixels=torch.randn(2,16,1,3,2,2))
  g=GeometryGoal(cov.numpy(),"gramian",rank=8);h=GeometryGoal((transform@cov@transform.T).numpy(),"gramian",rank=8)
  g.prepare(Toy(j),info,actions);h.prepare(Toy(j,transform),info,actions)
  pred=Toy(j).rollout({},actions)["predicted_emb"];goal=torch.randn(2,1,d)
  a=g(dict(predicted_emb=pred,goal_emb=goal));b=h(dict(predicted_emb=pred@transform.T,goal_emb=goal@transform.T))
  torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)
  count=g.extra_rollouts;g.prepare(Toy(j),info,actions);self.assertEqual(count,g.extra_rollouts)
 def test_cache_distinguishes_translated_images_with_identical_moments(self):
  torch.manual_seed(11)
  j=torch.randn(6,8);g=GeometryGoal(np.eye(6),"gramian",rank=4)
  actions=torch.randn(1,3,4,2)
  pixels=torch.arange(12,dtype=torch.float32).reshape(1,1,1,3,2,2).expand(1,3,1,3,2,2)
  info=dict(pixels=pixels)
  g.prepare(Toy(j),info,actions);count=g.extra_rollouts
  g.prepare(Toy(j),info,actions);self.assertEqual(g.extra_rollouts,count)
  translated=pixels.roll(1,-1)
  self.assertEqual(float(pixels.sum()),float(translated.sum()))
  self.assertEqual(float(pixels.square().sum()),float(translated.square().sum()))
  g.prepare(Toy(j),dict(pixels=translated),actions)
  self.assertGreater(g.extra_rollouts,count);self.assertEqual(g.prepare_calls,2)
if __name__=="__main__":unittest.main()
