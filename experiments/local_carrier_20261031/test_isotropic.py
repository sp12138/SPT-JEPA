"""Stdlib CPU contracts: one untuned scale, immutable inputs, no GPU provenance."""
import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location('isotropic_prepare',Path(__file__).with_name('prepare_isotropic.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class IsotropicContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name)
        (root/'as_jepa_lewm').mkdir();(root/'as_jepa_lewm/spatial_carrier.py').write_text('fixed model')
        self.src=dict(source=str(root),source_commit='new-fixed-source')
        self.original=dict(source_commit='96aca13f99b25dcf880346a1e699318150139ae5',measured_frames=128,
            scale_floor=1e-6,encoder_initial_sha256='frozen',arms={})
        for arm in ('local','dense'):
            self.original['arms'][arm]=dict(scale_floor_coordinates=0,projection_sha256=arm,
                raw_std_min=.001,raw_std_mean=.5005,raw_std_max=1.,parameter_counts=dict(total=123),
                model_kwargs=dict(spatial_mode=arm,state_slots=64,state_width=16,
                    spatial_mean=[i/1024 for i in range(1024)],spatial_scale=[.001]*512+[1.]*512))
        self.path=root/'original.json';self.path.write_text(json.dumps(self.original));self.original_bytes=self.path.read_bytes()

    def test_exact_shared_scalar_no_input_mutation(self):
        before=copy.deepcopy(self.original);out=p.isotropic_record(self.original,self.path,self.src)
        self.assertEqual(self.original,before);self.assertEqual(self.path.read_bytes(),self.original_bytes)
        self.assertEqual(out['normalization'],'global_rms');self.assertIsNone(out['scale_floor'])
        self.assertEqual(out['execution']['kind'],'CPU JSON statistics derivation only')
        for arm in ('local','dense'):
            old=self.original['arms'][arm];new=out['arms'][arm]
            self.assertEqual(new['model_kwargs']['spatial_mean'],old['model_kwargs']['spatial_mean'])
            self.assertEqual(new['model_kwargs']['spatial_scale'],[math.sqrt((.001**2+1)/2)]*1024)
            self.assertEqual(new['parameter_counts'],old['parameter_counts']);self.assertEqual(new['projection_sha256'],arm)

    def test_clipped_missing_or_nonfinite_variance_is_rejected(self):
        for mode in ('clipped','zero','nonfinite'):
            bad=copy.deepcopy(self.original)
            if mode=='clipped':bad['arms']['local']['scale_floor_coordinates']=1
            elif mode=='zero':bad['arms']['local']['model_kwargs']['spatial_scale'][0]=0
            else:bad['arms']['local']['model_kwargs']['spatial_scale'][0]=float('nan')
            with self.assertRaises(AssertionError):p.isotropic_record(bad,self.path,self.src)

    def test_global_scale_preserves_orthogonal_metric(self):
        points=[(-2.,-.2),(-1.,.1),(1.,-.1),(2.,.2)]
        rotate=lambda x:((x[0]-x[1])/math.sqrt(2),(x[0]+x[1])/math.sqrt(2))
        rotated=[rotate(x) for x in points]
        def variance(points):
            means=[sum(x[j] for x in points)/len(points) for j in range(2)]
            return [sum((x[j]-means[j])**2 for x in points)/len(points) for j in range(2)]
        v,w=variance(points),variance(rotated)
        self.assertAlmostEqual(sum(v),sum(w),places=14)
        delta=[points[0][j]-points[1][j] for j in range(2)];rd=rotate(delta)
        self.assertAlmostEqual(sum(x*x for x in delta)/sum(v),sum(x*x for x in rd)/sum(w),places=14)
        # Coordinate whitening need not commute with the same rotation.
        self.assertGreater(abs(sum(delta[j]**2/v[j] for j in range(2))-sum(rd[j]**2/w[j] for j in range(2))),1.)


if __name__=='__main__':unittest.main()
