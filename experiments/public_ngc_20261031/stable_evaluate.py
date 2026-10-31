"""Run the original cost adapter with one uniform stable NGC implementation."""
import argparse
from pathlib import Path
from common import read,write,sha,reexec


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--inner',action='store_true');a=p.parse_args()
    s=read(a.spec);assert s['numeric_variant']=='uniform-fp32-direct-attachment'
    if not a.inner:reexec(s,__file__)
    import objective,evaluate
    from stable_cost import StableNGCCost
    objective.NGCCost=StableNGCCost
    evaluate.main()
    path=Path(s['output'])/'metrics.json';r=read(path)
    r['ngc_diagnostic']['numeric_variant']=s['numeric_variant']
    r['ngc_diagnostic']['stable_cost_sha256']=sha(Path(__file__).with_name('stable_cost.py'))
    write(path,r)
