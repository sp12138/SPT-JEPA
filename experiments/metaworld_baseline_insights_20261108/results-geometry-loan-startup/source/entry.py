import argparse,importlib.util
from pathlib import Path
p=argparse.ArgumentParser()
for n in ['checkpoint','data','manifest','output']:p.add_argument('--'+n,required=True)
p.add_argument('--geometry')
p.add_argument('--budget',type=int,default=500);p.add_argument('--held',action='store_true')
a=p.parse_args()
spec=importlib.util.spec_from_file_location('support_evaluator',Path(__file__).with_name('evaluate_support.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.main(**vars(a))
