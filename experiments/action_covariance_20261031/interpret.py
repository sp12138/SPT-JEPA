"""Post-completion same-controller comparisons; no model or task execution."""
import argparse,json
from pathlib import Path
import numpy as np
from collect import ci,read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);a=p.parse_args();root=Path(a.report);r=read(root/'results.json');assert r['complete48']and r['all_exits0']
    rows={(x['task'],x['family'],x['training_seed'],x['case_set'],x['mode']):np.asarray(x['successes'],float)for x in r['rows']};comparisons=[]
    for task,method in [('cube','local1024'),('tworoom','F64free')]:
        for baseline in ['TF64','native-raw']:
            for case in ['primary','fresh']:
                for mode in ['original','correlated']:
                    d=np.stack([rows[(task,method,s,case,mode)]-rows[(task,baseline,s,case,mode)]for s in [1,2]])
                    comparisons.append(dict(task=task,method=method,baseline=baseline,case_set=case,mode=mode,perseed_delta_pp=(100*d.mean(1)).tolist(),**ci(d.mean(0))))
    write(root/'model-comparisons.json',dict(comparisons=comparisons,limitation='Conditional on2 trained seeds and observed environment realizations; capacities/exposure differ; no equivalence or causal representation claim.'))
    print(json.dumps(comparisons))
if __name__=='__main__':main()
