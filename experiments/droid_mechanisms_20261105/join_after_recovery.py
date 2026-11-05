"""Transfer the original two GPUs only after their fixed recovery work exits."""
import argparse
import json
import os
from pathlib import Path
import time

from common import OLD, PYTHON, arm_root, bootstrap, environment


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--campaign',required=True);a=p.parse_args()
    root=Path(a.root);dump=bootstrap(arm_root(root,'tf'))
    deadline=json.loads((root/'APPROVED.json').read_text())['deadline']
    old_workers=json.loads((root/'old-evaluation-launch-dev15.json').read_text())
    while time.time()<deadline:
        if (root/'HOLD').exists():
            break
        marker=OLD/'recovery-20261105/HOLD'
        record=json.loads(marker.read_text()) if marker.exists() else {}
        completed=record.get('passed') is True and record.get('completed_episodes')==1584
        active=False
        for item in old_workers:
            proc=Path(f"/proc/{item['pid']}/cmdline")
            if proc.exists():
                cmd=proc.read_bytes().replace(b'\0',b' ').decode()
                if 'old_evaluation.py' in cmd or 'recovery-finish-20261105-v1.py eval' in cmd:
                    active=True
        if completed and not active:
            command=[PYTHON,'-u',a.campaign,'actor','--root',str(root),'--host','dev15',
                     '--gpus','0','1','--deadline',str(deadline)]
            dump(root/'recovery-v2/joined-dev15-g0.json',dict(time=time.time(),pid=os.getpid(),command=command,
                 original_fixed_evaluations_completed=True,old_workers_exited=True))
            os.execve(PYTHON,command,environment([0,1]))
        time.sleep(20)
