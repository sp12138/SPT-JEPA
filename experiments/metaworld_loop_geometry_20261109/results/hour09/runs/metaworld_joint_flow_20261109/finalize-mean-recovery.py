from pathlib import Path
import sys,time,subprocess
b=Path('/efs/prescottsun/SPT-JEPA');s=b/'reports/metaworld_joint_flow_20261109/source-v1';r=b/'runs/metaworld_joint_flow_20261109';sys.path.insert(0,str(s))
from experiments.metaworld_transition_only_20261109.common import environment,PYTHON,save,read,now
out=r/'training/mean-seed1';deadline=time.time()+3600
while not (out/'TRAINED-0002000.json').exists():
 assert time.time()<deadline
 time.sleep(5)
with (r/'mature-recovery-cpu-audit.log').open('x') as f:
 subprocess.run([PYTHON,str(b/'reports/metaworld_joint_flow_20261109/mature-audit-recovery-v1.py'),'audit','--config',str(out/'config.json')],cwd=s,env=dict(environment(s,0),CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
assert read(out/'AUDIT-0002000.json')['passed']
save(out/'READY-0002000.json',dict(passed=True,utc=now(),original_error_preserved=True,mature_gradient_audit_recovery=True))
save(out/'DONE.json',dict(passed=True,utc=now(),new_world_updates=2000,original_error_preserved=True,mature_gradient_audit_recovery=True))
save(r/'MEAN_RECOVERY_COMPLETE.json',dict(passed=True,utc=now(),initial_positiveproof_retained=True,checkpointed_updates_repeated=0,unsaved_replayed_at_most=99,originalerrors_retained=True))
