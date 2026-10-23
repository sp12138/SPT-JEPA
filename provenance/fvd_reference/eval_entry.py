"""Coordinate complete checkpoints before invoking the unchanged Native evaluator."""
from pathlib import Path
import runpy,sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_io_v2 import verified_artifact_renames
from fvd_coordination import guarded_run,commit_score,protocol_hash,evaluation_body
J=Path(__file__).resolve().parent
name,ckpt,output=sys.argv[1:4]
def body():
    with verified_artifact_renames(output):
        runpy.run_path(str(evaluation_body()),run_name='__main__')
if '--smoke' in sys.argv:
    body()
else:
    expected=protocol_hash()
    guarded_run(output,body,lambda:commit_score(name,ckpt,output,expected),expected)
