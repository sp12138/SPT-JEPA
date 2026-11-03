"""Uniformly pin original RANDOM encoders after an exact-init gate rejection.

Use the existing trainer's encoder-only initialization field. No trained state,
architecture,objective,optimizer or data change; retain all original runs.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path


def main(root):
    root=Path(root);old=json.loads((root/'plan.json').read_text());new=copy.deepcopy(old)
    destination=root/'training-paired-v2';destination.mkdir(exist_ok=False)
    for recipe in new['recipes']:
        if recipe['arm']=='native-lewm':continue
        source=root/f"training/temporal-forecast-seed{recipe['seed']}/initial-model"
        cfg=json.loads((Path(recipe['run'])/'config.json').read_text())
        recipe['original_run']=recipe['run'];run=destination/recipe['name'];run.mkdir()
        cfg.update(output=str(run),initial_encoder_checkpoint=str(source),
                   initial_encoder_scope='Original same-seed RANDOM encoder captured before the first MetaWorld update. Uniformly loaded by all three AS arms;not trained/pretrained. Other model tensors,optimizer and schedule fresh.')
        path=run/'config.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
        recipe.update(run=str(run),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      original_random_encoder_sha256=hashlib.sha256((source/'weights.pt').read_bytes()).hexdigest())
    new.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               supersedes_for_formal_runs=str(root/'plan.json'),
               scope='Original two-update technical outcomes retained. Exact initializer gate rejected one encoder projection tensor;cause unassigned. Six AS arms fresh with uniform original RANDOM encoder load through existing native trainer. Two unchanged native controls reused without repeating SGD. No architecture/loss/data/schedule changes or longer budget.')
    (root/'plan-paired-v2.json').write_text(json.dumps(new,indent=2)+'\n');print(json.dumps(new))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    main(Path(parser.parse_args().root))
