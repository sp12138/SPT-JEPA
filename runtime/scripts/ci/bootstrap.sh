#!/usr/bin/env bash
# CI-host orchestration tools and Koala authentication. No workload dependencies.
set -euo pipefail
python3 -m pip install --quiet pyyaml==6.0.3 boto3==1.43.58
VERSION=v0.1.2
curl -fsSL -m 120 -o /usr/local/bin/koala-cli \
  "https://mirrors.tencent.com/repository/generic/koala-client/${VERSION}/koala-cli-linux-amd64"
echo '2d71b7170dd6ebb23c0b9144207c23654a3f616192807f21012dff1dbfb33888  /usr/local/bin/koala-cli' | sha256sum -c -
chmod +x /usr/local/bin/koala-cli
apt-get update -qq
apt-get install -y -qq rsync openssh-client
python3 - <<'PY'
import os
from pathlib import Path
import yaml

def required(name):
    value = os.environ[name]
    if not value or '{{' in value:
        raise ValueError(f'{name} is unresolved')
    return value

config = {'current_profile': 'ci', 'profiles': {'ci': {
    'project_id': 'ieg-infra', 'user': 'api-ieg-infra',
    'secret_id': required('KOALA_APPID'), 'secret_key': required('KOALA_SECRETKEY'),
    'default_cluster': 'tenc-aws-yfxn-us-west', 'default_namespace': 'ieg-infra',
}}}
path = Path.home() / '.koala/config.yaml'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(yaml.safe_dump(config))
path.chmod(0o600)
PY
