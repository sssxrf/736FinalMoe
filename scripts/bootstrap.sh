#!/usr/bin/env bash
# Run from a verified source release (or a fresh Linux clone). No sudo is used.
set -euo pipefail
umask 077
if [[ $# -ne 2 ]]; then
  echo 'usage: bash scripts/bootstrap.sh REMOTE_ROOT TASK_SCRATCH_ROOT' >&2
  exit 2
fi
root=$(realpath -m -- "$1")
scratch=$(realpath -m -- "$2")
release=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
case "$root" in /|/home|/tmp) echo "Refusing broad runtime root: $root" >&2; exit 2;; esac
case "$scratch" in /tmp/*) ;; *) echo 'Scratch must be a task-owned subdirectory of /tmp' >&2; exit 2;; esac
[[ "$scratch" != /tmp ]] || exit 2
mkdir -p -- "$root/envs" "$root/locks" "$root/setup" "$scratch"
audit_dir=${MOE_RUN_DIR:-"$root/setup"}
mkdir -p -- "$audit_dir"
exec 9>"$root/locks/setup.lock"
flock -n 9 || { echo 'Another setup is running for this task' >&2; exit 3; }
if [[ ! -e "$scratch/.moe-task-scratch" ]]; then
  if [[ -n $(find "$scratch" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo 'Refusing an unmarked nonempty scratch directory' >&2
    exit 2
  fi
  printf '%s\n' "$root" >"$scratch/.moe-task-scratch"
fi
[[ $(cat "$scratch/.moe-task-scratch") == "$root" ]] || { echo 'Scratch belongs to another runtime root' >&2; exit 2; }
cd -- "$release"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$scratch/hf-metadata"
export TMPDIR="$root/setup/tmp"
mkdir -p -- "$TMPDIR"
python3 - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 12) or platform.machine() != 'x86_64':
    raise SystemExit('This lock targets Linux x86_64 Python 3.12; install that interpreter first.')
PY
python3 -m moe_shared.audit --output "$audit_dir/audit-before.json" --scratch "$scratch"
lock_sha=$(sha256sum requirements.lock | cut -d ' ' -f 1)
env_dir="$root/envs/$lock_sha"
if [[ ! -f "$env_dir/.complete" ]]; then
  python3 - "$root" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 28 * 1024**3:
    raise SystemExit('Environment installation requires 8 GiB plus 20 GiB free filesystem headroom.')
PY
  # --without-pip works on Ubuntu images lacking the python3-venv ensurepip
  # package. Bootstrap a hash-verified pinned pip wheel, without root packages.
  python3 -m venv --without-pip "$env_dir"
  "$env_dir/bin/python" - <<'PY'
import hashlib, io, sysconfig, urllib.request, zipfile
url = 'https://files.pythonhosted.org/packages/ef/7d/500c9ad20238fcfcb4cb9243eede163594d7020ce87bd9610c9e02771876/pip-24.3.1-py3-none-any.whl'
expected = '3790624780082365f47549d032f3770eeb2b1e8bd1f7b2e02dace1afa361b4ed'
data = urllib.request.urlopen(url, timeout=120).read()
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit('Bootstrap pip wheel hash mismatch')
with zipfile.ZipFile(io.BytesIO(data)) as wheel:
    wheel.extractall(sysconfig.get_path('purelib'))
PY
  "$env_dir/bin/python" -m pip install --no-cache-dir -r requirements.lock
  "$env_dir/bin/python" -m pip check
  "$env_dir/bin/python" - "$env_dir/environment.json" "$lock_sha" <<'PY'
import importlib.metadata as m, json, platform, subprocess, sys
import torch, transformers
assert torch.__version__ == '2.5.1+cu124'
assert transformers.__version__ == '4.46.3'
assert torch.cuda.is_available(), 'CUDA unavailable to installed PyTorch'
assert torch.cuda.is_bf16_supported(), 'BF16 unsupported by target GPU'
packages = {d.metadata['Name']: d.version for d in m.distributions()}
data = {'python': sys.version, 'platform': platform.platform(), 'dependency_lock_sha256': sys.argv[2],
        'packages': packages, 'torch_cuda_build': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'compute_capability': torch.cuda.get_device_capability(0)}
with open(sys.argv[1], 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
PY
  printf '%s\n' "$lock_sha" >"$env_dir/.complete"
else
  [[ $(cat "$env_dir/.complete") == "$lock_sha" ]] || { echo 'Environment completion marker mismatch' >&2; exit 4; }
  echo "Reusing dependency-locked environment $lock_sha"
fi
"$env_dir/bin/python" scripts/acquire_model.py --config configs/model.json --scratch "$scratch"
"$env_dir/bin/python" -m moe_shared.prepare --model-dir "$scratch/model" --output "$scratch/backing"
"$env_dir/bin/python" -m moe_shared.audit --output "$audit_dir/audit-after.json" --scratch "$scratch"
printf 'Setup complete: env=%s model=%s backing=%s\n' "$env_dir" "$scratch/model" "$scratch/backing"
