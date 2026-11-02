# When Does the SSD Matter?

CS736 starter: sharing DRAM between experts and paused-session KV in single-GPU
MoE inference. This repository is the authoritative source. Windows runs the
controller; Linux is a replaceable execution target. No remote IDE is required.

The implementation uses `allenai/OLMoE-1B-7B-0924-Instruct` in BF16, pinned to
`7f1c97f440f06ce36705e4f2b843edb5925f4498`. It validates mechanisms on this model;
it does not establish performance results for a larger Qwen3 FP8 system.

Current validation: **19 remote tests pass, but the real pretrained smoke and pilot
are blocked by repeated execution-target restarts that erase `/tmp`.** See
[the validation record](docs/validation.md) and [selected evidence](summaries/validation.json).

## Laptop workflow

Requirements on Windows: PowerShell 5.1+, Python 3.10+, Git, and OpenSSH `ssh/scp`.
No Windows PyTorch installation is needed. Keep the private SSH key outside the
repository. Existing host-key verification remains enabled. For a new target,
verify its host fingerprint through your provider before accepting it with SSH.

Run from the local checkout in PowerShell:

```powershell
Copy-Item configs/machine.example.json configs/machine.local.json
# Edit the ignored local JSON with the target and the laptop's private-key path.
python -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action doctor -Wait
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action setup
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action status
# After setup reports completed with exit_code 0:
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action fixtures -Wait
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action smoke
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action status
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action fetch
# After the smoke summary reports complete=true and passed=true:
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action pilot
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action status
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action fetch
```

Do not overwrite an existing `machine.local.json` when repeating these commands.
`-Wait` is optional for doctor/setup/smoke/pilot. Without it, the launch returns
immediately. Use `-RunId <printed-id>` with status/fetch to retrieve an earlier
run. `deploy` creates a release without launching anything. Every launch action
deploys the current local working tree, so changes need not be committed first.
Never interpret a successful SSH launch as a completed experiment: inspect the
durable exit code and the experiment's correctness report.

The EFabric gateway tested here requires `"scp_protocol": "legacy"` (`scp -O`);
the controller also supports `"sftp"`. Both retain SSH host-key verification.
The controller was tested in native Windows PowerShell, not WSL. If using WSL,
keep the checkout at its Windows location and invoke the controller explicitly:

```bash
cd /mnt/e/UWMadisonCourses/2026Fall/736OS/finalproject/736FinalMoe
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$PWD/scripts/remote.ps1")" -Action status
```

## Fresh Linux machine or replacement target

Use Ubuntu 24.04/x86_64 with Python 3.12, Bash, `flock`, CA certificates, Git, and
a CUDA-capable NVIDIA driver. A CUDA toolkit is unnecessary: the pinned PyTorch
wheel includes its CUDA 12.4 runtime. A driver's displayed CUDA version alone
does not prove PyTorch compatibility. This configuration requires BF16 support
and approximately 24 GiB GPU memory. The default PyTorch allocator cap is 22 GiB.

After publishing this repository, clone its actual URL. No remote URL has been
chosen automatically. For the laptop-controller path, fill a new ignored machine
JSON, verify its SSH host key, and repeat doctor/setup/smoke. Models and environments
are recreated from pinned source specifications; never copy a Windows environment.

For a Linux clone, the equivalent durable workflow is:

```bash
git clone <your-published-repository-url> 736FinalMoe
cd 736FinalMoe
runtime_root="$HOME/cs736-moe-runtime"
task_scratch="/tmp/cs736-moe-$(id -u)"
python3 scripts/linux.py --root "$runtime_root" --scratch "$task_scratch" doctor
python3 scripts/linux.py --root "$runtime_root" --scratch "$task_scratch" setup
python3 scripts/linux.py --root "$runtime_root" --scratch "$task_scratch" status --run <setup-run-id>
# Once setup completes, run doctor again to audit the installed CUDA stack:
python3 scripts/linux.py --root "$runtime_root" --scratch "$task_scratch" doctor
python3 scripts/linux.py --root "$runtime_root" --scratch "$task_scratch" smoke
python3 scripts/linux.py --root "$runtime_root" --scratch "$task_scratch" status --run <smoke-run-id>
```

`setup` invokes `bash scripts/bootstrap.sh ROOT SCRATCH` in its immutable release.
It creates a rootless environment at `ROOT/envs/<requirements.lock SHA256>`,
checks the pinned package set/CUDA/BF16, downloads the immutable model remotely,
checks weight hashes, then prepares expert files. A matching completed environment
is reused. Model acquisition itself is reproducible as
`ENV/bin/python scripts/acquire_model.py --config configs/model.json --scratch SCRATCH`.
The dependency lock targets Python 3.12 only. No system driver or host setting is
changed. Linux-clone/WSL convenience paths are separate from the tested native
Windows controller; consult the selected acceptance summary for checks actually run.

## Architecture and accounting

`moe_shared/run.py` first loads a short-context, fully resident reference. It saves
CPU logits for the recorded trajectory, deletes that model, releases its GPU
allocations, then runs each offloaded case sequentially. The reference logits and
comparison scratch have an explicit reservation in every offloaded host pool.

`prepare.py` converts the checkpoint once into 1,024 expert-addressable BF16 raw
files and separate dense tensors, with shapes, byte offsets, and SHA256 hashes in
a manifest. It exposes only individual safetensors source tensors during conversion.
Conversion is outside timed experiments. A miss reads only the selected expert.

`experts.py` constructs OLMoE on the meta device, loads dense tensors into CUDA
through counted 8 MiB CPU staging, and replaces only expert modules with proxies.
The pinned Transformers attention and router remain intact. Nonempty expert calls
load real weights, transfer them to CUDA, assert CUDA placement, and execute the
original `OlmoeMLP`. Zero-token expert calls return the mathematically empty result.
No checkpoint state dictionary or hidden CPU expert model remains during serving.

| Resource | Location and rule |
| --- | --- |
| Dense layers/router/attention | CUDA; resident during each case |
| Active-session KV | CUDA; explicit maximum context and derived payload bound |
| Hot experts | CUDA; global LRU with a fixed byte budget |
| Host experts | The one shared host pool; deterministic LRU eviction |
| Paused retained KV | Same pool; can evict host experts |
| CPU staging and validation tensors | Same pool, reserved before allocation |
| Expert backing and spilled KV | Same task-owned filesystem under `/tmp` |
| Python/allocator/runtime memory, OS page cache | Outside application pool; separate process/cgroup measurements |

All transfers use synchronous pageable memory; pinned-buffer usage is explicitly
zero. GPU payload accounting includes dense weights, expert cache, active KV bound,
and a workspace reservation. The PyTorch allocator is capped separately; CUDA
context/driver allocations are outside that allocator. Pool accounting counts
tensor payloads, not Python object overhead or allocator retention.

`kv.py` implements the actual three transitions. RETAIN copies KV into counted
host allocations. SPILL copies through counted staging, writes and fsyncs raw KV
plus metadata, releases the buffers, then verifies/checksums and reloads on return.
RECOMPUTE drops KV and replays the saved prefix when the next turn arrives. All
actions clear the original GPU cache in place; resume cannot reuse stale GPU KV.
Token history, the pending final output token, cache sequence length, seen-token
metadata, attention masks, cache positions, and position IDs are preserved.

## Controlled workload and validation

The scheduler alternates one token step between two sessions. One session completes
a turn and pauses while the other continues. Later-turn arrival is previous-turn
completion plus configured think time. All cases consume the exact same recorded
real-text prompt/output token IDs from `trajectories.json`; outputs are teacher-forced,
not free-running model generations. A seed alone would not provide this control.

Every token's next-logit vector is compared against the resident reference with
`abs(error) <= 0.125 + 0.02 * abs(reference)`. Reports include maximum/mean errors,
per-checkpoint argmax agreement, and returning-turn agreement. Acceptance requires
all logits within tolerance, at least 95% overall argmax agreement, actual expert
misses/CUDA execution, each selected KV action, and an active step while paused.
The real pretrained smoke must pass all three actions. Tiny CUDA fixtures additionally
test single-slot eviction, budget cleanup, stale-cache removal, and corrupt spill
rejection; tiny models do not substitute for pretrained acceptance.

The smoke uses 8-token prefixes and deliberately small expert caches. The pilot
uses a 96-token paused prefix, two host allocations, one fixed GPU expert budget,
the same LRU policy, and all three actions. Host budgets are derived from an expert
slot count plus 0.5 MiB slack and the counted validation reservation. Retained KV
therefore competes with expert slots near a budget boundary. These are artificial
mechanism-stressing limits, not a claim that the physical host is full.

The initial pilot has a 900-second wall cap including model loading and fixture
checks, excluding installation/download/conversion. Deadline interruption is a
failure/partial report, never a successful trial. Application caches start empty
per case. Filesystem page cache is uncontrolled and likely warm after preparation
and earlier cases. Trial order is fixed; the small pilot is descriptive and does
not support causal policy rankings or strong tail-latency claims.

`events.jsonl` records arrival, queue, pause, resume, expert miss, token, completion,
and failure timestamps. JSONL/CSV measurements report initial/returning TTFT,
queueing, per-session ITL, TPOT, transfer bytes, expert/cache statistics, KV actions,
GPU peaks, and process/cgroup counters. TTFT ends when logits and token selection
are CPU-visible. Correctness checks, audit sampling, and logs add overhead to later
token intervals. Component durations overlap; never add them into a critical path.
Counts, means, medians, and maxima are reported; there is no p99 estimate.

## Storage and reproducibility limits

The scratch guard permits at most 35 GiB of task-owned logical data and requires
20 GiB free filesystem headroom. It checks downloads plus converted duplicates,
conversion, and spill operations. Free space is shared and cannot be reserved by
this guard. Environment files/logs live under the runtime root on `/home`; large
model/backing files live under the task-owned `/tmp` directory. `/home` and `/models`
on EFabric are JuiceFS. The audit records mount, filesystem, `lsblk`, and visible
sysfs evidence. Unless physical backing is established, `/tmp` is labeled
**ext4; physical backing unverified**. No local-NVMe performance claim is justified.

An application expert miss measures a logical file read. `/proc/self/io`, visible
cgroup `io.stat`, and block counters are reported separately. Filesystem cache can
serve those reads without device I/O; other work can affect cgroup/device counters.
The audit records memory current/stat/events, swap and accessible ancestor limits,
CPU quota/affinity, free space, and GPU/software capability. Hidden ancestor limits
cannot be inferred. Sampled cgroup current provides a run-local lower bound on peak;
kernel `memory.peak` may include earlier setup or other work and is not reset.
No global cache drop, raw-device writes, or driver changes are performed.

Filesystem free space does not reveal a Kubernetes ephemeral-storage quota or
guarantee that a workspace will remain alive. If the target restarts, `/tmp` may
vanish even while the remote home directory and logs survive. The controller
reports disappeared workers as `lost`, and smoke/pilot fail preflight on missing
model/backing data. Inspect portal eviction events or use a stable replacement
target before retrying setup. `fixtures` validates CUDA mechanisms independently
of the large pretrained download; passing it is not pretrained acceptance.

Deployment snapshots select current bytes from explicit source roots, including
new/uncommitted files, and exclude credentials, local config, Git data, environments,
weights, caches, and results. The manifest records commit (when present), dirty
status, per-file hashes, and archive hash in its deployment receipt. Remote content
is verified before a read-only release directory is published. Workers verify it
again, run that release, and store logs/status outside it. Run IDs, PID birth times,
and explicit exit codes survive SSH disconnects. A task-root lock prevents overlapping
experiments/setup launched by this workflow; unrelated GPU users remain possible.

Fetch downloads only bounded compact report types, verifies the archive and every
file, and writes `results/<run-id>/` locally with a transfer receipt. Weights and
expert/KV binaries are excluded. Keep full results ignored. To deliberately select
a small, hash-linked report for Git after a successful fetch:

```powershell
python scripts/summarize.py results/<smoke-run-id> results/<pilot-run-id> --output summaries/acceptance.json
```

Review `git diff` and `git status` before staging. This implementation does not commit,
configure a remote, or push automatically. Add your repository remote, stage the
source/config/examples and selected summary, commit after review, then push. Never
stage the ignored local machine configuration or SSH key. The source manifest ties
results to the actual tested working-tree bytes even before that commit exists.

