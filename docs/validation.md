# Validation record - November 1-2, 2026 (America/Chicago)

**Persistent preparation, verified reuse, disposable-stage recovery, and real
pretrained correctness have passed and been fetched locally.** The successful
smoke exercises genuine CUDA expert misses and retain, spill/reload, and recompute.
The bounded pilot lost its worker during staging after the observed process/mount
environment was replaced. No pilot or SSD-performance result is claimed.

This continuation began at clean Git commit `236706e`, with no configured remote
or applicable `AGENTS.md`. All edits originated in the authoritative Windows
checkout; the user's new `.vscode` files were preserved. No commit or push was
performed. Run manifests identify the exact deployed working-tree snapshots.
Timestamps and run IDs below use UTC.

## Disappearing-file investigation

Historical setup `20261102T023925Z-setup-6b8a78ac` completed at
`02:42:13.884Z`; smoke `20261102T024300Z-smoke-f1470994` started at
`02:43:01.814Z`. Both resolved model preparation to the same
`/tmp/cs736-rxu292-moe` tree, and these runs did not overlap. Smoke found the
previously verified model provenance missing. Earlier setup
`20261102T023003Z-setup-17eea790` lost its worker and had no observed exit code.

Review covered temporary-directory scopes, cleanup handlers, deployment cleanup,
resolved paths, and overlapping launches. No application cleanup of the prepared
model was found: cleanup concerned fixture directories, task-owned KV files, and
uploaded source archives. The former model location itself was disposable `/tmp`.
Historical mount records contain successive pod-scoped roots
`decc1343-8842-48f3-8188-b01427be4883`,
`78636a30-b3d3-4759-96e2-57186df506b0`, and now
`2e69c951-d208-4fea-aae5-fc2fd01a47fc`. This establishes changed pod-scoped mount
identity, not its infrastructure cause. Restart, replacement, or eviction remains
a hypothesis; no portal event proves which occurred. The previous statement that
the process namespace changed was unsupported by the historical records and is
withdrawn. Host boot ID alone cannot establish a container restart.

New setup/recovery records include UTC time, hostname, PID/start identity,
namespace links where accessible, mount identity, resolved paths, and sentinel
existence before and after each stage. From `05:38` through the verified reuse at
`05:51`, hostname was `rxu292-advanced-os-final-0`, self PID/mount namespace inodes
were `4026534176`/`4026534175`, and PID 1 start ticks were `270117667`. PID 1's
namespace links were permission-denied. Artifact mount ID `2738` (JuiceFS,
`0:350`) and scratch mount ID `2734` (ext4, `252:0`) stayed constant. Artifact and
scratch sentinels were created at `05:38:09.102Z` and retained the same hashes;
no further disappearance was observed during these stages. These observations
support continuity over this interval, not an unlimited workspace lifetime.

## Persistent preparation and recovery

Canonical files are plain directories under
`/home/rxu292/advanced-os-final/artifacts/`, outside releases/runs and not symlinks
into `/tmp`. Model revision is
`7f1c97f440f06ce36705e4f2b843edb5925f4498` for
`allenai/OLMoE-1B-7B-0924-Instruct`. The model directory contains nine verified
checkpoint/tokenizer/config files. The converted directory contains 1,024 real
BF16 experts (12 MiB each) and 147 dense tensor files.

Converter version is `2`, binary format version `1`, and converter source SHA256
is `c51a99bb91db56e9c43da5f3b46832d87c3d8ba9bf8d82de093a0193666d0c87`.
Per-file checksums, sizes, and tensor metadata are recorded. Downloads reuse
verified files and resume matching incomplete downloads. Conversion uses stable
`.preparing` storage and verified per-file receipts. Completion is published only
after all files validate. Interrupted-conversion fixtures verify valid files
survive and no completion manifest appears prematurely.

Before preparation, the home filesystem reported 64 GiB total, 5.492 GiB used,
and 58.508 GiB available; the runtime tree occupied 5.159 GiB. After canonical
preparation and subsequent logs, filesystem use was 31.283 GiB with 32.717 GiB
available. Canonical artifacts occupy 25.780 GiB and coexist with environments,
caches, and logs. The 35 GiB task-data cap and 20 GiB free-headroom guards passed.
`quota -s` was unavailable: filesystem capacity was checked, but an independent
account quota was not established. Free space remains shared and is rechecked.

Initial setup completed in 255.19 seconds. The second setup reused the verified
payloads but added `converter_version` to the completion schema; it is not evidence
of byte-identical publication. The third setup took 72.22 seconds to reverify and
reported `reused_completion_manifest: true`, preserving completion timestamp
`05:44:48.214971Z` and exact SHA256
`154ef707c9cf4997f714555102f9faed265ffd6ba9d98835938965d89260773e`.

The focused recovery run copied one real expert into a dedicated disposable
probe, staged and verified it, deleted only that probe's staging directory,
observed its absence, and reconstructed it. Canonical expert and manifest hashes
remained unchanged, and both staged copies matched the canonical SHA256.
The probe uses approximately 24 MiB for source-copy plus stage. This tests staging
recovery, not inference acceptance or recovery of an in-flight process.

## Real pretrained validation and fixture checks

Remote software is Python 3.12.3, PyTorch 2.5.1+cu124, Transformers 4.46.3, and
CUDA runtime 12.4. The RTX A5000 supports BF16 and has compute capability 8.6.
The visible memory limit is 64 GiB, swap limit is zero, CPU quota is
`2400000 100000` (24 cores), and affinity exposes 32 CPUs. Hidden ancestor limits
cannot be ruled out. The passing smoke included 35 passing remote fixtures with no
skips. After adding dependency-lock gate coverage, local Windows fixtures passed
28 of 36 tests with eight platform/PyTorch skips. Fixtures remain separate from
the following pretrained results.
Final remote fixture run `20261102T061153Z-fixtures-739ee745` passed all 36 tests
with zero skips after the environment replacement and was fetched locally.
Python compilation, PowerShell parsing, and Git whitespace checks also passed.

The successful resident reference saved nine logit checkpoints, forced real-text
token histories, scheduler events, software versions, and resource samples.
Reference inference peak PyTorch GPU allocation was 13,850,363,392 bytes
(12.899 GiB); sampled cgroup peak was 29,354,663,936 bytes (27.339 GiB).
Resident-model weak references were released; GPU allocation returned to its
8,519,680-byte pre-load baseline before proceeding. An earlier reference produced
real outputs but failed an overly strict cleanup assertion requiring less than
1 MiB total GPU allocation. Its failed status and logs are preserved separately.

The first full smoke exposed a recompute bug: replaying the whole saved prefix in
one batch differed from the original prefill/decode chunks in BF16. Returning
maximum absolute logit errors were `0.390625` and `0.25`, and one of nine argmax
values changed. The corrected implementation replays the original chunk sequence.
The required tolerances remain `atol=0.125`, `rtol=0.02`, minimum argmax agreement
`0.95`; no tolerance was relaxed. The failed run remains available for diagnosis.

Successful smoke `20261102T060152Z-smoke-a42a4b3d` used persistent JuiceFS for
correctness, four GPU expert slots (48 MiB), global LRU, and 64 host expert slots.
All 27 logit-vector comparisons across retain, spill/reload, and recompute were
exact, with 100% argmax agreement, including six returning-turn checkpoints.
KV restoration and budget assertions passed. Retain and spill each executed
2,209 real CUDA expert misses; recompute executed 2,885. Another session made two
token steps while its peer was paused in every case. The shared host pool peaked
at 808,324,608 bytes against an 808,848,896-byte limit. Each offloaded case peaked
at 1,015,793,152 bytes of PyTorch GPU allocation; total sampled cgroup peak was
30,359,113,728 bytes (28.274 GiB), including filesystem page cache.

Experiment execution took 48.53 seconds, separate from 23.71 seconds of artifact
preflight; total worker duration was 78.94 seconds. All 38 fetched files
(4,938,213 bytes) verified. **Pretrained acceptance is complete.**

## Pilot loss and surviving artifacts

Pilot `20261102T060402Z-pilot-35700330` launched only after the successful smoke
was fetched. It requested two repetitions of each host-budget/action case under
a 900-second experiment cap. Staging began at `06:04:28.134745Z`; the last durable
heartbeat was `06:05:04.195517Z`. The worker then disappeared. Its observed status
is **lost, exit_code=null**. There is no staging completion, experiment preflight
completion, reference output, or measured pilot result. All 16 surviving compact
files (67,653 bytes) were fetched and verified. The evidence does not establish
how much staging had finished when the process was lost.

Post-loss doctor `20261102T060739Z-doctor-a29a79f9` recorded these changes at
`06:07:41.779297Z`:

| Identity | Before | After |
| --- | --- | --- |
| Self PID namespace inode | `4026534176` | `4026534177` |
| Self mount namespace inode | `4026534175` | `4026534176` |
| PID 1 start ticks | `270117667` | `271336556` |
| `/tmp` pod root | `2e69c951-d208-4fea-aae5-fc2fd01a47fc` | `08c42093-2fed-4949-9ab5-104f13f14c99` |
| Task scratch and original sentinel | Present | Absent |
| Artifact path/inode and sentinel hash | Present | Unchanged |

Hostname and host boot ID stayed unchanged. These observations establish replacement
of the observed process/mount environment and scratch loss. They do not identify
the platform trigger. The new cgroup's zero OOM counters cannot explain the old
environment's termination. Canonical artifacts and durable logs survived on JuiceFS.
Setup `20261102T060858Z-setup-37dbb694` then verified every checkpoint and converted
file, completed with exit 0 in 89.95 seconds, and reused the environment and all
payloads without download/conversion. It preserved the exact completion hash
`154ef707...60773e`. The original artifact sentinel survived; setup explicitly
created a new scratch sentinel after recording its absence. Home still had
32.70 GiB available. There was no pilot retry. Stable compute is required for the
storage pilot; persistent files cannot keep an in-flight process alive.

## Local evidence and tested controller commands

The following directories under local `results/` contain compact logs, stage
records, source/model/config provenance, file-hash manifests, and verified transfer
receipts. Model weights and expert/KV backing files were not fetched.

| Local run directory | Observed result |
| --- | --- |
| `20261102T053807Z-setup-e3ca7579` | Persistent acquisition/conversion completed, exit 0 |
| `20261102T054356Z-setup-1d0c1dc3` | Verified payload reuse; completion schema updated |
| `20261102T055000Z-setup-51bc802f` | Exact completion-manifest reuse, exit 0 |
| `20261102T054613Z-recovery-2f181d51` | Dedicated-stage deletion/reconstruction passed |
| `20261102T054638Z-reference-d7a93547` | Real outputs saved; cleanup assertion failed |
| `20261102T055349Z-reference-5789fd68` | Resident reference and release checks passed |
| `20261102T055449Z-smoke-d159dd28` | Retain/spill passed; recompute failed |
| `20261102T060152Z-smoke-a42a4b3d` | All three pretrained KV actions passed, exit 0 |
| `20261102T060402Z-pilot-35700330` | Worker lost during staging; no experiment result |
| `20261102T060739Z-doctor-a29a79f9` | Post-loss process/mount audit, exit 0 |
| `20261102T060858Z-setup-37dbb694` | All persistent files reverified after replacement, exit 0 |
| `20261102T061153Z-fixtures-739ee745` | Final 36 remote fixtures passed, exit 0 |

Native Windows PowerShell exercised `deploy`, `doctor`, `setup`, `fixtures`,
`recovery`, `reference`, `smoke`, `pilot`, `status`, and `fetch`. Commands included:

```powershell
python -m unittest discover -s tests -v
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action setup
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action recovery -Wait
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action reference
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action smoke
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action pilot
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action doctor -Wait
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action fixtures -Wait
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action status -RunId <run-id>
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action fetch -RunId <run-id>
```

The Linux-clone helper and WSL path are documented but were not separately run.
Publishing/clone validation awaits user-reviewed commits and a configured remote.
The pilot gate also checks the exact dependency-lock hash, preventing a different
environment specification from reusing an older smoke acceptance.

`summaries/acceptance.json` is the selected compact record for review/Git. It was
generated with `python scripts/summarize.py` from nine hash-verified local run
directories and retains source/archive hashes, model/config/environment identity,
transfer receipts, setup/recovery/reference outcomes, and the lost pilot status.
The original `summaries/validation.json` remains as historical evidence. Final
documentation and selected summaries were written after measured runs; each run's
source manifest identifies exactly what executed.

The next local review command is `git status --short`, followed by `git diff` and
inspection of the new files. No commits, pushes, or remote configuration were done.
Before retrying storage measurements, obtain stable compute/provider lifecycle
information; on a new target edit only the ignored machine JSON, run doctor/setup/
smoke and fetch, then run `powershell -NoProfile -ExecutionPolicy Bypass -File
.\scripts\remote.ps1 -Action pilot`. Published-clone and WSL execution, a successful
full-size scratch staging/pilot, and controlled physical-SSD attribution remain
untested or blocked. Full canonical artifact corruption is detected but not
automatically repaired; interrupted preparation and disposable staging are recoverable.

## Limits on interpretation

`/home` and `/models` are JuiceFS. `/tmp` is **ext4; physical backing unverified**:
visible device-mapper information is insufficient to establish the full mapping.
Preparation, hashing, and staging warm filesystem caches and are outside measured
experiment timing. No global cache drop or raw-device access was performed.
Application logical reads, process I/O, and cgroup/device counters are separate;
an expert-cache miss does not establish an SSD read. Page cache is outside the
application tensor pool but inside cgroup usage. Sampled cgroup peaks are lower
bounds; kernel historical peaks may include preparation and previous runs.
GPU case peaks are PyTorch allocator peaks reset after loading, so transient model
loading peaks and CUDA context/driver allocations are outside those reported peaks.
I/O and H2D transfers are synchronous, timings may overlap and are not summed as
critical-path delay, and tiny samples do not support p99 claims. Persistent files
allow a new process to recover preparation; they cannot keep a process alive
through replacement or restart. The smaller OLMoE BF16 test does not validate
larger Qwen3 FP8 results.
