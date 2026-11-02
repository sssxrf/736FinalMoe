# Validation record — November 1, 2026 (America/Chicago)

**Implementation and tiny CUDA validation are complete. Real pretrained acceptance
is blocked, and the pilot was not run.** No MoE-serving performance result or SSD
conclusion is claimed.

All source originated in the local Windows checkout. The starting Git commit was
`4d7add9` (`Initial commit`), the checkout was clean, and no remote was configured.
No commit or push was performed. The changes remain available for review.

## Checks actually run

| Stage | Result |
| --- | --- |
| Native Windows PowerShell controller | Deploy, doctor, setup, fixtures, status and fetch exercised |
| Local fixture suite | 19 tests: 15 passed; 4 skipped because Linux locking or PyTorch/CUDA is required |
| Python compilation / PowerShell parsing / Git whitespace check | Passed |
| Remote dependency installation | Python 3.12.3; PyTorch 2.5.1+cu124; Transformers 4.46.3; `pip check` passed |
| Remote CUDA / BF16 audit | RTX A5000, compute capability 8.6; CUDA/BF16 available |
| Pinned real model acquisition | Nine files downloaded and verified; exact configured revision |
| Expert preparation | 1,024 addressable real BF16 experts prepared; 12 MiB each |
| Remote tiny fixtures | 19 passed, zero skipped; explicit exit code 0 |
| Real pretrained resident reference | Not reached: prepared `/tmp` data disappeared before smoke |
| Real pretrained expert/KV acceptance | Not completed |
| Pilot and performance measurements | Not run |

The CUDA fixture covers actual expert-file misses, single-slot GPU eviction,
CUDA execution, exact dense/expert/buffer equality, retain, spill/reload, recompute,
spill corruption rejection, and pool cleanup. Another fixture runs the full token
scheduler with two sessions, returning turns, nine fixed checkpoints per case,
and all three KV actions. These fixtures use a tiny randomly initialized OLMoE;
they are explicitly separate from pretrained acceptance.

An initial fixture failed because its resident construction cast nonpersistent
RoPE buffers to BF16. The fixture now loads its saved checkpoint through the same
`from_pretrained` BF16/eager/CUDA path as the real reference, preserving FP32 RoPE.
Exact parameter/buffer and direct-logit checks pass without relaxing their tolerance.

## Execution-target blocker

The first setup worker disappeared during download. A subsequent audit found an
empty `/tmp`, a different process namespace, and surviving `/home` files. The
controller reported `lost` with `exit_code: null`.

The second setup completed successfully, including model verification and expert
conversion. Before smoke began, the target changed again and its `/tmp` contents
were gone. Smoke exited 1 with a missing
`/tmp/cs736-rxu292-moe/model/model-provenance.json` traceback. The final implementation
now reports this condition explicitly in a failed preflight summary.

The exact infrastructure cause is unknown. A hidden ephemeral-storage limit or
portal eviction is a hypothesis, not an established fact. The successful setup
audit recorded **25.78 GiB task scratch**, **80.18 GiB filesystem free**, and a
**27.04 GiB historical cgroup memory peak** against the visible **64 GiB limit**.
Visible OOM counters were zero. Filesystem headroom checks therefore passed, but
they cannot guarantee workspace lifetime or reveal all orchestration quotas.

The visible CPU quota is `2400000 100000` (24 cores), with a 32-CPU affinity mask.
Accessible swap is zero; host ancestor limits beyond the cgroup namespace are
unknown. `/tmp` is **ext4; physical backing unverified**. A device-mapper chain
points toward an NVMe partition, but the full physical mapping was not verified.
The account home and `/models` are JuiceFS.

The final tiny fixture report records **8.21 MiB peak PyTorch GPU allocation** and
**22 MiB peak reserved allocation**. Those values exclude CUDA context/driver
allocations and do not describe the real pretrained model. No real-model GPU peak,
TTFT, TPOT, or expert/KV I/O measurements are available.

## Local evidence

Full reports are ignored runtime data under `results/`. Every listed directory
contains a file-hash manifest and verified transfer receipt:

- `results/20261102T022947Z-doctor-786f27b3/`: initial machine audit.
- `results/20261102T023003Z-setup-17eea790/`: first setup and observed lost-worker status.
- `results/20261102T023925Z-setup-6b8a78ac/`: successful setup, model provenance, before/after audits.
- `results/20261102T024300Z-smoke-f1470994/`: failed pretrained smoke traceback and source provenance.
- `results/20261102T025113Z-fixtures-c660141f/`: final 19 passing remote tests and resource report.

`summaries/validation.json` is a deliberately selected compact record suitable for
Git. It includes source hashes, release archive hashes, pinned-model information,
and transfer receipts. Each run's source manifest identifies the exact tested
uncommitted snapshot. Documentation and the selected summary were added afterward.

## Resume validation

First inspect EFabric portal eviction/restart events and confirm that a target can
retain at least 26 GiB of task scratch long enough to run the workload. Alternatively,
use a stable replacement target by changing only the ignored machine JSON. No code
change or remote IDE is required.

Then run from the local PowerShell checkout:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action setup -Wait
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action smoke -Wait
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action fetch
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action pilot -Wait
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\remote.ps1 -Action fetch
```

Smoke must report complete/passed, exercise real expert misses and all three KV
actions, and be fetched locally before treating this starter as fully validated.
The 900-second pilot cap remains in effect. The pilot defaults to one repetition
per host-budget/action case; increase repetitions only after observing its cost.

The direct Linux-clone helper and WSL invocation are documented but not separately
executed in this session. Published-clone validation is pending because this
repository has no configured remote and has not been committed or pushed.
