# TorchWasm 0.92 local qualification

The qualified source is `6d752c6bb1eda66dee4b36918dea0906991cbb03`, on the
`a3fd2b8b3836793f721cc03dc7918c584edfe6d8` main snapshot. It promotes the
[bounded cumulative-product workspace development](torch-cumprod-column-memory.md)
to version 0.92. Relative to that development snapshot, only the package version
constant changes among the 80 public resources; all 102 SDK modules retain their
hashes. The new package/application checks below execute the actual 0.92 source.
Historical latency/allocation observations retain their original baselines and
scope. The 64 KiB limit bounds the packed workspace, not output, RSS or WASM heap.
Production WebGPU remains disabled.

## Fresh package and application verification

- Exported all 92 current PaperLeet modules with matching 65-source autograd and
  6-source schedules profiles. All 17,476 preceding non-index data files retain
  their Git content identities; 186 new immutable files are added.
- Built six public/private wheel, source-distribution and rebuilt-wheel
  artifacts. Both target-only installed-package smokes pass, including unchanged
  `import torch`, the launcher, autograd, JVP/HVP and import isolation.
- Complete SDK from the installed public package: **14,659 passed** with NumPy
  2.5.4; **14,654 passed and five skipped** with NumPy 1.26.4. The skips are the
  same rank-construction limits as the preceding complete SDK. Ordinary child
  imports are separately preflighted from the installed target on both runtimes.
- Complete native application: **33,656 passed**, with the existing optional
  Docker submission-image test skipped. Public and private imports resolve to
  the recorded package source.
- Real Node/Pyodide: **92/92 modules**, **742 tests** and **421 mutants** pass.
  The deployed profiles also pass **14,632 independent cases in 100 SDK suites**.
  Standalone/CLI and optional GPU package suites are excluded from these deployed
  profiles. After learner execution, the SDK VM adds schedules-only files from
  the separately published profile and checks that shared files agree. The
  preceding full 80-source Node/Chrome SDK campaigns pass 14,643
  selected cases each; their identical numerical/test hashes are bound through
  the exact version-only bridge, rather than claimed as fresh 0.92 executions.
- All **254 registered frontend commands**, both builds, production metadata,
  immutable-asset retention and exact generated-Worker checks pass. Actual Chrome
  runs the nested attention page and passes all seven judge tests. Its appended
  probes exercise zero factors, JVP/HVP, nonfinite values, ownership and six
  wide/high-rank column scans through the actual new helper, including gradients
  and preserved caller storage. The served runtime has all 65 expected resources.

An initial export environment lacked the editable private-vendor mapping. An
initial installed-SDK harness also omitted the target path from child imports:
four minimum-environment import tests failed while numerical tests passed. Both
original invocations are retained. Correcting the host mapping and child import
path passes fresh verification; no library behavior was changed to mask errors.

## Retained browser timing observation

The complete concurrent verification campaign records one budget excess:
`diffusion/flow_matching` passes its eight tests and four mutants, but takes
266,662 ms for calibration and 329,763 ms overall, above its 240,000/300,000 ms
budgets. Its learner phase remains within 60,000 ms. This module does not call
`cumprod`; its original report remains unchanged.

After every other owned qualification process finishes, one fresh isolated
Node/Pyodide process repeats the same source, manifest and runtime. All eight
tests/four mutants pass; calibration is 135,748 ms, learner execution 24,359 ms
and total execution 163,087 ms, with no budget excess. This does not establish a
root cause, universal timing guarantee or whole-application speedup. Both raw
observations are retained. The earlier CPU/WASM kernel timing and allocation
records remain separate measured comparisons.

## Evidence and independent restoration

The [primary machine record](https://github.com/PaperLeet/paperleet.github.io/blob/4c0679e73f12f79ec154df66b56a40bb41b18491/packages/torchwasm/benchmarks/results/qualification-0.92-column-workspace.json)
binds complete frozen source inputs, package artifacts, raw logs/results and
preceding numerical observation archives. Its **16,910,840-byte** v1 archive
contains **3,665 logical files from 3,150 local blobs**, without omitted ancestor
archives or Git providers. SHA-256:
`168c4ddc788c471b8dc5cb175e53fc0d0a29e3865de9e96c19c526e3aa7775af`.
Third-party runtimes/dependencies and historical inactive product assets remain
separate. This is not a complete historical repository mirror.

The public integrity/current-source verifier passes. The
[restoration record](https://github.com/PaperLeet/paperleet.github.io/blob/4c0679e73f12f79ec154df66b56a40bb41b18491/packages/torchwasm/benchmarks/results/qualification-0.92-column-workspace-replay.json)
records full outside-checkout recovery and independent readback of all 3,665
files. The restored 1,306-case cumulative-product module plus 13 standalone,
import and CLI cases pass: **1,319** on modern NumPy and **1,317 plus two existing
rank skips** on minimum NumPy. First-party source and tests load only from the
restored tree. This is a focused portability check, not another full application
or WASM campaign. The separate 840,785-byte v1 replay archive contains 198 local
files/blobs; SHA-256:
`f7d08a62461b44a5feada797f5c19bc9bbdeb6eefd1a732246098c16ad2865e9`.

From `packages/torchwasm`, using fresh output directories:

```sh
python benchmarks/verify_qualification.py benchmarks/results/qualification-0.92-column-workspace.json --check-current-source
python benchmarks/restore_qualification.py benchmarks/results/qualification-0.92-column-workspace.json benchmarks/results/qualification-0.92-column-workspace-harness.tar.gz /tmp/torchwasm-092-proof
python -I -B /tmp/torchwasm-092-proof/torch-iteration-1149/replay_native.py /tmp/torchwasm-092-proof /tmp/torchwasm-092-replay
```

The replay needs NumPy and pytest in that Python environment. Append `2` on the
recorded NumPy 1.26.4 environment to assert its two rank-construction skips.
Full WASM/frontend replay needs the separately installed third-party runtimes
and remapping of the captured absolute harness paths. The package README was
captured while qualification was in progress; this record closes those local
gates. Updating its wording or optional test metadata requires fresh artifacts.

## Main integration has a new provenance boundary

During this campaign, main advanced to
`45471d6cd1f474e804bc4b5e61bb0f0cc4b025ee`, introducing a reviewed upstream
snapshot lock for `PaperLeet/stridetorch`. Its 185 managed files pin standalone
commit `7f158d09ea65d4cfae00500af0c81faa5f6ddddd`. A separate immutable main
snapshot passes the offline source check and all four provenance regressions.
Applying this candidate's four changed managed files correctly fails the source
check without overwriting bytes. That expected rejection is not an adoption gate.

This record qualifies the stated frozen main snapshot; it does not qualify the
newer provenance integration. Preserve its optional test dependency metadata,
establish a matching immutable standalone source revision, update the upstream
pin truthfully and qualify changed package/provenance/integration inputs. Do not
weaken or bypass the source lock. A fresh strict local pack and unchanged
release-chain/native Git push guards remain required for main adoption.

This candidate has not been adopted by main. No CI dispatch, tag, package
publication or deployment was initiated by this campaign. The independent
repository and other working checkouts remain untouched.
