# Column-major cumulative-product development

This is the historical development snapshot at `6eb494f37262d32190d833efa340e2e903f5c3b7`.
The [bounded-memory followup](torch-cumprod-column-memory.md) supersedes its
full selected temporary. Measurements and archive identities below remain
unchanged; current-source verification here refers to that frozen snapshot.

This is development after qualified TorchWasm 0.91 at
`13ae2bf7af07ac4f20a6cefd0425024eede5ac98`. Package metadata remains 0.91.0;
this source has not been adopted on main or published as a new release.

Large non-widened real `cumprod` scans can keep a column-major work array
when the input is Fortran contiguous, the scan uses the first axis, and the
number of independent scans is at least the scan length. Accumulation
remains in place; the result is then materialized as owned C-contiguous data.
Tall batches retain the 0.91 C-buffer route. Widened, complex and integer
paths retain their existing rules. Requested dtype conversion, initial
complex identity multiplication and input ownership remain intact.

Only `_product_scan` changes among reduction-helper ASTs. The other 79 public
source files and 101 other SDK modules retain their baseline hashes. This is
a layout choice, with an additional temporary for selected inference calls.
It does not resolve the previously recorded flat CPU-inference regression.

## Correctness and independent references

The complete native SDK passes 14375 tests on NumPy 2.5.4. NumPy 1.26.4
passes 14372 with three existing high-rank NumPy limitations skipped.
The complete affected 1022-case module passes without skips in independent
Node/Pyodide and a real Chrome Worker. Offline 80-file public exports,
65-source default exports and ordinary `import torch` alias smokes pass too.

The 366 new cases contain 300 large-matrix counted-monomial value/VJP/direct-JVP
oracles, 42 negative-stride NumPy-import rejection boundaries and 24 composed
HVP oracles on wide 2D/3D column-major arrays. C/F, strided, read-only and
broadcast layouts, float32-to-float64 requests, zero factors and both sides
of the near-square routing boundary are covered. Negative-stride imports
retain the established native-style rejection; they are not newly supported.

A separate 1312-case matrix preserves the previous 696 cases and adds 616
large-matrix/3D and requested-dtype cases. Native PyTorch 2.14 matches all
baseline and candidate outputs exactly after canonicalizing NaN payloads.
Other values and non-NaN signs remain part of each digest. Baseline/candidate
canonical outputs match on all 1312 cases in both CPU environments and in
independent Node and Chrome Pyodide runtimes.

Native PyTorch 2.6 itself raises a `masked_scatter` mixed-dtype RuntimeError
in 28 float32-input/float64-output, special-factor reverse cases. These cases
remain in the corpus: both minimum-environment implementations match their
independent PyTorch 2.14 canonical references exactly. The other 1284 cases
are compared directly with PyTorch 2.6. Its 100 complex128 differences remain
within the recorded 1e-10 relative tolerance and are unchanged by this patch.
No native-reference case is silently dropped or reported as a native 2.6 pass.

## Complete-flow observations

The final corpus has 284 cases with four observations per implementation and
environment, alternating ABBA/BAAB per case. CPU implementations run in
separate persistent processes; Node uses independent Pyodide runtimes. Tests,
allocation checks and timing runs execute separately. Each observation includes
public Tensor construction and inference or square/sum/backward. The original
132 controls remain, with a constant-element geometry grid, requested dtype
conversion, 3D-axis controls and 255/256/257-row routing-boundary cases added.
Finite unit factors and zero patterns isolate execution cost; these are warm
complete-library calls, not import, transfer or full-application throughput.

Each table entry is the median of per-case candidate/baseline latency ratios.
There are 30 selected inference and 30 selected reverse cases, plus 112
unchanged controls for each flow. A ratio below one is faster.

| Environment | Selected inference | Selected reverse | Control inference | Control reverse |
| --- | ---: | ---: | ---: | ---: |
| NumPy 2.5.4 | 0.583 | 0.837 | 1.006 | 1.002 |
| NumPy 1.26.4 | 0.532 | 0.816 | 0.991 | 1.003 |
| Node/Pyodide | 0.406 | 0.864 | 1.002 | 1.002 |

All final selected inference ratios are below one. The minimum-environment
reverse maximum is 1.00048, so universal speedup is not claimed. Controls retain
anomalies: maximum inference ratios are 1.373 on modern NumPy, 1.033 on minimum
NumPy and 1.415 on Node. Every point, sample range and raw observation is kept.
The final selector restores the old route for tall batches; it is not an
optimal-layout guarantee for every machine, shape or NumPy version.

Native `tracemalloc` measurements use 36 cases and three observations after
warmup in each environment, with a NumPy-buffer visibility control. Preexisting
matrix inputs are excluded; flat controls retain their input copy. For the
measured 65536-element selected float64 matrices, inference peak rises from
about 0.50 MiB to 1.00 MiB. Float32-to-float64 inference rises from about 1.00 MiB
to 1.50 MiB. Retained output bytes are unchanged, and tall-matrix inference peaks
remain identical. These are native traced peaks, not RSS or WASM heap peaks.
Bounded packing and reuse of owned dtype-conversion buffers remain useful
next steps before selecting the next version for main adoption.

## Evidence and retained development observations

The [machine record](https://github.com/PaperLeet/paperleet.github.io/blob/4c0679e73f12f79ec154df66b56a40bb41b18491/packages/torchwasm/benchmarks/results/cumprod-column-layout.json)
binds exact 80-file source and 102-module SDK identities, current test receipts,
native-version differences, all final measurements and earlier prototypes.
Its self-contained v1 archive contains 1571 logical files from 354 local blobs,
without external archive or Git providers. SHA-256:
`6492c13676c370d1c1212c7f31992c04963dffa517e4b51980f8002ed93614f9`.
The public verifier passes both integrity and current-source checks.

The first unrestricted CPU-direct route made several large calls slower.
Unrestricted F-buffer routing also regressed tall inference; both experiments
are retained. An initial test fixture incorrectly expected negative-stride
NumPy import to succeed; its 42 import failures are preserved and the contract
is now tested as rejection. Reference-harness output-Path shadowing and a
Python-tuple/JSON-list identity error are preserved with their corrected
harnesses and successful complete results. These failed invocations are not
reported as aggregate passes.

The archive restores frozen first-party sources, tests, harnesses and raw
observations without repository history. Third-party Python/NumPy/pytest,
native Torch, Pyodide and Chrome runtimes are separate dependencies. Original
absolute paths remain historical metadata. The archived
`torch-iteration-1115/replay_restored.py` verifies all restored SDK-file hashes
and runs the 1022-case affected module using restored source paths.

The [restoration record](https://github.com/PaperLeet/paperleet.github.io/blob/4c0679e73f12f79ec154df66b56a40bb41b18491/packages/torchwasm/benchmarks/results/cumprod-column-layout-replay.json)
records actual outside-checkout recovery and read-back hashes for all 1571
logical files. The restored affected module passes all 1022 cases on both
NumPy environments, with no skips. Its 527557-byte v1 replay archive retains
193 first-party snapshot/evidence files with no external providers. This is
an independent focused replay, not another full SDK/application campaign.
The replay helper's `original_capture_paths_used: false` refers specifically
to first-party source/test inputs; third-party interpreter environments are
reused. See the [benchmark README](../../packages/torchwasm/benchmarks/README.md)
for recovery and replay commands.

Fresh versioning, rebuilt public/private packages, full WASM SDK/application
runs, immutable exports and current-main frontend/build/actual integration
Worker qualification remain required for the next adoption. No CI dispatch,
tag, publication, deployment or main push was performed for this development.
