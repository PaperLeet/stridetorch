# Bounded cumulative-product workspace development

This is the historical 0.91-metadata development snapshot at `be8da23694e244c6833d5157752c45e702b974fe`.
The [0.92 qualification](torch-runtime-0.92-qualification.md) binds the same
numerical/test source after its version-only change and adds fresh package,
application and browser integration checks. Current-source verification below
refers to the captured development tree; use the 0.92 record for current source.

This is development after `6eb494f37262d32190d833efa340e2e903f5c3b7`.
The qualified main baseline is `13ae2bf7af07ac4f20a6cefd0425024eede5ac98`.
Package metadata remains 0.91.0; no next-version main adoption or release is
claimed. This reduces inference allocation while retaining the wide-layout
benefit relative to main. Some calls become slower than the preceding,
unbounded column-buffer development; the measurements below retain that cost.

## Implementation and boundaries

Eligible large real column-major scans pack independent first-axis prefixes
into a fresh working array capped at **64 KiB**. A prefix must fit the budget,
the first axis must be contiguous, and the independent-column count must be
at least the scan length. The implementation slices a retained dimension and
iterates remaining coordinates, accumulating each packed array in place and
copying it into the owned C-contiguous output. Partial final tiles and
higher-dimensional positions preserve their logical column identity.

Requested real input conversion happens directly into each packed tile.
There is no full converted input or full F-contiguous output temporary on
this route. Other eligible scans can reuse an already owned `astype` result
when its dtype and layout match the accumulation buffer. Caller arrays and
aliases remain unchanged. Complex-to-real conversion retains its single
precast warning and its existing arithmetic; it does not perform per-tile
complex conversion. Complex-output and widened-output rules retain their
previous behavior. A common empty coordinate tail avoids NumPy iterator setup.

The cap applies to the new packed workspace. It is **not** a bound on the
output, total operation memory, all conversion paths, RSS or WASM heap use.
F-like strided inputs that first materialize a converted array can reuse it,
but that conversion still scales with input size. A scan too long for one
work tile retains the existing C-buffer route. This does not remove the
previously documented flat float64 CPU-inference cost.

Only `_product_scan` changes among existing reduction functions; the new
private `_product_column_scan` and budget constant implement the packer.
The other 79 public resources and 101 other SDK modules retain parent hashes.
Ordinary `import torch`, the launcher, export profiles and public API names
retain their existing behavior. Production WebGPU remains disabled.

## Numerical and runtime validation

- Complete native SDK: **14,659 passed** on NumPy 2.5.4; **14,654 passed and
  5 skipped** on NumPy 1.26.4. The skips are its three prior high-rank limits
  and the two added 64-dimensional input cases it cannot construct.
- Complete WASM SDK: **14,643/14,643 passed**, zero skips, in both independent
  Node/Pyodide and a real Chrome Worker. The same 13 host-only standalone/CLI
  cases and 3 subprocess cases remain outside WASM; native runs include them.
- The affected module contains **1,306 cases**. Its 284 added cases cover
  224 distinct column tags/casts/layouts/ownership, 36 working-budget controls,
  6 high-rank cases, 12 saved-storage guards and 6 complex-cast value/warning
  cases. Odd sizes, non-divisible tiles, remaining coordinate dimensions,
  C/F/F-strided/read-only layouts and odd integers above `2**53` are included.
  Scalar Python conversion and unit-sign monomials provide independent oracles.
- Direct modern-native PyTorch inference independently matches all 224 added
  tagged-column cases, including finite values and zero signs. All 1,312
  existing expanded reference cases preserve their complete canonical outputs
  relative to the development parent in both native and both WASM hosts.
  PyTorch 2.14 matches all modern cases exactly after NaN canonicalization.
- PyTorch 2.6 retains its 28 mixed-dtype special-factor reverse exceptions;
  these are matched exactly to a newly generated independent PyTorch 2.14
  reference. Its other 1,284 cases are checked directly. The 100 complex128
  differences within the recorded `1e-10` tolerance remain unchanged.
- Offline 80-file public exports, 65-source defaults and ordinary aliases pass.
  Full-suite execution and native-reference checks remain separate evidence.

## Native allocation

Forty flows per native environment have three traced-allocation observations
after warmup. A control confirms NumPy data buffers are visible. Matrix inputs
preexist the measurement; flat controls retain an input copy. Values/gradients
are checked for every observed allocation flow.

For the measured 65,536-element packed matrices, float64 inference peak falls
from about **1.00 MiB to 0.565 MiB**, and float32-to-float64 from about
**1.50 MiB to 0.565 MiB**, in both environments. Compatible C-order requested
conversion falls from about **1.00 MiB to 0.502 MiB**. Tall F-order controls
keep their previous peak. The result storage remains the same size.

At 1,048,576 elements, the measured F-order float64 peak falls from about
**16 MiB to 8.06 MiB**; float32-to-float64 falls from about **24 MiB to
8.06 MiB**. C-order conversion falls from about **16 MiB to 8.00 MiB**.
These are complete native API inference traced peaks, with preexisting input
excluded. They are not whole-model, native-allocator-total or WASM memory claims.

## Complete-flow latency

Each environment uses four per-case observations for each implementation,
alternating ABBA/BAAB. Native implementations run in separate persistent
processes; Node uses independent Pyodide runtimes. Tests, allocation checks
and timing runs execute separately. Timing includes public Tensor construction
and inference or square/sum/backward. Matrix inputs preserve their layouts;
the original flat fixtures retain their input copy. Calibration targets about
30 ms; raw iteration counts, every observation and sample ranges are retained.
Warm unit-factor/zero workloads do not establish full-application throughput.

The following entries are medians of per-case candidate/baseline latency
ratios. A value below one is faster. Keep the two comparisons separate: their
shape distributions and weights differ, so their medians are not interchangeable.

Compared with the **preceding development** commit, the corpus has 372 cases:
54 packed inference, 50 packed reverse, 18 C-cast inference, 16 C-cast reverse,
and 118/116 respective controls.

| Environment | Packed inference | Packed reverse | C cast inference | C cast reverse | Control inference | Control reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NumPy 2.5.4 | 0.981 | 1.000 | 0.934 | 0.994 | 1.002 | 0.999 |
| NumPy 1.26.4 | 1.002 | 1.008 | 0.944 | 0.980 | 1.014 | 1.008 |
| Node/Pyodide | 1.105 | 1.026 | 0.966 | 0.997 | 1.005 | 1.004 |

The packed CPU groups are near parity and compatible requested C conversion
improves in these medians. Packed WASM inference costs about **10.5%** more
than the unbounded preceding buffer, with a 2.6% reverse median cost. Individual
regressions remain: packed-inference maxima are 1.184/1.199/1.423 for modern,
minimum and Node respectively. Tiny 16-element float64 inference has 1.035,
1.052 and 1.014 median ratios. No universal speedup is claimed.

Compared directly with **qualified main 0.91**, a separate 120-case corpus
has 28 packed inference, 24 packed reverse, 16/14 C-cast and 20/18 controls.

| Environment | Packed inference | Packed reverse | C cast inference | C cast reverse | Control inference | Control reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NumPy 2.5.4 | 0.574 | 0.851 | 0.954 | 0.986 | 1.008 | 1.003 |
| NumPy 1.26.4 | 0.537 | 0.841 | 0.948 | 0.969 | 0.994 | 1.004 |
| Node/Pyodide | 0.726 | 0.915 | 0.963 | 0.992 | 0.993 | 1.000 |

The selected packed group retains a main-relative latency benefit in these
medians. Node packed-inference/reverse maxima are 1.020/1.030, and controls
also retain outliers. These results do not prove an optimal route for every
shape, machine, NumPy version or allocator history.

An initial million-element float64 C-order Node control measured a 1.328
candidate/development ratio, although its accumulation route is unchanged.
Three additional fresh-process A0/A1/B diagnostics use identical parent bytes
in A0/A1, the candidate in B, 24 inference cases and four observations per
variant. Identical-baseline large-C float64 ratios span 0.875–1.123; B/A0 spans
0.858–1.070 and B/A1 spans 0.898–1.137. The original 1.328 observation is
retained. Pointer alignments and heap capacities are recorded, but they do
not prove a root cause. The diagnostic does not replace the original corpus.

## Evidence, restoration and qualification scope

The [machine record](https://github.com/PaperLeet/paperleet.github.io/blob/4c0679e73f12f79ec154df66b56a40bb41b18491/packages/torchwasm/benchmarks/results/cumprod-column-memory.json)
binds exact source/test identities, all validation receipts, both latency
corpora, native allocation, all A/A/B controls and preliminary 64/256 KiB
experiments. Its v1 archive contains 1,174 logical files from 329 local blobs,
without external archive or Git providers. SHA-256:
`df2c73df423b83ea17b10c799f96478a97fb83dcd62af9d7275e25b0509b7bae`.
The public integrity/current-source verifier passes. The exact first-party
WASM package loader and 17 pinned third-party runtime file fingerprints are
included; third-party binaries/interpreter environments are separate dependencies.

Earlier fixture failures are preserved: NumPy's high-rank `.flat` restriction
is avoided in the reference calculation by flattening it explicitly, and
output mutation is performed through a detached alias, retaining the established
non-leaf write boundary while checking shared-storage guards.

The earlier interpreter environments lost their dependency directories during
this work; no cause is inferred. Final validation and timing use newly created,
pinned Python/NumPy/PyTorch/pytest environments with recorded package manifests.
A first complete Chrome attempt stopped on host ENOSPC while writing progress,
and has no complete verdict. After reversible removal of an ignored generated
web build, an unchanged full selection passes with buffered console capture.
Tracked status, source, tests and measurement bytes remain intact. Recovery of
that unrelated generated UI build remains local and is outside the numerical
proof archive's restoration scope.

The [restoration record](https://github.com/PaperLeet/paperleet.github.io/blob/4c0679e73f12f79ec154df66b56a40bb41b18491/packages/torchwasm/benchmarks/results/cumprod-column-memory-replay.json)
records actual outside-checkout recovery and matching read-back hashes for
all 1,174 primary logical files. The restored affected module passes 1,306
cases on modern NumPy and 1,304 with its two rank-construction skips on the
minimum environment. Its separate 523,239-byte v1 archive retains all 193
replay/snapshot files without external providers. This focused replay is not
a second full SDK/application campaign. First-party imports use only the
restored tree; third-party Python/NumPy/pytest remain separate dependencies.
See the [benchmark README](../../packages/torchwasm/benchmarks/README.md) for
verification, recovery and focused replay commands.

A next version, rebuilt public/private packages, fresh PaperLeet application
and immutable-export checks, current-main frontend/build/actual integration
Worker checks and main adoption remain pending. No CI dispatch, tag, package
publication, deployment or main push was performed for this development.
