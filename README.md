# TorchWasm

A lightweight NumPy-backed tensor and automatic differentiation library with a
bounded PyTorch-compatible API. Run Python learning code in CPython or Pyodide
without installing native PyTorch in the runtime.

This draft adopts the **TorchWasm 0.91** package and launcher name. The GitHub
repository currently retains its existing `PaperLeet/stridetorch` identity.
This branch prepares the locally qualified 0.92 numerical source, including
bounded column-scan workspace and owned conversion reuse. Product integration
and its immutable upstream pin are verified separately. No release, package
publication or deployment is created by this source candidate.
See [source provenance](docs/source-snapshot.json) and the
[0.92 product qualification](docs/validation/torch-runtime-0.92-qualification.md).

TorchWasm grew out of [PaperLeet](https://github.com/Hongyang-Du/PaperLeet).
It now has an independent source tree, tests and release workflow. It contains
no problem catalog, product UI, account service or user data.

## Install from source

```sh
git clone https://github.com/PaperLeet/stridetorch.git
cd stridetorch
python -m pip install .
torchwasm train.py
```

The launcher lets supported existing `import torch` code run against TorchWasm.
For explicit integration:

```python
from torchwasm import as_torch

with as_torch():
    import torch
    x = torch.tensor([2., 3.], requires_grad=True)
    x.square().sum().backward()
    print(x.grad.tolist())  # [4.0, 6.0]
```

Ordinary `import torchwasm` does not replace installed Torch modules. Alias
activation is process-wide while active; use it before importing user code in
an isolated process or Worker. Supported APIs and dtype/gradient restrictions
are described in the [API reference](docs/API.md).

## Python and WASM checks

```sh
python -m pip install -e '.[test]'
python -m pytest tests -q
python -m pip wheel --no-deps . --wheel-dir dist
npm ci
npm run test:wasm
```

The test extra installs native PyTorch as an independent oracle. The library
itself needs NumPy only. The WASM check runs real Pyodide/NumPy, mounts this
repository's source, checks gradients and trains a small model with Adam. Its
runtime version and downloaded NumPy dependencies are pinned and hash checked.
No PaperLeet checkout or service is needed.

For a browser host, install the pure-Python wheel with NumPy in a compatible
Pyodide environment, or embed the modules returned by
`torchwasm.runtime.source_files()`. The host owns the Worker lifecycle and
any decision to rerun unsupported work elsewhere.

## Local 0.92 qualification

The independent checkout passes all 14,659 SDK cases on modern NumPy and
14,654 with five rank-construction skips on NumPy 1.26. Its real Pyodide flow
checks import isolation, gradients, zero-safe scans, JVP/HVP, the odd-tile
column route and Adam training without a PaperLeet runtime or catalogue.
The [machine record](docs/validation/qualification-0.92-independent.json)
includes an independently hash-verifiable source/SDK and raw-receipt archive.

```sh
python scripts/check-evidence.py docs/validation/qualification-0.92-independent.json --check-current-source
```

Product integration and publication remain separate steps; this candidate
creates neither a package release nor a deployment.

## Scope and contributions

This is a bounded compatibility implementation, not a complete PyTorch port.
Check declared shape, dtype and derivative support. WebGPU modules are opt-in
experiments and are not the default production backend. Historical performance
claims have their measured scope in [validation records](docs/validation).

Operator, differentiation and runtime changes belong here. Paper exercises and
learning-product changes belong in PaperLeet. See [contribution guidance](CONTRIBUTING.md),
[runtime roadmap](ROADMAP.md) and [benchmark tools](benchmarks/README.md).
PaperLeet adopts reviewed versions separately from its product releases.

## History and license

The relevant development history was extracted from the existing public
repository. Original GitHub PRs remain there, and the [history record](docs/history.md)
links the source snapshot and original-to-extracted commit map. Large historical
benchmark archives remain in the original repository and are linked from their
records rather than duplicated here.

Apache-2.0; preserve [LICENSE](LICENSE) and [NOTICE](NOTICE).
