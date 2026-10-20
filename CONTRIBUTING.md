# Contributing to TorchWasm

Runtime operators, differentiation, Torch API compatibility, CPU/WASM correctness
and performance work belong here. Paper exercises, teaching statements, learning
interfaces and product features belong in PaperLeet.

Install from this repository with `python -m pip install -e '.[test]'`, then run
`python -m pytest tests -q`. Native PyTorch is a test oracle, not a runtime
dependency. Run `npm ci` and `npm run test:wasm` for the independent Pyodide smoke
flow. It needs no PaperLeet checkout, problem catalog, account or cloud API.

Every supported operation needs tests for its declared shape/dtype and gradient
behavior, including relevant boundaries. Unsupported behavior must fail clearly.
Do not claim full PyTorch or production WebGPU support. Benchmark results should
retain workload, versions, source hashes and raw measurements, and distinguish
whole-flow results from individual kernels. Do not run timing benchmarks at the
same time as tests or other benchmarks.

Publish a reviewed version before updating PaperLeet's dependency. Product
integration tests live with the product and are not required to import this
library. See [the history record](docs/history.md) for original development records.
