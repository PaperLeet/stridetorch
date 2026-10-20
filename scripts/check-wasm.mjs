// Independent real Python/NumPy WASM integration; no PaperLeet source or services.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {loadPyodide} from 'pyodide';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = path.join(root, 'node_modules/pyodide');
const cache = path.join(root, '.wasm');
await fs.mkdir(cache, {recursive: true});
const lock = JSON.parse(await fs.readFile(path.join(runtime, 'pyodide-lock.json'), 'utf8'));
const packages = new Set();
async function prepare(name) {
  if (packages.has(name)) return;
  packages.add(name);
  const entry = lock.packages[name];
  assert(entry, `Missing pinned WASM dependency: ${name}`);
  for (const dependency of entry.depends) await prepare(dependency);
  const target = path.join(cache, entry.file_name);
  let bytes;
  try { bytes = await fs.readFile(target); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  if (!bytes || createHash('sha256').update(bytes).digest('hex') !== entry.sha256) {
    const response = await fetch(`https://cdn.jsdelivr.net/pyodide/v0.29.4/full/${entry.file_name}`,
      {signal: AbortSignal.timeout(120000)});
    assert(response.ok, `WASM package download failed: ${response.status}`);
    bytes = Buffer.from(await response.arrayBuffer());
    assert.equal(createHash('sha256').update(bytes).digest('hex'), entry.sha256);
    await fs.writeFile(target, bytes);
  }
}
await prepare('numpy');
const py = await loadPyodide({indexURL: runtime, packageCacheDir: cache});
await py.loadPackage('numpy');
async function mount(directory, target) {
  py.FS.mkdirTree(target);
  for (const entry of await fs.readdir(directory, {withFileTypes: true})) {
    if (entry.name === '__pycache__') continue;
    const source = path.join(directory, entry.name), destination = target + '/' + entry.name;
    if (entry.isDirectory()) await mount(source, destination);
    else py.FS.writeFile(destination, new Uint8Array(await fs.readFile(source)));
  }
}
await mount(path.join(root, 'src/stridetorch'), '/library/stridetorch');
await py.runPythonAsync(`
import sys
sys.path.insert(0, '/library')
import numpy as np
from stridetorch import as_torch
assert 'paperleet' not in sys.modules
with as_torch():
    import torch
    x = torch.tensor([2., 3.], requires_grad=True)
    x.square().sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [4., 6.])
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    inputs = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    targets = torch.tensor([[0.], [1.], [2.], [3.]])
    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        loss = (model(inputs) - targets).square().mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.1, losses
assert 'torch' not in sys.modules
`);
console.log(`PASS standalone WASM: Pyodide ${py.version}, NumPy, gradient correctness, real Adam training and import isolation`);
