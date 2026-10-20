"""Check a standalone qualification archive and optionally this checkout's SDK."""
from pathlib import Path
import argparse,hashlib,importlib.util,json
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('record',type=Path)
parser.add_argument('--check-current-source',action='store_true')
args=parser.parse_args()
spec=importlib.util.spec_from_file_location('torchwasm_evidence',root/'benchmarks/verify_qualification.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.verify(args.record.resolve(),root,False)
if args.check_current_source:
    record=json.loads(args.record.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    source=root/'src/torchwasm'
    assert {str(p.relative_to(source)):sha(p)for p in source.rglob('*')if p.is_file()and '__pycache__'not in p.parts}==record['public_files']
    assert {p.name:sha(p)for p in(root/'tests').glob('*.py')}==record['sdk_files']
    result['current_source_checked']=True
print(json.dumps(result,indent=2))
