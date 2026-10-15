"""Package only runtime files and check an isolated extracted submission."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
NAMES = ['run.py', 'rag.py', 'grounding.py', 'nova_sdk.py', 'requirements.txt', 'SOURCES.md']
target = ROOT / 'submission.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in NAMES:
        path = ROOT / name
        if path.is_symlink():
            raise ValueError('Symlinks are not allowed')
        if name.endswith('.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=name)
        archive.write(path, name)
with tempfile.TemporaryDirectory(prefix='nova-v2-') as tmp:
    tmp = Path(tmp)
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == NAMES
        assert archive.testzip() is None
        archive.extractall(tmp / 'submission')
    guidelines = tmp / 'guidelines' / 'fixture'
    guidelines.mkdir(parents=True)
    (guidelines / 'example.md').write_text(
        '---\ntitle: "Synthetic retrieval check"\n---\n# Overview\n'
        'diplopia ptosis weakness synthetic reference, not patient data.', encoding='utf-8')
    script = '''import os, sys
sys.path.insert(0, sys.argv[1])
os.environ['NOVA_GUIDELINES_DIR'] = sys.argv[2]
import run, rag, grounding, nova_sdk
assert run.__file__.startswith(sys.argv[1])
state = run.prepare()
assert state['rag'] is not None
assert state['rag'].search('diplopia ptosis')
state['rag'].close()
print('Isolated import / prepare / SQLite FTS5 retrieval: PASS')
'''
    subprocess.run([sys.executable, '-I', '-c', script, str(tmp / 'submission'),
                    str(guidelines.parent)], check=True, timeout=60)
manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in NAMES}
manifest['submission.zip'] = hashlib.sha256(target.read_bytes()).hexdigest()
(ROOT / 'SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(f'ZIP: {target.name}; {len(NAMES)} files; {target.stat().st_size} bytes')
