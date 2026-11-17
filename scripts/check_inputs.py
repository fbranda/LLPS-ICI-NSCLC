#!/usr/bin/env python3
from pathlib import Path
import hashlib, sys
root=Path(__file__).resolve().parents[1]
errors=[]
manifest=root/'SHA256_INPUTS.txt'
for line in manifest.read_text().splitlines():
    if not line.strip() or line.startswith('#'):continue
    expected,rel=line.split('  ',1)
    f=root/rel
    if not f.is_file():
        errors.append(f'Missing: {rel}')
        continue
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    if h.hexdigest()!=expected:
        errors.append(f'Checksum mismatch: {rel}')
if errors:
    print('INPUT AUDIT FAILED')
    for x in errors:print(x)
    sys.exit(1)
print('INPUT AUDIT PASSED: original GEO/PhaSePro source files match SHA-256 manifest.')
