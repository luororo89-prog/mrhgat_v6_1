from pathlib import Path
import shutil
import hashlib
import json

source=Path(r'D:\多功能网络构建\26上实验\mrhgat_v6_1')
target=Path(__file__).resolve().parent
manifest=[]
def copy(path):
    relative=path.relative_to(source)
    destination=target/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,destination)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    after=hashlib.sha256(destination.read_bytes()).hexdigest()
    assert before==after
    manifest.append({'file':str(relative),'sha256':after})

for folder in ('src','tests','configs','data'):
    for path in (source/folder).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            copy(path)
for folder in ('reports/method_sources','results/runs/20260928_md_stability'):
    for path in (source/folder).rglob('*'):
        if path.is_file():
            copy(path)
for name in ('D6_literature_method_basis.md','D6_cause_diagnosis.md','D6_cause_diagnosis.json',
             'extract_method_papers.py','fetch_method_code.py','diagnose_recovery.py','执行口径.md'):
    copy(source/'reports'/name)
reference=source/'results/runs/20260928_212256_D6_necessary_gate'
for name in ('input_graph.npz','B_reference.npz','input_manifest.json','code_version.json','config.yaml'):
    if (reference/name).exists():
        copy(reference/name)
copy(source/'requirements-lock.txt')
(target/'reports/location_manifest.json').write_text(json.dumps({'source':str(source),'destination':str(target),'verified_files':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified copied files',len(manifest))
