from pathlib import Path
import urllib.request
import json
import re
import io
import zipfile
import hashlib
from concurrent.futures import ThreadPoolExecutor

OUT=Path(__file__).parent/'method_sources'/'official_code'
OUT.mkdir(parents=True,exist_ok=True)
def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research-method-audit'}),timeout=40).read()
page=fetch('https://yanbeiliu.github.io/publications/').decode()
at=page.lower().index('independence promoted')
print('IPGDN_AUTHOR_LINKS',page[at:at+950])
repos=['ihollywhy/FactorGCN.PyTorch','emalgorithm/directed-graph-neural-network','tech-srl/how_attentive_are_gats']
links=re.findall(r'href="(https://github.com/[^"]+)"',page[at:at+950])
if links:
    repos.append('/'.join(links[0].split('/')[3:5]))
def repo_files(repo):
    meta=json.loads(fetch('https://api.github.com/repos/'+repo))
    tree=json.loads(fetch('https://api.github.com/repos/'+repo+'/git/trees/'+meta['default_branch']+'?recursive=1'))
    dest=OUT/repo.replace('/','__')
    dest.mkdir(exist_ok=True)
    (dest/'tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
    items=[]
    for t in tree['tree']:
        p=t['path']
        if t['type']=='blob' and (p.endswith(('.py','.md','.txt')) or p=='LICENSE') and t.get('size',0)<150000 and not any(x in p.lower() for x in ['dataset/','data/','ogb/']):
            url='https://raw.githubusercontent.com/'+repo+'/'+tree['sha']+'/'+p
            data=fetch(url)
            target=dest/p
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
            items.append({'path':p,'url':url,'sha256':hashlib.sha256(data).hexdigest()})
    return {'repo':repo,'commit':tree['sha'],'files':items}
records=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for r in pool.map(repo_files,repos):
        records.append(r)
        print(r['repo'],r['commit'],[f['path'] for f in r['files']],flush=True)
url='https://raw.githubusercontent.com/JianxinMa/jianxinma.github.io/master/assets/DisenGCN-py3.zip'
data=fetch(url)
z=zipfile.ZipFile(io.BytesIO(data))
items=[]
for name in z.namelist():
    if name.endswith(('.py','.md','.txt')) and not name.startswith('/') and '..' not in Path(name).parts:
        target=OUT/'DisenGCN'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        content=z.read(name)
        target.write_bytes(content)
        items.append({'path':name,'sha256':hashlib.sha256(content).hexdigest()})
records.append({'repo':'DisenGCN_author_archive','url':url,'sha256':hashlib.sha256(data).hexdigest(),'files':items})
(OUT/'source_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('DisenGCN', [i['path'] for i in items],flush=True)
