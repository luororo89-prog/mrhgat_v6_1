from pathlib import Path
import json
from pypdf import PdfReader

root = Path(r'C:\Users\lx554\WPS Cloud\WPSDrive\1817177196\WPS企业云盘\北京大学\我的企业文档\博后三胎\文献下载')
out = Path(__file__).parent / 'method_sources'
out.mkdir(exist_ok=True)
names = ['7_ResiliNet_npj.pdf','5_DisenGCN_Ma_pmlr.pdf','8_IPGDN_Liu_arxiv.pdf','9_GATv2_Brody_arxiv.pdf','10_FactorizableGCN_Yang_arxiv.pdf','1_DirGNN_Rossi_arxiv.pdf']
records=[]
for name in names:
    path=next(root.rglob(name))
    reader=PdfReader(path)
    texts=[]
    links=[]
    for i,page in enumerate(reader.pages,1):
        texts.append(f'\n=== PDF PAGE {i} ===\n'+(page.extract_text() or ''))
        for annot in page.get('/Annots',[]):
            a=annot.get_object().get('/A',{})
            if '/URI' in a:
                links.append(str(a['/URI']))
    (out/(path.stem+'.txt')).write_text('\n'.join(texts),encoding='utf-8')
    item=dict(path=str(path),pages=len(reader.pages),characters=sum(map(len,texts)),links=sorted(set(links)))
    records.append(item)
    print(json.dumps(item,ensure_ascii=False))
(out/'pdf_inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
