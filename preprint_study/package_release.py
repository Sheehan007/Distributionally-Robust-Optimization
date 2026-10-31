"""Build public, data-free code and TeX archives from the explicit inventory."""
from pathlib import Path
import hashlib,json,shutil,zipfile
from pypdf import PdfReader
from public_inventory import public_files,paper_files
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'

def write_zip(path,entries):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for local,relative in sorted(entries,key=lambda x:str(x[1])):archive.write(local,str(relative))
    with zipfile.ZipFile(path) as archive:assert archive.testzip() is None

def main():
    OUT.mkdir(exist_ok=True);(OUT/'pdf').mkdir(exist_ok=True)
    pdf=OUT/'pdf'/'DRO_Review_Preprint.pdf';shutil.copy2(ROOT/'paper/main.pdf',pdf)
    pages=len(PdfReader(pdf).pages)
    abstract=(ROOT/'abstract.txt').read_text().strip()
    (OUT/'abstract.txt').write_text(abstract+'\n')
    assert abstract.isascii() and len(abstract)<=1920
    metadata=dict(title='Distributionally Robust Portfolio Choice under Stochastic Volatility: A Review and Reproducible Multi-Asset Study',
        authors='Sheehan Mathur (Birla Institute of Technology and Science Pilani, Dubai Campus)',abstract=abstract,
        suggested_primary_category='q-fin.PM',comments=f'{pages} pages; selective review and reproducible computational study',
        software_release='https://github.com/Sheehan007/Distributionally-Robust-Optimization/releases/tag/v1.0.0',
        arxiv_status='Not submitted to arXiv',provider_observations_distributed=False,abstract_characters=len(abstract))
    if (ROOT/'archive_record.json').exists():metadata['software_archive']=json.loads((ROOT/'archive_record.json').read_text())
    (OUT/'arxiv_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    source=OUT/'DRO_arXiv_source.zip'
    papers=paper_files(ROOT)
    # The primary source bundle does not need the verification ledger.
    write_zip(source,[(ROOT/p,p.relative_to('paper')) for p in papers if p.name!='SOURCE_LEDGER.md'])
    selected=public_files(ROOT)
    manifest=dict(version='1.0.0',provider_observations_distributed=False,
        input_policy='Canonical hashes and acquisition instructions; exact numerical reproduction is conditional on matching authorized inputs.',
        sha256={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in selected})
    (ROOT/'public_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    code=OUT/'DRO_Study_Code_v1.0.0.zip'
    write_zip(code,[(ROOT/p,p) for p in selected]+[(ROOT/'public_manifest.json',Path('public_manifest.json'))])
    hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pdf,source,code]}
    (OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(pages=pages,public_files=len(selected),source_files=len(papers),archives=[str(source),str(code)]),indent=2))

if __name__=='__main__':main()
