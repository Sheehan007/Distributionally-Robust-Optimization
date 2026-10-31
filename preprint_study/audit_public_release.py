"""Verify a published manifest and enforce the public artifact allowlist."""
from pathlib import Path
import hashlib,json
from public_inventory import public_files
ROOT=Path(__file__).resolve().parent

def main():
    paths=public_files(ROOT);manifest_path=ROOT/'public_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        assert set(manifest['sha256'])=={str(p) for p in paths}
        for p in paths:assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==manifest['sha256'][str(p)],p
    for p in paths:
        assert p.parent!=Path('data') or p.name=='snapshot_manifest.json',p
        assert not any(x in p.name for x in ['_daily.csv','_trades.csv','_forecasts.csv','_filtered_states.csv','_raw.json']),p
    assert len(list((ROOT/'results').glob('*.pt')))==168
    assert len(list((ROOT/'results').glob('*_calibration_20??.json')))==28
    assert len(json.loads((ROOT/'data/snapshot_manifest.json').read_text())['canonical_sha256'])==5
    print(f'Public inventory verified: {len(paths)} files; no provider observations or time-series caches selected.')

if __name__=='__main__':main()
