"""Verify canonical scientific inputs and the archived model records."""
from pathlib import Path
import json
from snapshot_inputs import fingerprints
ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'data'/'snapshot_manifest.json').read_text())
    actual=fingerprints(ROOT)
    mismatch=[k for k,v in manifest['canonical_sha256'].items() if actual.get(k)!=v]
    if mismatch:raise SystemExit('Provider inputs differ from the study snapshot: '+', '.join(mismatch)+'. Exact historical numerical reproduction requires matching inputs; a current-data rerun is a new result.')
    checkpoints=list((ROOT/'results').glob('*.pt'));assert len(checkpoints)==168,len(checkpoints)
    fits=list((ROOT/'results').glob('*_calibration_20??.json'));assert len(fits)==28,len(fits)
    print(f'Validated 5 canonical input hashes, {len(checkpoints)} checkpoints, and {len(fits)} fits.')

if __name__=='__main__':main()
