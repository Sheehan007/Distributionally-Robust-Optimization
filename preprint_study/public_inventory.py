"""Explicit public allowlist: no provider observations or time-series caches."""
from pathlib import Path
import re

AGGREGATES=[
 'forecast_metrics.csv','forecast_bootstrap.csv','validation_gates.csv','gate_audit.csv',
 'training_all.csv','SPY_training.csv','IWM_training.csv','TLT_training.csv','GLD_training.csv',
 'calibration_all.csv','SPY_calibration.csv','IWM_calibration.csv','TLT_calibration.csv','GLD_calibration.csv','calibration_scores.csv',
 'portfolio_metrics.csv','portfolio_bootstrap.csv','portfolio_quantiles.csv','portfolio_sensitivity.csv','cash_metrics.csv',
 'paired_metrics.csv','paired_bootstrap.csv','paired_mc_summary.csv','paired_solver_summary.csv',
 'monte_carlo_summary.csv','full_epigraph_validation.csv','asset_calendar_returns.csv']
JSON_SUMMARIES=['calibration_accuracy.json','checkpoint_replay.json','gold_concentration.json','portfolio_metadata.json',
 'study_summary.json','validation_summary.json']

def paper_files(root):
    root=Path(root);todo=[Path('main.tex')];seen=set()
    while todo:
        p=todo.pop()
        if p in seen:continue
        seen.add(p);s=(root/'paper'/p).read_text()
        for name in re.findall(r'\\input\{([^}]+)\}',s):
            v=Path(name);todo.append(v if v.suffix else v.with_suffix('.tex'))
        for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s):
            v=Path(name);seen.add(v if v.suffix else v.with_suffix('.pdf'))
    seen.update([Path('references.bib'),Path('main.bbl'),Path('LICENSE'),Path('SOURCE_LEDGER.md')])
    return [Path('paper')/p for p in sorted(seen)]

def public_files(root,include_pdf=True):
    root=Path(root)
    paths=[p.relative_to(root) for p in root.glob('*.py')]
    paths += [Path(n) for n in ['README.md','DATA_ACCESS.md','LICENSE','CITATION.cff','.zenodo.json','.gitignore',
        'protocol.json','calibration_starts.json','environment.json','requirements.txt','abstract.txt','data/snapshot_manifest.json']]
    if (root/'archive_record.json').exists():paths.append(Path('archive_record.json'))
    paths += [Path('results')/n for n in AGGREGATES+JSON_SUMMARIES]
    paths += [p.relative_to(root) for p in (root/'results').glob('*.pt')]
    paths += [p.relative_to(root) for p in (root/'results').glob('*_calibration_20??.json')]
    paths += paper_files(root)
    if include_pdf:paths.append(Path('paper/main.pdf'))
    for p in paths:assert (root/p).is_file(),p
    assert len(set(paths))==len(paths)
    return sorted(paths)
