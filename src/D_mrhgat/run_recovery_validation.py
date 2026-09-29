"""Run accuracy and the original plan's directional prerequisites together."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from D_mrhgat.run_validation import main
from D_mrhgat.complete_recovery_gate import complete
from D_mrhgat.extend_recovery_gate import extend
from D_mrhgat.run_null_validation import run_null_validation

if __name__=='__main__':
    main()
    status=json.loads((ROOT/'reports/D6_progress.json').read_text(encoding='utf-8'))
    run=Path(status['run_directory'])
    environment=json.loads((run/'environment.json').read_text(encoding='utf-8'))
    if complete(run,environment['device']):
        if extend(run,environment['device']):
            run_null_validation(run,environment['device'])
