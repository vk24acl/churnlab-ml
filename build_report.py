"""Rebuild the self-contained data shell after running the training pipeline."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
report=json.loads((root/'reports/metrics.json').read_text())
model_path=root/'reports/model.json'
model=json.loads(model_path.read_text()) if model_path.exists() else None
template=(root/'report_template.html').read_text()
(root/'index.html').write_text(template.replace('__REPORT__',json.dumps(report)).replace('__MODEL__','null').replace('<script src="app.js"></script>', '<script type="module">window.MODEL=await (await fetch("reports/model.json")).json();await import("./app.js");</script>'))
