import json
import core
from demo_data import DOCUMENTS, CASES

for name,pages in DOCUMENTS.items():
    print(core.index_pages(name,pages),flush=True)
model='phi4-mini:latest'
run,results=core.evaluate([CASES[0],CASES[-1]],model,4,
    lambda i,n:print(f'Live case {i+1}/{n}',flush=True))
print(json.dumps({'run':run,'metrics':core.metrics(results),'results':results},indent=2),flush=True)
if any('error' in result for result in results): raise SystemExit(1)
