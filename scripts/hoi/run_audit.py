"""Re-run the local, read-only M1-M6 audit; never recapture M0 or write NPZ.

Run from this repository's existing virtualenv. USD is read with a separate
existing pxr-capable environment; no packages are installed.
"""
import argparse,subprocess,sys
from pathlib import Path
from baseline import ROOT,sha
import json
from supermimic.hoi.provenance import effective_sources
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--usd-python',default='/home/han/XJINproject/magicZ1/Z1/soma_retargeter_autoconfig/.venv/bin/python')
a=p.parse_args()
base=json.loads((ROOT/'docs/hoi/asset_audit/baseline.json').read_text())
sources=effective_sources([{'path':path,'sha256':v['sha256']} for path,v in base['inputs'].items()])
for source in sources:
    if sha(source['path'])!=source['sha256']:raise RuntimeError('Protected input changed: '+source['path'])
for name,python in [('audit_npz',sys.executable),('asset_common',sys.executable),('read_usd',a.usd_python),('audit_robot',sys.executable),('audit_contact',sys.executable),('audit_internal',sys.executable),('audit_support',sys.executable),('visualize_audit',sys.executable),('visualize_support',sys.executable),('build_manifest',sys.executable)]:
    subprocess.run([python,str(ROOT/'scripts/hoi'/f'{name}.py')],cwd=ROOT,check=True)
subprocess.run([sys.executable,'-m','pytest','tests/test_hoi_asset_audit.py','-q'],cwd=ROOT,check=True)
for source in sources:
    if sha(source['path'])!=source['sha256']:raise RuntimeError('Protected input changed: '+source['path'])
print('Audit checks completed. Physical hand/contact and support gates remain FAILED. No training.')
