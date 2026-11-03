"""Shared experiment plans, source manifests and idle-GPU checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_manifest(output):
    output = Path(output).resolve(); values = {}
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in {'models','datasets','results','.venv','.git','__pycache__'}
                         and not (Path(base)/d).resolve().is_relative_to(output))
        for name in sorted(files):
            path = Path(base)/name
            if path.is_file() and name != '.DS_Store':
                values[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def gpu_inventory():
    text = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    values = []
    for line in text.splitlines():
        index,uuid,name,total,used,util = [s.strip() for s in line.split(',')]
        values.append(dict(index=int(index),uuid=uuid,name=name,memory_total_mb=int(total),
                           memory_used_mb=int(used),utilization=int(util)))
    return values


def compute_apps():
    text = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    return [dict(uuid=v[0],pid=int(v[1]),name=v[2],memory=v[3]) for line in text.splitlines()
            if len(v := [s.strip() for s in line.split(',')]) == 4]


def is_idle(gpu, apps):
    return gpu['memory_used_mb'] < 256 and gpu['utilization'] < 5 and not any(a['uuid']==gpu['uuid'] for a in apps)


def configurations(methods, ks):
    configs = []
    if 'greedy' in methods:
        configs.append(dict(name='greedy',method='greedy',alg='greedy',k=None,candidates=1))
    for method,alg in [('psd_confidence','confidence_threshold'),('psd_local_leap','local_leap')]:
        if method in methods:
            configs.extend(dict(name=f'{method}-k{k}',method='PSD',alg=alg,k=k,candidates=2*k) for k in ks)
    return configs
