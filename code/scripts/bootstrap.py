#!/usr/bin/env python3
"""Find a compatible runtime or install a project-local environment."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROBE = """import json,sys
from importlib.metadata import version
assert (3,10)<=sys.version_info[:2]<=(3,12)
assert version('transformers')=='4.53.3'
assert (2,7)<=tuple(map(int,version('torch').split('+')[0].split('.')[:2]))<(2,8)
for name in ['accelerate','datasets','sentencepiece','protobuf','huggingface_hub']:
 version(name)
import torch,transformers
assert torch.version.cuda is not None, 'A CUDA-enabled PyTorch build is required'
print(json.dumps({'python':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__}))
"""


def candidates():
    values = [sys.executable, str(ROOT / '.venv/bin/python')]
    if os.environ.get('PSD_PYTHON'):
        values.insert(0, os.environ['PSD_PYTHON'])
    conda_bins = [shutil.which('conda'), os.environ.get('CONDA_EXE')]
    conda_bins.extend(str(Path.home()/name/'bin/conda') for name in ['miniconda3','anaconda3','miniforge3','mambaforge'])
    for conda in dict.fromkeys(p for p in conda_bins if p and Path(p).is_file()):
        try:
            data = json.loads(subprocess.check_output([conda, 'env', 'list', '--json'], text=True, timeout=20))
            values.extend(str(Path(p) / 'bin/python') for p in data['envs'])
        except (subprocess.SubprocessError, ValueError, KeyError):
            pass
    return list(dict.fromkeys(p for p in values if Path(p).is_file()))


def main():
    os.chdir(ROOT)
    # Planning/help must work without packages, GPUs, checkpoints or network.
    if any(s in sys.argv[1:] for s in ['--help', '-h', '--dry-run']):
        sys.path.insert(0, str(ROOT))
        from evaluation.experiment import main as plan
        plan()
        return
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
    runtime = None
    for python in candidates():
        try:
            result = subprocess.run([python, '-B', '-c', PROBE], capture_output=True, text=True, env=env, timeout=60)
        except (subprocess.SubprocessError, OSError):
            continue
        if result.returncode == 0:
            runtime = python
            print('Runtime:', result.stdout.strip(), flush=True)
            break
    if runtime is None:
        if '--offline' in sys.argv:
            raise SystemExit('No compatible offline runtime. Install requirements.txt or set PSD_PYTHON.')
        runtime = str(ROOT / '.venv/bin/python')
        creator = None
        for name in ['python3.12', 'python3.11', 'python3.10']:
            if shutil.which(name):
                creator = shutil.which(name); break
        if creator is None and (3, 10) <= sys.version_info[:2] <= (3, 12):
            creator = sys.executable
        if creator is None:
            raise SystemExit('Python 3.10–3.12 is required to create .venv. Set PSD_PYTHON to a compatible interpreter.')
        subprocess.run([creator, '-m', 'venv', str(ROOT / '.venv')], check=True)
        subprocess.run([runtime, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([runtime, '-m', 'pip', 'install', 'torch==2.7.1', '--index-url',
                        os.environ.get('PSD_TORCH_INDEX', 'https://download.pytorch.org/whl/cu128')], check=True)
        subprocess.run([runtime, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')], check=True)
    os.execv(runtime, [runtime, '-B', '-m', 'evaluation.experiment'] + sys.argv[1:])


if __name__ == '__main__':
    main()
