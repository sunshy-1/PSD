"""Validate existing assets; fetch only missing assets into repository paths."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = json.loads((ROOT / 'configs/assets.json').read_text())


def checkpoint_complete(path, expected_type):
    try:
        config = json.loads((path / 'config.json').read_text())
        if config['model_type'] != expected_type:
            raise ValueError('Wrong checkpoint model_type: ' + str(path))
        index = path / 'model.safetensors.index.json'
        names = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        return (all((path / n).is_file() and (path / n).stat().st_size > 1024 for n in names)
                and (path / 'tokenizer_config.json').is_file()
                and any((path / n).is_file() for n in ['tokenizer.json', 'tokenizer.model', 'vocab.json']))
    except FileNotFoundError:
        return False


def ensure_assets(families, datasets, offline=False):
    records = {'models': {}, 'datasets': {}}
    for family in families:
        asset = ASSETS['models'][family]; path = ROOT / 'models' / asset['directory']
        if not checkpoint_complete(path, asset['model_type']):
            if offline:
                raise FileNotFoundError('Offline checkpoint incomplete: ' + str(path))
            from huggingface_hub import HfApi, snapshot_download
            revision = HfApi().model_info(asset['repo_id'], revision=asset['revision']).sha
            print('Downloading checkpoint:', asset['repo_id'], '->', path, flush=True)
            snapshot_download(asset['repo_id'], revision=revision, local_dir=path,
                allow_patterns=['*.safetensors', '*.json', '*.model', '*.py', '*.txt', '*.tiktoken', '*.jinja'])
            (path / 'download_provenance.json').write_text(json.dumps(dict(repo_id=asset['repo_id'], revision=revision), indent=2))
        if not checkpoint_complete(path, asset['model_type']):
            raise RuntimeError('Checkpoint verification failed: ' + str(path))
        index = path / 'model.safetensors.index.json'
        provenance = path / 'download_provenance.json'
        records['models'][family] = dict(path=path.relative_to(ROOT).as_posix(), repo_id=asset['repo_id'],
            provenance=json.loads(provenance.read_text()) if provenance.exists() else 'existing_local_checkpoint',
            config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
            weight_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None)
    for dataset in datasets:
        asset = ASSETS['datasets'][dataset]; directory = ROOT / 'datasets' / dataset
        for split, count in asset['files'].items():
            path = directory / (split + '.jsonl')
            if not path.exists():
                from datasets import load_dataset, DownloadConfig
                print('Preparing dataset:', asset['repo_id'], split, flush=True)
                rows = load_dataset(asset['repo_id'], asset['config'], split=split, revision=asset['revision'],
                    download_config=DownloadConfig(local_files_only=offline))
                directory.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.jsonl.tmp')
                tmp.write_text(''.join(json.dumps(dict(row), ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')
                tmp.replace(path)
            with path.open(encoding='utf-8') as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            if len(rows) != count:
                raise ValueError(f'Dataset {path} has {len(rows)} rows; expected {count}')
            required = {'gsm8k': {'question','answer'}, 'humaneval': {'prompt','test','entry_point'},
                        'mbpp': {'text','code','test_list','task_id'}}[dataset]
            if any(not required <= row.keys() for row in rows):
                raise ValueError('Dataset schema mismatch: ' + str(path))
            records['datasets'][dataset+'/'+split] = dict(rows=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                repo_id=asset['repo_id'], requested_revision=asset['revision'])
    return records
