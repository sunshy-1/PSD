"""Public one-command experiment driver. Planning requires only the stdlib."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .experiment_utils import ROOT, atomic_json, configurations, source_manifest, gpu_inventory, compute_apps, is_idle


def parser():
    p = argparse.ArgumentParser(description='Prepare assets, wait for idle GPUs, evaluate the original DAG and summarize every configuration.')
    p.add_argument('--families', default='dream,llada,openpangu')
    p.add_argument('--datasets', default='gsm8k,humaneval,mbpp')
    p.add_argument('--methods', default='greedy,psd_confidence,psd_local_leap')
    p.add_argument('--ks', default='1,3,5,7', help='Speculative depths; original DAG candidates = 2*k')
    p.add_argument('--samples', default='50', help='First N test cases, or all')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--block-length', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threshold', type=float, default=.9)
    p.add_argument('--gpus', help='Physical GPU indices or UUIDs; defaults to CUDA_VISIBLE_DEVICES or all GPUs')
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', type=Path, help='Resume the saved plan; protocol options come from that plan')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def make_plan(args, p):
    def subset(value, valid):
        parts=value.split(',')
        if not parts or len(set(parts))!=len(parts) or any(v not in valid for v in parts):
            p.error('Choose a unique subset of '+','.join(valid))
        return parts
    families=subset(args.families, ['dream','llada','openpangu'])
    datasets=subset(args.datasets, ['gsm8k','humaneval','mbpp'])
    methods=subset(args.methods, ['greedy','psd_confidence','psd_local_leap'])
    try:ks=[int(k) for k in args.ks.split(',')]
    except ValueError:p.error('--ks must contain integers')
    if not ks or len(set(ks))!=len(ks) or any(k<1 or k>32 for k in ks):p.error('Use unique k values between 1 and 32')
    sizes={'gsm8k':1319,'humaneval':164,'mbpp':500}
    try:counts={d:sizes[d] if args.samples=='all' else min(int(args.samples),sizes[d]) for d in datasets}
    except ValueError:p.error('--samples must be a positive integer or all')
    if any(n<1 for n in counts.values()):p.error('--samples must be positive')
    if args.max_new_tokens<1 or args.block_length<1 or args.max_new_tokens%args.block_length:
        p.error('--max-new-tokens must be positive and divisible by --block-length')
    if not 0<=args.threshold<=1:p.error('--threshold must be in [0,1]')
    return dict(protocol_version=1,families=families,datasets=datasets,configurations=configurations(methods,ks),
        case_counts=counts,max_new_tokens=args.max_new_tokens,block_length=args.block_length,seed=args.seed,
        threshold=args.threshold,graph_variant='dag',batch_size=1)


def available_gpus(selection):
    inventory=gpu_inventory()
    if selection is None:selection=os.environ.get('CUDA_VISIBLE_DEVICES')
    if selection is not None:
        tokens=selection.split(',')
        selected=[g for g in inventory if str(g['index']) in tokens or g['uuid'] in tokens]
        if len(selected)!=len(tokens):raise ValueError('Unknown or duplicate physical GPU selection: '+selection)
        return selected
    return inventory


def schedule(run, plan, selection):
    import fcntl
    with (run/'.driver.lock').open('a') as driver_lock:
        try:fcntl.flock(driver_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another driver is already running this output directory')
        _schedule(run, plan, selection)


def _schedule(run, plan, selection):
    import fcntl
    from .experiment_report import summarize
    cards=available_gpus(selection)
    if not cards:raise RuntimeError('No NVIDIA GPU is visible')
    pending=[(f,d) for f in plan['families'] for d in plan['datasets'] if not (run/f/d/'COMPLETED.json').exists()]
    running={};quarantined=set();attempts={};events=[];last_wait=0
    lock_root=ROOT/'results'/'.gpu_locks';lock_root.mkdir(parents=True,exist_ok=True)
    try:
        while pending or running:
            for index,entry in list(running.items()):
                proc,job,log,lock=entry;code=proc.poll()
                if code is None:continue
                log.close();fcntl.flock(lock,fcntl.LOCK_UN);lock.close();del running[index]
                if code==75:
                    quarantined.add(index);attempts[job]=attempts.get(job,0)+1
                    destination=run/'discarded_attempts'/f'{job[0]}_{job[1]}_{attempts[job]}'
                    destination.parent.mkdir(exist_ok=True);(run/job[0]/job[1]).rename(destination)
                    pending.insert(0,job);events.append(dict(event='foreign_process_discard',gpu=index,job=job,path=str(destination),time=time.time()))
                    atomic_json(run/'gpu_events.json',events)
                    print('GPU contention: discarded the entire job attempt; retrying on another idle GPU.',job,index,flush=True)
                elif code:
                    raise RuntimeError(f'Worker {job} failed ({code}); see {run/job[0]/job[1]/"worker.log"}')
                else:
                    print('Completed:',*job,flush=True);summarize(run)
            if pending:
                inventory={g['index']:g for g in gpu_inventory()};apps=compute_apps()
                for card in cards:
                    index=card['index']
                    if not pending:break
                    if index in running or index in quarantined or not is_idle(inventory[index],apps):continue
                    lock=(lock_root/(card['uuid']+'.lock')).open('a')
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:lock.close();continue
                    job=pending.pop(0);folder=run/job[0]/job[1];folder.mkdir(parents=True,exist_ok=True)
                    log=(folder/'worker.log').open('a',encoding='utf-8')
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='8',
                        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
                    command=[sys.executable,'-B','-m','evaluation.experiment_worker','--run',str(run),
                        '--family',job[0],'--dataset',job[1],'--gpu',str(index)]
                    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    running[index]=(proc,job,log,lock)
                    print('Started:',*job,'GPU',index,'pid',proc.pid,flush=True)
            if pending and not running and all(c['index'] in quarantined for c in cards):
                raise RuntimeError('All selected GPUs were quarantined after contention. Resume later on other GPUs.')
            if pending and not running and time.time()-last_wait>60:
                print('Waiting for an idle GPU (no compute processes, memory <256 MiB, utilization <5%).',flush=True);last_wait=time.time()
            if pending or running:time.sleep(3)
    finally:
        # Only subprocesses created by this driver are terminated on interruption.
        for proc,job,log,lock in running.values():
            if proc.poll() is None:proc.terminate()
        for proc,job,log,lock in running.values():
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close();lock.close()
    report=summarize(run,require_complete=True)
    atomic_json(run/'COMPLETED.json',dict(configurations=len(report['results']),records=sum(r['samples'] for r in report['results']),sources_unchanged=True))
    print((run/'summary.md').read_text(),flush=True)
    print('Saved results:',run,flush=True)


def main():
    p=parser();args=p.parse_args()
    if args.resume:
        run=args.resume.resolve();plan=json.loads((run/'run_plan.json').read_text())
    else:
        plan=make_plan(args,p)
        run=(args.output or ROOT/'results'/time.strftime('dag_%Y%m%d_%H%M%S')).resolve()
    if ROOT.is_relative_to(run):p.error('Output must not be the repository root or its parent')
    if args.dry_run:
        print(json.dumps(dict(**{k:v for k,v in plan.items() if k not in {'assets','source_hashes'}},
            result_rows=len(plan['families'])*len(plan['datasets'])*len(plan['configurations']),
            measurements=sum(plan['case_counts'].values())*len(plan['families'])*len(plan['configurations']),output=str(run)),indent=2))
        return
    from .assets import ensure_assets
    assets=ensure_assets(plan['families'],plan['datasets'],args.offline)
    if args.prepare_only:print(json.dumps(assets,indent=2));return
    from .code_execution import validate_executor
    if any(d!='gsm8k' for d in plan['datasets']):validate_executor(run)
    sources=source_manifest(run)
    if args.resume:
        if sources!=plan['source_hashes']:raise RuntimeError('Source changed since the saved run; start a new output directory')
        if assets!=plan['assets']:raise RuntimeError('Model/dataset provenance changed since the saved run')
    else:
        if (run/'run_plan.json').exists():raise FileExistsError('Output already contains a plan; use --resume')
        plan.update(assets=assets,source_hashes=sources,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        atomic_json(run/'run_plan.json',plan)
    print('Run:',run,flush=True)
    schedule(run,plan,args.gpus)


if __name__=='__main__':main()
