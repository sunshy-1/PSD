"""One model/dataset job; one model load, shared prompts and resumable records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

from .experiment_utils import ROOT, atomic_json, source_manifest, gpu_inventory, compute_apps, is_idle


class GPUContention(RuntimeError):
    pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--family',required=True);parser.add_argument('--dataset',required=True);parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();run=args.run;plan=json.loads((run/'run_plan.json').read_text());out=run/args.family/args.dataset
    out.mkdir(parents=True,exist_ok=True);configs=plan['configurations'];count=plan['case_counts'][args.dataset];total=count*len(configs)
    rows=[];stop=threading.Event();foreign=[]
    def status(state,**extra):
        atomic_json(out/'status.json',dict(state=state,pid=os.getpid(),family=args.family,dataset=args.dataset,
            gpu=args.gpu,completed=len(rows),total=total,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra))
    try:
        status('WAITING_FOR_IDLE_GPU');checks=[]
        for _ in range(3):
            gpu=next(g for g in gpu_inventory() if g['index']==args.gpu)
            if not is_idle(gpu,compute_apps()):raise GPUContention('GPU changed owner before loading')
            checks.append(dict(time=time.time(),**gpu));time.sleep(2)
        atomic_json(out/'gpu_idle_checks.json',checks);uuid=gpu['uuid']
        def guard():
            unexpected=[a for a in compute_apps() if a['uuid']==uuid and a['pid']!=os.getpid()]
            if unexpected:foreign.extend(unexpected)
            if foreign:raise GPUContention('Foreign GPU process: '+repr(foreign[:3]))
        def monitor():
            while not stop.is_set():
                try:guard()
                except GPUContention:return
                except Exception as exc:foreign.append(dict(monitor_error=repr(exc)));return
                stop.wait(.5)
        threading.Thread(target=monitor,daemon=True).start()
        before=source_manifest(run)
        assert before==plan['source_hashes'],'Source files changed after planning'
        import torch
        import transformers
        from . import benchmark as bench
        from .assets import ASSETS
        from .scoring import score_math, prepare_code, EXTRACTION_VERSION
        from .code_execution import score_code
        from src.psd import get_graph_structure
        for c in configs:
            if c['method']=='PSD':assert len(get_graph_structure(c['k']))==c['candidates']
        labels={'gsm8k':'GSM8K 5-shot','humaneval':'HumanEval 0-shot','mbpp':'MBPP 3-shot'}
        cases=bench.load_jsonl(ROOT/'datasets'/args.dataset/'test.jsonl')[:count]
        prompts=bench.prepare_datasets(ROOT/'datasets',count,[args.dataset])[labels[args.dataset]]
        metadata=dict(family=args.family,dataset=args.dataset,cases=count,configurations=configs,
            prompt_hashes=[hashlib.sha256(p.encode()).hexdigest() for p in prompts],gpu=gpu,
            python=sys.executable,torch=torch.__version__,transformers=transformers.__version__,
            max_new_tokens=plan['max_new_tokens'],block_length=plan['block_length'],seed=plan['seed'],
            graph_variant='dag',extraction_version=EXTRACTION_VERSION,policy='resolved EOS prefix; task stops on decoded stable prefix')
        if (out/'metadata.json').exists():
            old=json.loads((out/'metadata.json').read_text())
            for key in ['prompt_hashes','configurations','max_new_tokens','block_length','seed','extraction_version']:
                assert old[key]==metadata[key],('Resume protocol changed',key)
        atomic_json(out/'metadata.json',metadata)
        records=out/'records.jsonl'
        if records.exists():
            content=records.read_bytes()
            if content and not content.endswith(b'\n'):
                complete,separator,tail=content.rpartition(b'\n')
                (out/('interrupted_tail_'+str(time.time_ns())+'.txt')).write_bytes(tail)
                records.write_bytes(complete+separator)
            with records.open(encoding='utf-8') as stream:rows=[json.loads(s) for s in stream if s.strip()]
        done={(r['case_index'],r['configuration']) for r in rows}
        assert len(done)==len(rows),'Duplicate measurements in resume file'
        status('LOADING_MODEL')
        cls,greedy,psd,spiffy,lopa=bench.load_family(args.family)
        model_path=ROOT/'models'/ASSETS['models'][args.family]['directory']
        model=cls.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True,trust_remote_code=True).to('cuda').eval()
        tokenizer=bench.AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=True)
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        tokenizer.padding_side='left';mask=ASSETS['models'][args.family]['mask_token_id']
        def generate(index,config):
            guard()
            def expired(*unused):raise TimeoutError('Generation exceeded 180 seconds')
            signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,180)
            try:
                result=bench.run_batch(args.family,model,tokenizer,greedy,psd,spiffy,lopa,[prompts[index]],
                    config['method'],config['k'] or 1,plan['block_length'],plan['max_new_tokens'],config['alg'],
                    plan['threshold'],mask,cache_sharing=False,seed=plan['seed'],graph_variant='dag',dataset_key=args.dataset)
            finally:signal.setitimer(signal.ITIMER_REAL,0)
            guard();return result
        for config in configs:
            status('WARMUP',configuration=config['name']);generate(0,config)
        for index,case in enumerate(cases):
            for config in configs:
                if (index,config['name']) in done:continue
                status('RUNNING',case_index=index,configuration=config['name'])
                result=generate(index,config);text=result['raw_texts'][0]
                result.update(family=args.family,dataset=args.dataset,configuration=config['name'],k=config['k'],
                    candidates=config['candidates'],alg=config['alg'],graph_variant='dag',case_index=index,
                    case_id=case.get('task_id',index),gpu_index=args.gpu,gpu_uuid=uuid,raw_text=text)
                if args.dataset=='gsm8k':result.update(score_math(text,case['answer']))
                else:
                    prepared=prepare_code(text,case,args.dataset)
                    result.update(prepared);result.update(score_code(prepared,out))
                guard()
                with records.open('a',encoding='utf-8') as file:
                    file.write(json.dumps(result,ensure_ascii=False)+'\n');file.flush()
                rows.append(result);done.add((index,config['name']))
            if (index+1)%5==0:print(args.family,args.dataset,'cases',index+1,'/',count,flush=True)
        assert len(rows)==total
        assert source_manifest(run)==before,'Source files changed during evaluation'
        atomic_json(out/'COMPLETED.json',dict(records=total,sources_unchanged=True,unresolved_prefix_masks=0))
        status('COMPLETED')
    except GPUContention as exc:
        status('GPU_CONTENTION',error=str(exc));sys.exit(75)
    except Exception as exc:
        status('FAILED',error=str(exc),traceback=traceback.format_exc());raise
    finally:stop.set()


if __name__=='__main__':main()
