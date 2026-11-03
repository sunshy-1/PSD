"""Aggregate every configuration from measured totals and scored samples."""
import csv
import json
from pathlib import Path

from .experiment_utils import atomic_json, source_manifest


def summarize(run, require_complete=False):
    run=Path(run);plan=json.loads((run/'run_plan.json').read_text());table=[];missing=[]
    for family in plan['families']:
        for dataset in plan['datasets']:
            folder=run/family/dataset;path=folder/'records.jsonl'
            rows=[]
            if path.exists():
                with path.open(encoding='utf-8') as stream:rows=[json.loads(l) for l in stream if l.strip()]
            keys={(r['case_index'],r['configuration']) for r in rows}
            assert len(keys)==len(rows),'Duplicate measurement keys'
            group=[]
            for config in plan['configurations']:
                values=[r for r in rows if r['configuration']==config['name']]
                if len(values)!=plan['case_counts'][dataset]:missing.append((family,dataset,config['name'],len(values)))
                if not values:continue
                assert all(r['forward_counter_matches'] and r['unresolved_prefix_masks']==0 for r in values)
                tokens=sum(r['generated_tokens'] for r in values);seconds=sum(r['seconds'] for r in values)
                forwards=sum(r['actual_forward_calls'] for r in values);n=len(values)
                item=dict(family=family,dataset=dataset,configuration=config['name'],method=config['method'],k=config['k'],
                    candidates=config['candidates'],graph_variant='dag',samples=n,correct=sum(r['correct'] for r in values),
                    accuracy=sum(r['correct'] for r in values)/n,accuracy_metric=values[0]['accuracy_metric'],
                    raw_accuracy=sum(r['raw_correct'] for r in values)/n if dataset!='gsm8k' else None,
                    strict_accuracy=sum(r['strict_correct'] for r in values)/n if dataset=='gsm8k' else None,
                    tokens=tokens,seconds=seconds,forward_calls=forwards,tps=tokens/seconds,tpf=tokens/forwards,
                    unresolved_prefix_masks=sum(r['unresolved_prefix_masks'] for r in values),
                    forward_counter_mismatches=sum(not r['forward_counter_matches'] for r in values))
                group.append(item)
            greedy=next((r for r in group if r['configuration']=='greedy'),None)
            for item in group:
                item['tps_speedup']=item['tps']/greedy['tps'] if greedy else None
                item['tpf_gain']=item['tpf']/greedy['tpf'] if greedy else None
                item['conversion']=item['tps_speedup']/item['tpf_gain'] if greedy and item['configuration']!='greedy' else None
            table.extend(group)
    if require_complete:
        assert not missing,missing
        assert source_manifest(run)==plan['source_hashes'],'Source manifest changed'
        assert all((run/f/d/'COMPLETED.json').exists() for f in plan['families'] for d in plan['datasets'])
    definitions=dict(accuracy='GSM8K harness v0.4.8 flexible exact match; code normalized pass@1. Raw code pass@1 and strict GSM8K shown separately.',
        tps='Sum of emitted prefix tokens / sum of CUDA-synchronized generation seconds, including stopping checks.',
        tpf='Sum of emitted prefix tokens / observed top-level model forward calls.',
        conversion='(TPS/greedy TPS)/(TPF/greedy TPF) for the same model/dataset.',
        k='Speculative depth excluding root; original DAG has 2*k candidates.',
        stopping='Complete the prefix before the first EOS; also stop at task delimiters in the decoded stable prefix.',
        extraction='Deterministic code container removal; no code repair and no selection using tests.')
    report=dict(complete=not missing,results=table,missing=missing,definitions=definitions,plan=plan)
    atomic_json(run/'summary.json',report)
    if table:
        with (run/'summary.csv').open('w',encoding='utf-8-sig',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Native DAG evaluation results','',
           '| Model | Dataset | Method | k | Candidates | N | ACC | Raw code ACC | TPS | TPF | Conversion |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        raw='—' if r['raw_accuracy'] is None else f"{100*r['raw_accuracy']:.1f}%"
        conversion='—' if r['conversion'] is None else f"{100*r['conversion']:.1f}%"
        lines.append(f"| {r['family']} | {r['dataset']} | {r['configuration']} | {r['k'] or '—'} | {r['candidates']} | {r['samples']} | {100*r['accuracy']:.1f}% | {raw} | {r['tps']:.2f} | {r['tpf']:.3f} | {conversion} |")
    lines.extend(['','ACC: GSM8K flexible exact match; code normalized pass@1. k is depth, not candidate count.',
                  'Raw outputs, token IDs, extraction rules, test verdicts and timings are retained per case.',
                  'These results are not a claim of exact reproduction of unpublished paper settings.',''])
    (run/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    return report
