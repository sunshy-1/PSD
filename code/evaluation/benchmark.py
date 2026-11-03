#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.stopping import generated_lengths, generated_token_ids


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_gsm8k_prompt(fewshot_cases, case):
    sections = []
    for example in fewshot_cases:
        sections.append(
            f"Question: {example['question']}\nAnswer: {example['answer']}"
        )
    sections.append(f"Question: {case['question']}\nAnswer:")
    return "\n\n".join(sections)


def format_mbpp_example(row):
    tests = "\n".join(row.get("test", row.get("test_list", [])))
    problem = row.get("prompt", row.get("text", ""))
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{problem.strip()} Your code should pass these tests:\n\n"
        f"{tests}\n[BEGIN]\n"
    )


def build_mbpp_prompt(row, fewshot_rows):
    examples = []
    for example in fewshot_rows:
        code = example["code"].replace("\r", "").replace("\t", "    ")
        examples.append(format_mbpp_example(example) + code.strip() + "\n[DONE]")
    examples.append(format_mbpp_example(row))
    return "\n\n".join(examples)


def prepare_datasets(data_dir, num_cases, requested_datasets):
    datasets = {}
    if "gsm8k" in requested_datasets:
        gsm8k_train = load_jsonl(data_dir / "gsm8k" / "train.jsonl")
        gsm8k_test = load_jsonl(data_dir / "gsm8k" / "test.jsonl")
        if num_cases > len(gsm8k_test):
            raise ValueError(f"num_cases must be between 1 and {len(gsm8k_test)} for GSM8K")
        gsm8k_fewshot = random.Random(1234).sample(gsm8k_train, 5)
        datasets["GSM8K 5-shot"] = [
            build_gsm8k_prompt(gsm8k_fewshot, row)
            for row in gsm8k_test[:num_cases]
        ]
    if "humaneval" in requested_datasets:
        humaneval_test = load_jsonl(data_dir / "humaneval" / "test.jsonl")
        if num_cases > len(humaneval_test):
            raise ValueError(
                f"num_cases must be between 1 and {len(humaneval_test)} for HumanEval"
            )
        datasets["HumanEval 0-shot"] = [
            row["prompt"] for row in humaneval_test[:num_cases]
        ]
    if "mbpp" in requested_datasets:
        mbpp_prompt = load_jsonl(data_dir / "mbpp" / "prompt.jsonl")
        mbpp_test = load_jsonl(data_dir / "mbpp" / "test.jsonl")
        if num_cases > len(mbpp_test):
            raise ValueError(f"num_cases must be between 1 and {len(mbpp_test)} for MBPP")
        mbpp_fewshot = [
            row for row in mbpp_prompt if row["task_id"] in {2, 3, 4}
        ]
        if len(mbpp_fewshot) != 3:
            raise ValueError("MBPP prompt split is missing standard 3-shot examples")
        datasets["MBPP 3-shot"] = [
            build_mbpp_prompt(row, mbpp_fewshot)
            for row in mbpp_test[:num_cases]
        ]
    return datasets


def load_family(family, repo_root=None):
    if family == "dream":
        from src.dream.model.generation_utils_dream_7b import diffusion_generate
        from src.dream.model.generation_utils_dream_7b_PSD import (
            diffusion_generate_PSD, diffusion_generate_Spiffy,
        )
        from src.dream.model.generation_utils_dream_7b_LoPA import diffusion_generate_LoPA
        from src.dream.model.modeling_dream import DreamModel
        return DreamModel, diffusion_generate, diffusion_generate_PSD, diffusion_generate_Spiffy, diffusion_generate_LoPA
    if family == "llada":
        from src.llada.model.generation_utils_llada import diffusion_generate
        from src.llada.model.generation_utils_llada_PSD import diffusion_generate_PSD, diffusion_generate_Spiffy
        from src.llada.model.generation_utils_llada_LoPA import diffusion_generate_LoPA
        from src.llada.model.modeling_llada import LLaDAModelLM
        return LLaDAModelLM, diffusion_generate, diffusion_generate_PSD, diffusion_generate_Spiffy, diffusion_generate_LoPA
    from src.openpangu.generation_utils.generation_utils import diffusion_generate
    from src.openpangu.generation_utils.generation_pangu_dllm_7b_PSD_utils import (
        diffusion_generate_PSD, diffusion_generate_Spiffy,
    )
    from src.openpangu.generation_utils.generation_pangu_dllm_7b_LoPA_utils import diffusion_generate_LoPA
    from src.openpangu.model.modeling_openpangu_dense import PanguEmbeddedForCausalLM
    return PanguEmbeddedForCausalLM, diffusion_generate, diffusion_generate_PSD, diffusion_generate_Spiffy, diffusion_generate_LoPA


def resolved_lengths(output_ids, prompt_length, eos_token_id, mask_token_id, stop_token_ids=(), stop_positions=None):
    return generated_lengths(output_ids, prompt_length, eos_token_id, mask_token_id, stop_token_ids, stop_positions)


def encode_prompts(tokenizer, prompts, device):
    prepared = []
    for prompt in prompts:
        if tokenizer.bos_token and not prompt.startswith(tokenizer.bos_token):
            prompt = tokenizer.bos_token + prompt
        prepared.append(prompt)
    encoded = tokenizer(prepared, return_tensors="pt", padding=True)
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def run_batch(
    family,
    model,
    tokenizer,
    diffusion_generate,
    diffusion_generate_PSD,
    diffusion_generate_spiffy,
    diffusion_generate_lopa,
    prompts,
    method,
    draft_steps,
    block_length,
    max_new_tokens,
    alg,
    threshold,
    mask_token_id,
    k_branches=3,
    lopa_strategy="conservative",
    cache_sharing=None,
    seed=0,
    graph_variant="dag",
    stop_token_ids=(),
    dataset_key=None,
):
    random.seed(seed)
    torch.manual_seed(seed)
    input_ids, attention_mask = encode_prompts(tokenizer, prompts, model.device)
    from evaluation.scoring import TextStopState
    text_stop = TextStopState(tokenizer, dataset_key, prompts) if dataset_key else None
    common_kwargs = {
        "inputs": input_ids,
        "attention_mask": attention_mask,
        "block_length": block_length,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "mask_token_id": mask_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_decode_steps": True,
        "stop_on_any_eos": False,
        "stop_token_ids": stop_token_ids,
        "text_stop_state": text_stop,
    }
    if cache_sharing is not None:
        common_kwargs["cache_sharing"] = cache_sharing
    actual_calls = 0
    def count_forward(*unused, **unused_kwargs):
        nonlocal actual_calls
        actual_calls += 1
    handle = model.register_forward_pre_hook(count_forward, with_kwargs=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        if method in {"baseline", "greedy", "confidence", "local_leap"}:
            if family in {"dream", "llada"}:
                common_kwargs["dual_cache"] = True
            method_alg = {"confidence": "confidence_threshold", "local_leap": "local_leap"}.get(method, "greedy")
            method_threshold = threshold if method == "confidence" else 0.0
            output_ids, decode_steps = diffusion_generate(
                model,
                threshold=method_threshold,
                num_small_blocks=1,
                alg=method_alg,
                **common_kwargs,
            )
        elif method == "lopa":
            if family in {"dream", "llada"}:
                common_kwargs["dual_cache"] = True
            output_ids, decode_steps = diffusion_generate_lopa(
                model, threshold=threshold, alg="confidence_threshold",
                k_branches=k_branches, lopa_strategy=lopa_strategy, **common_kwargs,
            )
        elif method == "spiffy":
            if diffusion_generate_spiffy is None:
                raise ValueError(f"Spiffy is not available for family={family}")
            if family in {"dream", "llada"}:
                common_kwargs["dual_cache"] = True
            if family == "dream":
                common_kwargs["draft_parallel"] = False
                common_kwargs["context_parallel_opener"] = False
            output_ids, decode_steps = diffusion_generate_spiffy(
                model,
                draft_steps=draft_steps,
                graph_variant=graph_variant,
                **common_kwargs,
            )
        elif method == "PSD":
            if family in {"dream", "llada"}:
                common_kwargs["dual_cache"] = True
            if family == "dream":
                common_kwargs["draft_parallel"] = False
                common_kwargs["context_parallel_opener"] = False
            output_ids, decode_steps = diffusion_generate_PSD(
                model,
                threshold=threshold,
                draft_steps=draft_steps,
                alg=alg,
                graph_variant=graph_variant,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
    finally:
        handle.remove()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    lengths, hit_eos = resolved_lengths(
        output_ids, input_ids.shape[-1], tokenizer.eos_token_id, mask_token_id, stop_token_ids,
        text_stop.ends if text_stop else None
    )
    previews = []
    token_ids, raw_texts = [], []
    for row_index, row in enumerate(output_ids):
        generated = generated_token_ids(row, input_ids.shape[-1], tokenizer.eos_token_id, mask_token_id, stop_token_ids,
                                        text_stop.ends[row_index] if text_stop else None)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        previews.append(text[:160])
        token_ids.append(generated.detach().cpu().tolist())
        raw_texts.append(text)
    total_tokens = sum(lengths)
    if actual_calls != int(decode_steps):
        raise RuntimeError(f"Forward count mismatch: actual={actual_calls}, reported={decode_steps}")
    return {
        "method": method,
        "batch_size": len(prompts),
        "prompt_tokens_padded": int(input_ids.shape[-1]),
        "generated_tokens": total_tokens,
        "generated_tokens_per_branch": lengths,
        "decode_steps": int(decode_steps),
        "aggregate_tpf": total_tokens / decode_steps if decode_steps else 0.0,
        "normalized_tpf": total_tokens / (decode_steps * len(prompts)) if decode_steps else 0.0,
        "seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "hit_eos_per_branch": hit_eos,
        "stop_on_any_eos": False,
        "previews": previews,
        "actual_forward_calls": actual_calls,
        "forward_counter_matches": actual_calls == int(decode_steps),
        "token_ids": token_ids,
        "raw_texts": raw_texts,
        "stop_token_ids": stop_token_ids,
        "unresolved_prefix_masks": 0,
        "task_stop_positions": text_stop.ends if text_stop else None,
    }


def aggregate(results, batch_sizes, requested_methods):
    rows = []
    for dataset in sorted({result["dataset"] for result in results}):
        for batch_size in batch_sizes:
            selected = [
                result
                for result in results
                if result["dataset"] == dataset and result["batch_size"] == batch_size
            ]
            if not selected:
                continue
            methods = {}
            for method in requested_methods:
                method_results = [item for item in selected if item["method"] == method]
                if not method_results:
                    continue
                total_tokens = sum(item["generated_tokens"] for item in method_results)
                total_steps = sum(item["decode_steps"] for item in method_results)
                total_seconds = sum(item["seconds"] for item in method_results)
                methods[method] = {
                    "batches": len(method_results),
                    "tokens": total_tokens,
                    "steps": total_steps,
                    "seconds": total_seconds,
                    "aggregate_tpf": total_tokens / total_steps,
                    "normalized_tpf": total_tokens / (total_steps * batch_size),
                    "tps": total_tokens / total_seconds,
                }
            for method in requested_methods:
                if method not in methods:
                    continue
                current = methods[method]
                rows.append(
                    {
                        "dataset": dataset,
                        "batch_size": batch_size,
                        "method": method,
                        "metrics": current,
                    }
                )
    return rows


def make_markdown(rows):
    lines = [
        "| Dataset | Method | TPF | TPS |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{row['metrics']['aggregate_tpf']:.3f} | {row['metrics']['tps']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(args, batch_sizes, methods, results):
    rows = aggregate(results, batch_sizes, methods)
    report = {
        "family": args.family,
        "dtype": args.dtype,
        "seed": args.seed,
        "model": os.path.relpath(args.model.resolve(), Path(__file__).resolve().parents[1]),
        "batch_sizes": batch_sizes,
        "num_cases": args.num_cases,
        "draft_steps": args.draft_steps,
        "graph_variant": args.graph_variant,
        "block_length": args.block_length,
        "max_new_tokens": args.max_new_tokens,
        "alg": args.alg,
        "threshold": args.threshold,
        "cache_sharing": args.cache_sharing,
        "merge_projections": args.merge_projections if args.family == "dream" else False,
        "compile": args.compile,
        "k_branches": args.k_branches,
        "lopa_strategy": args.lopa_strategy,
        "methods": methods,
        "stop_on_any_eos": False,
        "summary": rows,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    json_temporary.replace(args.output)
    markdown_path = args.output.with_suffix(".md")
    markdown_temporary = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    markdown = make_markdown(rows)
    markdown_temporary.write_text(markdown)
    markdown_temporary.replace(markdown_path)
    return markdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("dream", "llada", "openpangu"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        default="gsm8k,humaneval,mbpp",
        help="Comma-separated subset of gsm8k,humaneval,mbpp",
    )
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--num-cases", type=int, default=20)
    parser.add_argument("--draft-steps", type=int, default=7)
    parser.add_argument("--graph-variant", choices=("dag", "chain", "heuristic_tree"), default="dag")
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--alg", default="confidence_threshold")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        default="greedy,PSD",
        help="Comma-separated subset of greedy,confidence,local_leap,lopa,spiffy,PSD",
    )
    parser.add_argument("--k-branches", type=int, default=3)
    parser.add_argument("--lopa-strategy", choices=("aggressive", "moderate", "conservative"), default="conservative")
    parser.add_argument("--cache-sharing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--merge-projections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.block_length < 1 or args.max_new_tokens < 1 or args.max_new_tokens % args.block_length:
        parser.error("max-new-tokens must be positive and divisible by block-length")
    if args.num_cases < 1:
        parser.error("num-cases must be positive")

    methods = [method for method in args.methods.split(",") if method]
    supported_methods = {"greedy", "confidence", "local_leap", "lopa", "spiffy", "PSD", "baseline"}
    invalid_methods = [method for method in methods if method not in supported_methods]
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {invalid_methods}")
    requested_batch_sizes = [
        int(value) for value in args.batch_sizes.split(",") if value
    ]
    if any(batch_size < 1 for batch_size in requested_batch_sizes):
        raise ValueError("batch sizes must be positive")
    batch_sizes = [
        batch_size
        for batch_size in requested_batch_sizes
        if batch_size <= args.num_cases
    ]
    if not batch_sizes:
        raise ValueError("no batch size is less than or equal to num_cases")

    repo_root = Path(__file__).resolve().parents[1]
    model_class, diffusion_generate, diffusion_generate_PSD, diffusion_generate_spiffy, diffusion_generate_lopa = load_family(
        args.family, repo_root
    )
    model = model_class.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=True,
        local_files_only=True,
    ).to("cuda").eval()
    if args.family == "dream" and args.merge_projections:
        from src.dream.model.modeling_dream import merge_dream_projections
        merge_dream_projections(model)
    if args.compile:
        model = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None and args.family == "openpangu":
        mask_token_id = 45830
    if mask_token_id is None:
        raise ValueError("Could not resolve mask_token_id")

    dataset_names = {
        "gsm8k": "GSM8K 5-shot",
        "humaneval": "HumanEval 0-shot",
        "mbpp": "MBPP 3-shot",
    }
    invalid_datasets = [
        name.strip()
        for name in args.datasets.split(",")
        if name.strip().lower() not in dataset_names
    ]
    if invalid_datasets:
        raise ValueError(f"Unsupported datasets: {invalid_datasets}")
    requested_dataset_keys = [
        name.strip().lower()
        for name in args.datasets.split(",")
        if name.strip()
    ]
    all_datasets = prepare_datasets(
        args.data_dir, args.num_cases, requested_dataset_keys
    )
    datasets = {
        dataset_names[key]: all_datasets[dataset_names[key]]
        for key in requested_dataset_keys
    }
    results = []
    for dataset, prompts in datasets.items():
        for batch_size in batch_sizes:
            full_count = len(prompts) // batch_size * batch_size
            batches = [
                prompts[start : start + batch_size]
                for start in range(0, full_count, batch_size)
            ]
            warmup = batches[0]
            for method in methods:
                run_batch(
                    args.family,
                    model,
                    tokenizer,
                    diffusion_generate,
                    diffusion_generate_PSD,
                    diffusion_generate_spiffy,
                    diffusion_generate_lopa,
                    warmup,
                    method,
                    args.draft_steps,
                    args.block_length,
                    args.max_new_tokens,
                    args.alg,
                    args.threshold,
                    mask_token_id,
                    k_branches=args.k_branches, lopa_strategy=args.lopa_strategy,
                    cache_sharing=args.cache_sharing,
                    seed=args.seed,
                    graph_variant=args.graph_variant,
                    dataset_key=next(key for key, label in dataset_names.items() if label == dataset),
                )
            for batch_index, prompts_batch in enumerate(batches):
                for method in methods:
                    result = run_batch(
                        args.family,
                        model,
                        tokenizer,
                        diffusion_generate,
                        diffusion_generate_PSD,
                        diffusion_generate_spiffy,
                        diffusion_generate_lopa,
                        prompts_batch,
                        method,
                        args.draft_steps,
                        args.block_length,
                        args.max_new_tokens,
                        args.alg,
                        args.threshold,
                        mask_token_id,
                        k_branches=args.k_branches, lopa_strategy=args.lopa_strategy,
                        cache_sharing=args.cache_sharing,
                        seed=args.seed,
                        graph_variant=args.graph_variant,
                    dataset_key=next(key for key, label in dataset_names.items() if label == dataset),
                    )
                    result.update({"dataset": dataset, "batch_index": batch_index})
                    results.append(result)
                write_report(args, batch_sizes, methods, results)

    markdown = write_report(args, batch_sizes, methods, results)
    print(markdown)


if __name__ == "__main__":
    main()
