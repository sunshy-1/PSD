from src.stopping import EOSState, has_eos, active_token_mask, generation_policy
from src import psd as _psd
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import time
try:
    import torch_npu
except ImportError as e:
    pass
import torch.distributions as dists
from torch.nn import functional as F


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

# 返回每个位置置信度最高的token及对应的confidence
def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


def expand_cache(past_key_values: list, n: int, share: bool = False):
    """将 Cache 在 Batch 维扩展 n 倍"""
    new_past_key_values = []
    for i in range(len(past_key_values)):
        new_past_key_values.append(())
        for j in range(len(past_key_values[i])):
            # [Batch, ...] -> [Batch*N, ...]
            cache_tensor = past_key_values[i][j]
            if share and cache_tensor.shape[0] == 1:
                expanded_tensor = cache_tensor.expand(n, *cache_tensor.shape[1:])
            else:
                expanded_tensor = cache_tensor.repeat_interleave(n, dim=0)
            new_past_key_values[i] += (expanded_tensor,)
    
    return new_past_key_values


# 列表中的每个元素是一个字典，代表一个 Draft Node
# 'indices': 需要 unmask 的 token 在 sorted_confidence 中的索引 (对应论文中的 c_ij 的 i)
# 'level': 该节点在图中的深度 (用于计算加速步数)
# 'parents': 该节点的父节点索引列表 (-1 代表 Root/当前状态)


def local_leap_transfer_index(confidence, threshold=0.9, radius=4, relax_threshold=0.75):
    anchor_mask = confidence >= threshold
    anchor_neighborhood = F.max_pool1d(
        anchor_mask.to(dtype=torch.float32).unsqueeze(1),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    ).squeeze(1).bool()
    return anchor_neighborhood & (confidence >= relax_threshold)


def has_complete_eos_prefix(
    x, prompt_length, eos_token_id, mask_token_id, stop_on_any_eos=False
):
    return has_eos(x, prompt_length, eos_token_id, stop_on_any_eos, mask_token_id)


@torch.no_grad()
def token_transfer_one_step(
    x: torch.Tensor,            # [B, L] 当前的输入序列（包含 Prompt + 正在生成的 Block，未生成部分是 MASK）
    x0: torch.Tensor,           # [B, block_size] 模型预测的block内部的“全局完整答案”（即模型认为把block MASK 填满后长什么样）
    confidence: torch.Tensor,   # [B, block_size] 模型对 x0 中每个 token 的置信度分数
    threshold: float,
    mask_token_id: int,
    block_start: int,
    block_end: int,
    alg: str,
    ):
        
    x_new = x.clone()
    cur_tokens = x_new[:, block_start:block_end]
    mask_index = active_token_mask(cur_tokens, mask_token_id)
    confidence = torch.where(mask_index, confidence, -torch.inf)
    transfer_index = (F.one_hot(torch.max(confidence, dim=1)[1], num_classes=confidence.shape[1]) == 1)


    if alg == 'confidence_threshold':
        transfer_index |= (confidence > threshold)
        
    elif alg == "local_leap":
        transfer_index |= local_leap_transfer_index(confidence)   

    transfer_index &= mask_index
    cur_tokens[transfer_index] = x0[transfer_index]

    # 链式索引赋值 a[slice][mask] = b 有时无法按预期修改原始 a，因为 a[slice] 可能产生了一个临时对象

    # x[:, block_start:block_end][transfer_index] = x0[transfer_index]

    num_masks_per_batch = mask_index.sum(dim=1)
    num_new_tokens_per_batch = transfer_index.sum(dim=1)
    last_step_in_block = (num_new_tokens_per_batch >= num_masks_per_batch).all()
    return x_new, last_step_in_block 

@torch.no_grad()
def token_transfer_graph(
    x: torch.Tensor,            # [B, L] 当前的输入序列（包含 Prompt + 正在生成的 Block，未生成部分是 MASK）
    x0: torch.Tensor,           # [B, block_size] 模型预测的block内部的“全局完整答案”（即模型认为把block MASK 填满后长什么样）
    confidence: torch.Tensor,   # [B, block_size] 模型对 x0 中每个 token 的置信度分数
    mask_token_id: int,
    block_start: int,
    block_end: int,
    graph_structure: List[Dict] = None,
    graph_rank_indices=None,
    graph_rank_valid=None,
    **kwargs,
    ):

    blocks = _psd.draft_blocks(x, x0, confidence, mask_token_id,
                               block_start, block_end, graph_structure)
    batch, count = x.shape[0], len(graph_structure)
    drafts = x[:, None].expand(batch, count, x.shape[-1]).clone()
    drafts[:, :, block_start:block_end] = blocks.reshape(batch, count, block_end-block_start)
    return drafts.reshape(batch * count, -1)


@torch.no_grad()
def token_transfer_graph_blocks(
    x: torch.Tensor,
    x0: torch.Tensor,
    confidence: torch.Tensor,
    mask_token_id: int,
    block_start: int,
    block_end: int,
    graph_structure: List[Dict],
    graph_rank_indices=None,
    graph_rank_valid=None,
) -> torch.Tensor:
    return _psd.draft_blocks(x, x0, confidence, mask_token_id,
                             block_start, block_end, graph_structure, graph_rank_indices, graph_rank_valid)


def get_graph_structure(draft_steps, graph_variant="dag"):
    return _psd.get_graph_structure(draft_steps, graph_variant)


def prepare_graph_metadata(graph_structure, device):
    ranks, valid = _psd.rank_metadata(graph_structure, device)
    parents = [torch.tensor(n["parents"], dtype=torch.long, device=device) for n in graph_structure]
    return ranks, valid, parents


def graph_accept_reject(
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    confidence: torch.Tensor,
    x_draft_flat: torch.Tensor,
    draft_x0: torch.Tensor,
    draft_conf: torch.Tensor,
    graph_structure: List[Dict],
    threshold: float,
    mask_token_id: int,
    block_start: int,
    block_end: int,
    alg: str,
    graph_parent_indices=None,
    eos_token_id=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _psd.accept_full(x, x0_pred, confidence, x_draft_flat,
                            draft_x0, draft_conf, graph_structure, threshold,
                            mask_token_id, block_start, block_end, alg,
                            token_transfer_one_step, eos_token_id=eos_token_id)


def graph_accept_reject_blocks(
    x: torch.Tensor,
    x0_pred: torch.Tensor,
    confidence: torch.Tensor,
    draft_blocks_flat: torch.Tensor,
    draft_x0: torch.Tensor,
    draft_conf: torch.Tensor,
    graph_structure: List[Dict],
    threshold: float,
    mask_token_id: int,
    block_start: int,
    block_end: int,
    alg: str,
    graph_parent_indices=None,
    acceptance_mode="hierarchical",
    acceptance_stats=None,
    eos_token_id=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _psd.accept_blocks(x, x0_pred, confidence, draft_blocks_flat,
                              draft_x0, draft_conf, graph_structure, threshold,
                              mask_token_id, block_start, block_end, alg,
                              token_transfer_one_step, acceptance_mode, acceptance_stats, eos_token_id=eos_token_id)


def model_forward(
    model,
    block_input,
    past_key_values,
    cur_attn_mask,
    cur_position_ids,
    cur_draft_steps,
    temperature,
    top_p,
    top_k,
    block_length,
    dual_cache,
    replace_position=None,
    expanded_past_key_values=None,
    expanded_replace_position=None,
):

    if cur_draft_steps == 1:
        kv_cache = past_key_values
    else:
        kv_cache = expanded_past_key_values
        if kv_cache is None:
            kv_cache = expand_cache(past_key_values, cur_draft_steps)
    
    attn_mask = cur_attn_mask.repeat_interleave(cur_draft_steps, dim=0)
    # input shape [B*K, Block Size]
    if dual_cache:
        replace_position_dual_cache = expanded_replace_position
        if replace_position_dual_cache is None:
            replace_position_dual_cache = replace_position.repeat_interleave(cur_draft_steps, dim=0)
        output = model(input_ids=block_input,
                attention_mask=attn_mask,


                past_key_values=kv_cache,
                use_cache=True,

                replace_position=replace_position_dual_cache)
    else:
        output = model(input_ids=block_input,
                attention_mask=attn_mask,


                past_key_values=kv_cache,
                use_cache=True

            )
    
    logits = output.logits

    logits = logits[:, 0:block_length]
    conf, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
    return x0, conf


@generation_policy
@torch.no_grad()
def diffusion_generate_PSD(
    model,
    inputs: Optional[torch.Tensor] = None,
    top_p: Optional[int] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = 0.9,
    draft_steps: int = 7,
    alg: str = None,
    return_decode_steps: bool = False,
    **kwargs,
):  
    """Native block decoding with spatial transfer and batched DAG verification.

    ``draft_steps`` is speculative depth after the empty root. Verifier outputs
    for the selected accepted node are reused by the next spatial transfer.
    """

    block_length = kwargs.pop("block_length", 32)
    attention_mask = kwargs.pop("attention_mask", None)
    trace_buffer = kwargs.get("trace_buffer", None)
    temperature = kwargs.get("temperature", 0.0)
    mask_token_id = kwargs.get("mask_token_id", None)
    eos_token_id = kwargs.get("eos_token_id", None)
    stop_on_any_eos = kwargs.get("stop_on_any_eos", False)


    dual_cache = kwargs.get("dual_cache", False)
    graph_variant = kwargs.get("graph_variant", "dag")
    acceptance_mode = kwargs.get("acceptance_mode", "hierarchical")
    acceptance_stats = kwargs.get("acceptance_stats")
    acceptance_feedback = acceptance_stats
    if acceptance_mode == "deepest_only" and acceptance_feedback is None:
        acceptance_feedback = {}
    cache_sharing = kwargs.get("cache_sharing", False)


    assert mask_token_id is not None
    assert inputs is not None
    assert attention_mask is not None

    graph_structure = get_graph_structure(draft_steps, graph_variant)
    num_drafts = len(graph_structure)

    input_ids = inputs
    if type(kwargs.get('max_new_tokens', None)) is int:
        max_length = kwargs.get('max_new_tokens') + input_ids.shape[-1]
    else:
        max_length = kwargs.get('max_length', 2048)

    prompt_length = input_ids.shape[1]
    if block_length < 1 or max_length < prompt_length or (max_length - prompt_length) % block_length:
        raise ValueError("generation length must be nonnegative and divisible by positive block_length")
    num_blocks = (max_length - prompt_length) // block_length
    device = model.device
    graph_rank_indices, graph_rank_valid, graph_parent_indices = prepare_graph_metadata(
        graph_structure, device
    )

    # Pad input
    x = F.pad(input_ids, (0, max_length - prompt_length), value=mask_token_id)
    eos_state = EOSState(x, prompt_length, eos_token_id, mask_token_id,
                         kwargs.get("stop_on_any_eos", False))
    if max_length == prompt_length:
        return (x, 0) if return_decode_steps else x
    

    past_key_values = None
    padding_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
    position_ids = padding_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(padding_mask == 0, 1)
    # [B, N] --> [B, 1, N, N]
    padding_mask = torch.logical_and(
        padding_mask.unsqueeze(1).unsqueeze(-2),
        padding_mask.unsqueeze(1).unsqueeze(-1),
    )
    attention_mask = padding_mask


    start_time = time.time()
    decode_step=0
    for num_block in range(num_blocks):
        

        # Prepare masks and position ids

        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length


        # update cache
        if eos_state.update(x):
            return (x, decode_step) if return_decode_steps else x
        model_output = model(input_ids=x, attention_mask=attention_mask, use_cache=True)
        past_key_values = model_output.past_key_values
        logits = model_output.logits

        confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        # x[:, block_start] = x0[:, block_start]
        # decode_step += 1

        mask_index = eos_state.mask(x, 0, x.shape[-1])
        mask_index &= ~eos_state.finished[:, None]
        mask_index[:, block_end:] = 0

        confidence = torch.where(mask_index, confidence, -torch.inf)
        transfer_index = (F.one_hot(torch.max(confidence, dim=1)[1], num_classes=confidence.shape[-1]) == 1)

        if alg == 'confidence_threshold':
            transfer_index |= (confidence > threshold)
        elif alg == "local_leap":
            transfer_index |= local_leap_transfer_index(confidence)   
        transfer_index &= mask_index
        x[transfer_index] = x0[transfer_index]
        decode_step += 1
        if trace_buffer is not None:
            trace_buffer.append(x.detach().cpu().clone())

        if not eos_state.mask(x, block_start, block_end).any():
            if eos_state.update(x):
                if return_decode_steps:
                    return x, decode_step
                return x
            continue
        
        # Extract only previous block cache
        if not dual_cache:
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :, :block_start],)
            past_key_values = new_past_key_values
        else:
            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, block_start:block_end] = 1


        if dual_cache:
            cur_x_input = x[:, block_start:block_end]
            cur_attn_mask = attention_mask[:, :, block_start:block_end, :]
            cur_position_ids = position_ids[:, block_start:block_end]
        else:
            cur_x_input = x[:, block_start:]
            cur_attn_mask = attention_mask[:, :, block_start:, :]
            cur_position_ids = position_ids[:, block_start:]


        if dual_cache:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            output = model(input_ids=cur_x_input,
                    attention_mask=cur_attn_mask,


                    past_key_values=past_key_values,
                    use_cache=True,

                    replace_position=replace_position)
        else:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            output = model(input_ids=cur_x_input,
                    attention_mask=cur_attn_mask,


                    past_key_values=past_key_values,
                    use_cache=True

                    )


        logits = output.logits

        
        logits = logits[:, 0:block_length]


        decode_step += 1
        
        # shape: (B, L)
        confidence, x0_pred = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        expanded_past_key_values = None
        expanded_replace_position = None
        

        while True:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            if not eos_state.mask(x, block_start, block_end).any():
                break
            

            # 1. One-step transfer

            x, last_step_in_block = token_transfer_one_step(
                x=x,
                x0=x0_pred, 
                confidence=confidence, 
                threshold=threshold,
                mask_token_id=mask_token_id,
                block_start=block_start,
                block_end=block_end,
                alg=alg
            )
            if eos_state.update(x):
                if return_decode_steps:
                    return x, decode_step
                return x
            if trace_buffer is not None:
                trace_buffer.append(x.detach().cpu().clone())

            # [Early Termination] if is the last decoding step in the current block
            if last_step_in_block:
                break
            

            if num_drafts == 1:
                if dual_cache:
                    block_input = x[:, block_start:block_end]
                    x0_pred, confidence = model_forward(model, block_input, 
                            past_key_values, cur_attn_mask, cur_position_ids, num_drafts, temperature, top_p, top_k, block_length, dual_cache, replace_position)
                    decode_step += 1
                    continue
                else:
                    block_input = x[:, block_start:]
                    x0_pred, confidence = model_forward(model, block_input, 
                            past_key_values, cur_attn_mask, cur_position_ids, num_drafts, temperature, top_p, top_k, block_length, dual_cache, replace_position=None)
                    decode_step += 1
                    continue


            # 2. Drafting (Graph-based)


            if dual_cache:
                draft_blocks_flat = token_transfer_graph_blocks(
                    x=x,
                    x0=x0_pred,
                    confidence=confidence,
                    mask_token_id=mask_token_id,
                    block_start=block_start,
                    block_end=block_end,
                    graph_structure=graph_structure,
                    graph_rank_indices=graph_rank_indices,
                    graph_rank_valid=graph_rank_valid,
                )
            else:
                x_draft_flat = token_transfer_graph(
                    x=x,
                    x0=x0_pred,
                    confidence=confidence,
                    mask_token_id=mask_token_id,
                    block_start=block_start,
                    block_end=block_end,
                    graph_structure=graph_structure,
                    graph_rank_indices=graph_rank_indices,
                    graph_rank_valid=graph_rank_valid,
                )
            
            
            # 3. Parallel Verification

            if expanded_past_key_values is None:
                expanded_past_key_values = expand_cache(
                    past_key_values, num_drafts, share=cache_sharing
                )
                if dual_cache:
                    expanded_replace_position = replace_position.repeat_interleave(num_drafts, dim=0)
            

            if dual_cache:
                block_input = draft_blocks_flat
                draft_x0, draft_conf = model_forward(model, block_input, 
                        past_key_values, cur_attn_mask, cur_position_ids, num_drafts, temperature, top_p, top_k,
                        block_length, dual_cache, replace_position, expanded_past_key_values,
                        expanded_replace_position)
                decode_step += 1
            else:
                block_input = x_draft_flat[:, block_start:]
                draft_x0, draft_conf = model_forward(model, block_input, 
                        past_key_values, cur_attn_mask, cur_position_ids, num_drafts, temperature, top_p, top_k,
                        block_length, dual_cache, replace_position=None,
                        expanded_past_key_values=expanded_past_key_values)
                decode_step += 1


            # 4. Graph-based Accept / Reject Logic

            if dual_cache:
                x, x0_pred, confidence = graph_accept_reject_blocks(
                    x=x,
                    x0_pred=x0_pred,
                    confidence=confidence,
                    draft_blocks_flat=draft_blocks_flat,
                    draft_x0=draft_x0,
                    draft_conf=draft_conf,
                    graph_structure=graph_structure,
                    threshold=threshold,
                    mask_token_id=mask_token_id,
                    block_start=block_start,
                    block_end=block_end,
                    alg=alg,
                    graph_parent_indices=graph_parent_indices,
                    acceptance_mode=acceptance_mode,
                    acceptance_stats=acceptance_feedback,
                    eos_token_id=eos_token_id,
                )
                if acceptance_mode == "deepest_only" and not acceptance_feedback.pop(
                    "_last_deepest_success", False
                ):
                    recovery_input = x[:, block_start:block_end]
                    x0_pred, confidence = model_forward(
                        model,
                        recovery_input,
                        past_key_values,
                        cur_attn_mask,
                        cur_position_ids,
                        1,
                        temperature,
                        top_p,
                        top_k,
                        block_length,
                        dual_cache,
                        replace_position,
                    )
                    decode_step += 1
                    acceptance_feedback["recovery_forward_calls"] = (
                        acceptance_feedback.get("recovery_forward_calls", 0) + 1
                    )
            else:
                x, x0_pred, confidence = graph_accept_reject(
                    x=x,
                    x0_pred=x0_pred,
                    confidence=confidence,
                    x_draft_flat=x_draft_flat,
                    draft_x0=draft_x0,
                    draft_conf=draft_conf,
                    graph_structure=graph_structure,
                    threshold=threshold,
                    mask_token_id=mask_token_id,
                    block_start=block_start,
                    block_end=block_end,
                    alg=alg,
                    graph_parent_indices=graph_parent_indices,
                    eos_token_id=eos_token_id,
                )
            if eos_state.update(x):
                if return_decode_steps:
                    return x, decode_step
                return x
            if trace_buffer is not None:
                trace_buffer.append(x.detach().cpu().clone())


        # EOS early termination

        if eos_state.update(x):
            total_time = time.time() - start_time


            if return_decode_steps:
                return x, decode_step
            else:
                return x       


    total_time = time.time() - start_time


    if return_decode_steps:
        return x, decode_step
    else:
        return x


@generation_policy
def diffusion_generate_Spiffy(model, inputs=None, top_p=None, top_k=None,
                              draft_steps=7, return_decode_steps=False, **kwargs):
    """Deterministic lossless Spiffy: PSD verification with a one-token Greedy policy."""
    kwargs.pop("alg", None)
    kwargs.pop("threshold", None)
    kwargs.pop("spiffy_mode", None)
    if kwargs.get("temperature", 0.0) != 0:
        raise ValueError("Spiffy requires temperature=0 for greedy-equivalent verification")
    return diffusion_generate_PSD(
        model, inputs=inputs, top_p=top_p, top_k=top_k,
        threshold=0.0, draft_steps=draft_steps, alg="greedy",
        return_decode_steps=return_decode_steps, **kwargs,
    )
