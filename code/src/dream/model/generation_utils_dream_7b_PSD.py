from src.stopping import EOSState, has_eos, active_token_mask, generation_policy
from src import psd as _psd
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.distributed as dist
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


def sample_shifted_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    neg_entropy=False,
):
    if temperature == 0.0:
        source_logits = logits[:, :-1] if logits.shape[1] > 1 else logits
        confidence, x0 = sample_tokens(
            source_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            neg_entropy=neg_entropy,
        )
        if logits.shape[1] > 1:
            confidence = torch.cat([confidence[:, :1], confidence], dim=1)
            x0 = torch.cat([x0[:, :1], x0], dim=1)
        return confidence, x0
    shifted_logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return sample_tokens(
        shifted_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        neg_entropy=neg_entropy,
    )


def expand_cache(
    past_key_values: list,
    n: int,
    materialize: bool = False,
    workspace: Optional[list] = None,
):
    """在 Batch 维共享或复制 KV cache。"""
    new_past_key_values = []
    for i in range(len(past_key_values)):
        new_past_key_values.append(())
        for j in range(len(past_key_values[i])):
            cache_tensor = past_key_values[i][j]
            if cache_tensor.shape[0] == 1:
                expanded_tensor = cache_tensor.expand(n, *cache_tensor.shape[1:])
                if materialize:
                    workspace_tensor = None
                    if workspace is not None:
                        workspace_tensor = workspace[i][j]
                    if (
                        workspace_tensor is not None
                        and workspace_tensor.shape == expanded_tensor.shape
                        and workspace_tensor.dtype == expanded_tensor.dtype
                        and workspace_tensor.device == expanded_tensor.device
                    ):
                        workspace_tensor.copy_(expanded_tensor)
                        expanded_tensor = workspace_tensor
                    else:
                        expanded_tensor = expanded_tensor.clone()
            else:
                expanded_tensor = cache_tensor.repeat_interleave(n, dim=0)
            new_past_key_values[i] += (expanded_tensor,)
    
    return new_past_key_values


def _draft_parallel_info(enabled: bool, num_drafts: int) -> Tuple[int, int, int, int]:
    if not enabled or not dist.is_available() or not dist.is_initialized():
        return 0, 1, 0, num_drafts

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    start = num_drafts * rank // world_size
    end = num_drafts * (rank + 1) // world_size
    return rank, world_size, start, end


def _select_local_drafts(
    x_draft_flat: torch.Tensor,
    batch_size: int,
    num_drafts: int,
    start: int,
    end: int,
) -> torch.Tensor:
    return x_draft_flat.view(batch_size, num_drafts, -1)[:, start:end].reshape(
        batch_size * (end - start), -1
    )


def _gather_draft_results(
    local_blocks: torch.Tensor,
    local_x0: torch.Tensor,
    local_conf: torch.Tensor,
    batch_size: int,
    num_drafts: int,
    workspace: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size()
    counts = [
        num_drafts * (rank + 1) // world_size - num_drafts * rank // world_size
        for rank in range(world_size)
    ]
    local_count = counts[dist.get_rank()]
    max_count = max(counts)
    block_length = local_blocks.shape[-1]

    local_blocks = local_blocks.view(batch_size, local_count, block_length)
    local_x0 = local_x0.view(batch_size, local_count, block_length)
    local_conf = local_conf.view(batch_size, local_count, block_length)
    shape_key = (batch_size, max_count, block_length, world_size)
    if workspace.get("shape_key") != shape_key:
        workspace.clear()
        workspace["shape_key"] = shape_key
        workspace["local_pack"] = torch.zeros(
            (batch_size, max_count, block_length, 3),
            dtype=torch.float32,
            device=local_x0.device,
        )
        workspace["gathered_pack"] = torch.empty(
            (world_size * batch_size, max_count, block_length, 3),
            dtype=torch.float32,
            device=local_x0.device,
        )
        workspace["full_blocks"] = torch.empty(
            (batch_size, num_drafts, block_length),
            dtype=local_blocks.dtype,
            device=local_blocks.device,
        )
        workspace["full_x0"] = torch.empty(
            (batch_size, num_drafts, block_length),
            dtype=local_x0.dtype,
            device=local_x0.device,
        )
        workspace["full_conf"] = torch.empty(
            (batch_size, num_drafts, block_length),
            dtype=local_conf.dtype,
            device=local_conf.device,
        )

    local_pack = workspace["local_pack"]
    local_pack[:, :local_count, :, 0].copy_(local_blocks)
    local_pack[:, :local_count, :, 1].copy_(local_x0)
    local_pack[:, :local_count, :, 2].copy_(local_conf)
    dist.all_gather_into_tensor(workspace["gathered_pack"], local_pack)

    gathered_pack = workspace["gathered_pack"].view(
        world_size, batch_size, max_count, block_length, 3
    )
    full_blocks = workspace["full_blocks"]
    full_x0 = workspace["full_x0"]
    full_conf = workspace["full_conf"]
    offset = 0
    for rank, count in enumerate(counts):
        full_blocks[:, offset:offset + count].copy_(
            gathered_pack[rank, :, :count, :, 0]
        )
        full_x0[:, offset:offset + count].copy_(
            gathered_pack[rank, :, :count, :, 1]
        )
        full_conf[:, offset:offset + count].copy_(
            gathered_pack[rank, :, :count, :, 2]
        )
        offset += count
    return (
        full_blocks.reshape(batch_size * num_drafts, block_length),
        full_x0.reshape(batch_size * num_drafts, block_length),
        full_conf.reshape(batch_size * num_drafts, block_length),
    )


def _broadcast_base_predictions(
    x0: torch.Tensor,
    confidence: torch.Tensor,
    enabled: bool,
    workspace: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if enabled and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        shape_key = tuple(x0.shape)
        if workspace.get("shape_key") != shape_key:
            workspace.clear()
            workspace["shape_key"] = shape_key
            workspace["pack"] = torch.empty(
                (*x0.shape, 2), dtype=torch.float32, device=x0.device
            )
        pack = workspace["pack"]
        if dist.get_rank() == 0:
            pack[..., 0].copy_(x0)
            pack[..., 1].copy_(confidence)
        dist.broadcast(pack, src=0)
        x0.copy_(pack[..., 0])
        confidence.copy_(pack[..., 1])
    return x0, confidence


def _broadcast_block_token(x0: torch.Tensor, block_start: int, enabled: bool) -> None:
    if enabled and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        token = x0[:, block_start].contiguous()
        dist.broadcast(token, src=0)
        x0[:, block_start].copy_(token)


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


def select_focus_targets(
    confidence: torch.Tensor,
    block_tokens: torch.Tensor,
    mask_token_id: int,
    committed_count: int,
    expansion_factor: float,
    window_size: int,
    min_candidates: int = 1,
) -> torch.Tensor:
    if confidence.shape[0] != 1:
        raise ValueError("Focus-dLLM reference path currently supports batch size 1")
    masked = block_tokens.eq(mask_token_id)
    remaining = int(masked.sum().item())
    if remaining == 0:
        return torch.empty(0, dtype=torch.long, device=confidence.device)
    candidate_count = max(
        min_candidates,
        int(torch.ceil(torch.tensor(expansion_factor * max(committed_count, 1))).item()),
    )
    candidate_count = min(candidate_count, remaining)
    masked_confidence = torch.where(masked, confidence, -torch.inf)
    focus_indices = torch.topk(masked_confidence[0], candidate_count).indices
    radius = window_size // 2
    offsets = torch.arange(-radius, window_size - radius, device=confidence.device)
    expanded = focus_indices[:, None] + offsets[None, :]
    expanded = expanded.clamp_(0, block_tokens.shape[-1] - 1)
    return torch.unique(expanded.reshape(-1), sorted=True)


def dream_focus_query_indices(target_indices: torch.Tensor) -> torch.Tensor:
    if target_indices.numel() == 0:
        return target_indices
    predecessor_indices = torch.clamp(target_indices - 1, min=0)
    return torch.unique(
        torch.cat([target_indices, predecessor_indices]), sorted=True
    )


@torch.no_grad()
def focus_model_forward(
    model,
    block_input: torch.Tensor,
    fallback_x0: torch.Tensor,
    fallback_confidence: torch.Tensor,
    past_key_values,
    cur_attn_mask: Optional[torch.Tensor],
    target_indices: torch.Tensor,
    block_start: int,
    block_length: int,
    temperature: float,
    top_p,
    top_k,
    alg: Optional[str],
    focus_stats: Optional[Dict[str, Any]] = None,
    expanded_past_key_values=None,
):
    query_indices = dream_focus_query_indices(target_indices)
    if query_indices.numel() == 0:
        return fallback_x0.clone(), fallback_confidence.clone()

    focused_input = block_input.index_select(1, query_indices)
    focused_attn_mask = cur_attn_mask
    if focused_attn_mask is not None:
        focused_attn_mask = focused_attn_mask.index_select(2, query_indices)
    replace_indices = block_start + query_indices
    output = model(
        focused_input,
        attention_mask=focused_attn_mask,
        past_key_values=(
            expanded_past_key_values
            if expanded_past_key_values is not None
            else past_key_values
        ),
        use_cache=False,
        dual_cache=True,
        replace_indices=replace_indices,
    )
    raw_confidence, raw_x0 = sample_tokens(
        output.logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        neg_entropy=(alg == "entropy_threshold"),
    )
    focused_x0 = fallback_x0.clone()
    focused_confidence = fallback_confidence.clone()
    query_lookup = {
        int(query_index): slot
        for slot, query_index in enumerate(query_indices.tolist())
    }
    for target_index in target_indices.tolist():
        source_index = max(target_index - 1, 0)
        source_slot = query_lookup[source_index]
        focused_x0[:, target_index] = raw_x0[:, source_slot]
        focused_confidence[:, target_index] = raw_confidence[:, source_slot]

    if focus_stats is not None:
        batch_size = block_input.shape[0]
        focus_stats["focused_forward_calls"] = (
            focus_stats.get("focused_forward_calls", 0) + 1
        )
        focus_stats["query_tokens"] = focus_stats.get("query_tokens", 0) + (
            batch_size * int(query_indices.numel())
        )
        focus_stats["full_query_tokens"] = focus_stats.get(
            "full_query_tokens", 0
        ) + batch_size * block_length
    return focused_x0, focused_confidence


def has_complete_eos_prefix(
    x, prompt_length, eos_token_id, mask_token_id, stop_on_any_eos=False
):
    return has_eos(x, prompt_length, eos_token_id, stop_on_any_eos, mask_token_id)


def active_mask_to_first_eos(x, prompt_length, eos_token_id, mask_token_id):
    return active_token_mask(x[:, prompt_length:], mask_token_id, eos_token_id)


def mask_confidence_after_first_eos(confidence, x, prompt_length, block_start, block_end, eos_token_id):
    generated = x[:, prompt_length:]
    eos_mask = generated == eos_token_id
    has_eos = eos_mask.any(dim=-1)
    first_eos = eos_mask.to(torch.int64).argmax(dim=-1)
    block_positions = torch.arange(block_start, block_end, device=x.device) - prompt_length
    active_window = ~has_eos[:, None] | (block_positions[None, :] < first_eos[:, None])
    return torch.where(active_window, confidence, -torch.inf)


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
    inplace: bool = False,
    ):

    x_new = x if inplace else x.clone()
    cur_tokens = x_new[:, block_start:block_end]
    mask_index = active_token_mask(cur_tokens, mask_token_id)
    confidence = torch.where(mask_index, confidence, -torch.inf)
    transfer_index = (F.one_hot(torch.max(confidence, dim=1)[1], num_classes=confidence.shape[1]) == 1)


    if alg == 'confidence_threshold':
        transfer_index |= (confidence > threshold)
    elif alg == 'entropy_threshold':
        transfer_index |= (confidence > -threshold)
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
    **kwargs,
    ):

    blocks = _psd.draft_blocks(x, x0, confidence, mask_token_id,
                               block_start, block_end, graph_structure)
    batch, count = x.shape[0], len(graph_structure)
    drafts = x[:, None].expand(batch, count, x.shape[-1]).clone()
    drafts[:, :, block_start:block_end] = blocks.reshape(batch, count, block_end-block_start)
    return drafts.reshape(batch * count, -1)


@torch.no_grad()
def token_transfer_graph_local_blocks(
    x: torch.Tensor,
    x0: torch.Tensor,
    confidence: torch.Tensor,
    mask_token_id: int,
    block_start: int,
    block_end: int,
    graph_structure: List[Dict],
    draft_start: int,
    draft_end: int,
    graph_rank_offsets=None,
) -> torch.Tensor:
    return _psd.draft_blocks(x, x0, confidence, mask_token_id,
                             block_start, block_end, graph_structure[draft_start:draft_end])


def get_graph_structure(draft_steps, graph_variant="dag"):
    return _psd.get_graph_structure(draft_steps, graph_variant)


def prepare_graph_metadata(graph_structure, device):
    rank_offsets = [
        torch.as_tensor(
            [index - 1 for index in node["indices"]],
            dtype=torch.long,
            device=device,
        )
        for node in graph_structure
    ]
    edge_order = [
        (child_index, parent_index)
        for child_index, node in enumerate(graph_structure)
        for parent_index in node["parents"]
        if parent_index >= 0
    ]
    edge_lookup = {edge: index for index, edge in enumerate(edge_order)}
    return rank_offsets, edge_order, edge_lookup


def prepare_spiffy_rank_indices(graph_structure, device):
    return [
        torch.as_tensor(node["indices"], dtype=torch.long, device=device)
        for node in graph_structure
    ]


@torch.no_grad()
def spiffy_draft_blocks(
    x,
    x0,
    confidence,
    mask_token_id,
    block_start,
    block_end,
    graph_structure,
    rank_indices,
):
    return _psd.draft_blocks(x, x0, confidence, mask_token_id,
                             block_start, block_end, graph_structure)


@torch.no_grad()
def spiffy_accept_reject_blocks(
    x,
    x0_pred,
    confidence,
    draft_blocks_flat,
    draft_x0,
    draft_conf,
    graph_structure,
    mask_token_id,
    block_start,
    block_end,
    graph_edge_order,
    graph_edge_lookup,
):
    return _psd.accept_blocks(x, x0_pred, confidence, draft_blocks_flat,
                              draft_x0, draft_conf, graph_structure, 0.0,
                              mask_token_id, block_start, block_end, "greedy",
                              token_transfer_one_step)


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
    graph_edge_order=None,
    graph_edge_lookup=None,
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
    replace_indices=None,
    expanded_past_key_values=None,
    expanded_attn_mask=None,
    expanded_position_ids=None,
    alg=None,
):

    if cur_draft_steps == 1:
        kv_cache, attn_mask, pos_ids = past_key_values, cur_attn_mask, cur_position_ids
    else:
        kv_cache = expanded_past_key_values
        if kv_cache is None:
            kv_cache = expand_cache(past_key_values, cur_draft_steps, materialize=dual_cache)
        attn_mask = expanded_attn_mask
        if attn_mask is None:
            attn_mask = (
                None
                if cur_attn_mask is None
                else cur_attn_mask.repeat_interleave(cur_draft_steps, dim=0)
            )
        pos_ids = expanded_position_ids
        if pos_ids is None:
            pos_ids = cur_position_ids.repeat_interleave(cur_draft_steps, dim=0)

    # input shape [B*K, Block Size]
    if dual_cache:
        output = model(block_input,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=kv_cache,
                use_cache=False,
                dual_cache=True,
                replace_position=replace_position,
                replace_indices=replace_indices)
    else:
        output = model(block_input,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_values=kv_cache,
                use_cache=False,
                dual_cache=False)
    
    logits = output.logits[:, :block_length]
    conf, x0 = sample_shifted_tokens(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        neg_entropy=(alg == "entropy_threshold"),
    )
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
    temperature = kwargs.get("temperature", 0.0)
    mask_token_id = kwargs.get("mask_token_id", None)
    eos_token_id = kwargs.get("eos_token_id", None)
    stop_on_any_eos = kwargs.get("stop_on_any_eos", False)


    dual_cache = kwargs.get("dual_cache", False)
    use_context_parallel_opener = kwargs.get("context_parallel_opener", False)
    draft_parallel = kwargs.get("draft_parallel", False)
    graph_variant = kwargs.get("graph_variant", "dag")
    acceptance_mode = kwargs.get("acceptance_mode", "hierarchical")
    acceptance_stats = kwargs.get("acceptance_stats")
    acceptance_feedback = acceptance_stats
    if acceptance_mode == "deepest_only" and acceptance_feedback is None:
        acceptance_feedback = {}
    focus_enabled = kwargs.get("focus_enabled", False)
    focus_expansion_factor = kwargs.get("focus_expansion_factor", 4.0)
    focus_window_size = kwargs.get("focus_window_size", 8)
    focus_stats = kwargs.get("focus_stats")


    assert mask_token_id is not None
    assert inputs is not None
    assert attention_mask is not None
    if focus_enabled and not dual_cache:
        raise ValueError("Focus-dLLM-style selective updates require dual_cache=True")
    if focus_enabled and inputs.shape[0] != 1:
        raise ValueError("Focus-dLLM reference path currently supports batch size 1")

    graph_structure = get_graph_structure(draft_steps, graph_variant)
    num_drafts = len(graph_structure)
    _, draft_world_size, local_draft_start, local_draft_end = _draft_parallel_info(
        draft_parallel, num_drafts
    )
    local_num_drafts = local_draft_end - local_draft_start
    graph_rank_offsets, graph_edge_order, graph_edge_lookup = prepare_graph_metadata(
        graph_structure, model.device
    )
    draft_gather_workspace: Dict[str, Any] = {}
    base_broadcast_workspace: Dict[str, Any] = {}

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
    if bool(padding_mask.all().item()):
        attention_mask = None
    else:
        attention_mask = torch.logical_and(
            padding_mask.unsqueeze(1).unsqueeze(-2),
            padding_mask.unsqueeze(1).unsqueeze(-1),
        )


    start_time = time.time()
    decode_step=0
    expanded_cache_workspace = None
    for num_block in range(num_blocks):
        

        # Prepare masks and position ids

        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length


        # update cache
        opener_logit_index = torch.tensor([block_start - 1], dtype=torch.long, device=x.device)
        if use_context_parallel_opener:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            logits, past_key_values = context_parallel_opener(
                model, x, block_start - 1
            )
        else:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            model_output = model(
                x,
                attention_mask,
                position_ids,
                use_cache=True,
                logit_indices=opener_logit_index,
            )
            past_key_values = model_output.past_key_values
            logits = model_output.logits
        confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        _broadcast_block_token(x0, 0, draft_parallel)

        x[:, block_start] = x0[:, 0]
        decode_step += 1
        if eos_state.update(x):
            if return_decode_steps:
                return x, decode_step
            return x
        
        # Extract only previous block cache
        if not dual_cache:
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :block_start, :],)
            past_key_values = new_past_key_values
        else:
            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, block_start:block_end] = 1
            replace_indices = torch.arange(block_start, block_end, device=x.device)


        if dual_cache:
            cur_x_input = x[:, block_start:block_end]
            cur_attn_mask = None if attention_mask is None else attention_mask[:, :, block_start:block_end, :]
            cur_position_ids = position_ids[:, block_start:block_end]
        else:
            cur_x_input = x[:, block_start:]
            cur_attn_mask = None if attention_mask is None else attention_mask[:, :, block_start:, :]
            cur_position_ids = position_ids[:, block_start:]


        if dual_cache:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            output = model(cur_x_input,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=False,
                    dual_cache=True,
                    replace_position=replace_position,
                    replace_indices=replace_indices)
        else:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            output = model(cur_x_input,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    dual_cache=False)


        logits = output.logits[:, 0:block_length]


        decode_step += 1
        
        # shape: (B, L)
        confidence, x0_pred = sample_shifted_tokens(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            neg_entropy=(alg == "entropy_threshold"),
        )
        x0_pred, confidence = _broadcast_base_predictions(
            x0_pred, confidence, draft_parallel, base_broadcast_workspace
        )
        expanded_past_key_values = None
        expanded_attn_mask = None
        expanded_position_ids = None
        

        while True:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            active_block_mask = active_mask_to_first_eos(
                x, prompt_length, eos_token_id, mask_token_id
            )[:, block_start - prompt_length:block_end - prompt_length]
            if active_block_mask.sum() == 0:
                break

            
            # 1. One-step transfer

            confidence = mask_confidence_after_first_eos(
                confidence,
                x,
                prompt_length,
                block_start,
                block_end,
                eos_token_id,
            )
            mask_before_transfer = x[:, block_start:block_end].eq(mask_token_id)
            x, last_step_in_block = token_transfer_one_step(
                x=x,
                x0=x0_pred, 
                confidence=confidence, 
                threshold=threshold,
                mask_token_id=mask_token_id,
                block_start=block_start,
                block_end=block_end,
                alg=alg,
                inplace=True,
            )
            mask_after_transfer = x[:, block_start:block_end].eq(mask_token_id)
            committed_count = int(
                (mask_before_transfer & ~mask_after_transfer).sum().item()
            )
            focus_targets = None
            if focus_enabled:
                focus_targets = select_focus_targets(
                    confidence=confidence,
                    block_tokens=x[:, block_start:block_end],
                    mask_token_id=mask_token_id,
                    committed_count=committed_count,
                    expansion_factor=focus_expansion_factor,
                    window_size=focus_window_size,
                    min_candidates=max(
                        len(node["indices"]) for node in graph_structure
                    ),
                )
            if eos_state.update(x):
                if return_decode_steps:
                    return x, decode_step
                return x

            # [Early Termination] if is the last decoding step in the current block
            if last_step_in_block:
                break
            

            if num_drafts == 1:
                if dual_cache:
                    block_input = x[:, block_start:block_end]
                    if focus_enabled:
                        x0_pred, confidence = focus_model_forward(
                            model=model,
                            block_input=block_input,
                            fallback_x0=x0_pred,
                            fallback_confidence=confidence,
                            past_key_values=past_key_values,
                            cur_attn_mask=cur_attn_mask,
                            target_indices=focus_targets,
                            block_start=block_start,
                            block_length=block_length,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            alg=alg,
                            focus_stats=focus_stats,
                        )
                    else:
                        x0_pred, confidence = model_forward(model, block_input, 
                                past_key_values, cur_attn_mask, cur_position_ids, num_drafts, temperature, top_p, top_k,
                                block_length, dual_cache, replace_position, replace_indices, alg=alg)
                    decode_step += 1
                    continue
                else:
                    block_input = x[:, block_start:]
                    x0_pred, confidence = model_forward(model, block_input, 
                            past_key_values, cur_attn_mask, cur_position_ids, num_drafts, temperature, top_p, top_k, block_length, dual_cache, replace_position=None, alg=alg)
                    decode_step += 1
                    continue


            # 2. Drafting (Graph-based)


            local_draft_blocks = token_transfer_graph_local_blocks(
                x=x,
                x0=x0_pred,
                confidence=confidence,
                mask_token_id=mask_token_id,
                block_start=block_start,
                block_end=block_end,
                graph_structure=graph_structure,
                draft_start=local_draft_start,
                draft_end=local_draft_end,
                graph_rank_offsets=graph_rank_offsets,
            )


            # 3. Parallel Verification

            if local_num_drafts > 0 and expanded_past_key_values is None:
                if local_num_drafts == 1:
                    expanded_past_key_values = past_key_values
                    expanded_attn_mask = cur_attn_mask
                    expanded_position_ids = cur_position_ids
                else:
                    expanded_past_key_values = expand_cache(
                        past_key_values,
                        local_num_drafts,
                        materialize=dual_cache,
                        workspace=expanded_cache_workspace,
                    )
                    expanded_cache_workspace = expanded_past_key_values
                    expanded_attn_mask = (
                        None
                        if cur_attn_mask is None
                        else cur_attn_mask.repeat_interleave(local_num_drafts, dim=0)
                    )
                    expanded_position_ids = cur_position_ids.repeat_interleave(local_num_drafts, dim=0)

            if local_num_drafts == 0:
                draft_x0 = x0_pred.new_empty((0, block_length))
                draft_conf = confidence.new_empty((0, block_length))
            elif dual_cache:
                block_input = local_draft_blocks
                if focus_enabled:
                    draft_x0, draft_conf = focus_model_forward(
                        model=model,
                        block_input=block_input,
                        fallback_x0=x0_pred.repeat_interleave(
                            local_num_drafts, dim=0
                        ),
                        fallback_confidence=confidence.repeat_interleave(
                            local_num_drafts, dim=0
                        ),
                        past_key_values=past_key_values,
                        cur_attn_mask=expanded_attn_mask,
                        target_indices=focus_targets,
                        block_start=block_start,
                        block_length=block_length,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        alg=alg,
                        focus_stats=focus_stats,
                        expanded_past_key_values=expanded_past_key_values,
                    )
                else:
                    draft_x0, draft_conf = model_forward(model, block_input, 
                            past_key_values, cur_attn_mask, cur_position_ids, local_num_drafts, temperature, top_p, top_k,
                            block_length, dual_cache, replace_position, replace_indices,
                            expanded_past_key_values, expanded_attn_mask, expanded_position_ids, alg=alg)
            else:
                block_input = local_draft_blocks
                draft_x0, draft_conf = model_forward(
                    model,
                    block_input,
                    past_key_values,
                    cur_attn_mask,
                    cur_position_ids,
                    local_num_drafts,
                    temperature,
                    top_p,
                    top_k,
                    block_length,
                    dual_cache,
                    expanded_past_key_values=expanded_past_key_values,
                    expanded_attn_mask=expanded_attn_mask,
                    expanded_position_ids=expanded_position_ids,
                    alg=alg,
                )
            decode_step += 1

            if draft_world_size > 1:
                draft_blocks, draft_x0, draft_conf = _gather_draft_results(
                    local_draft_blocks,
                    draft_x0,
                    draft_conf,
                    batch_size=x.shape[0],
                    num_drafts=num_drafts,
                    workspace=draft_gather_workspace,
                )
            else:
                draft_blocks = local_draft_blocks


            # 4. Graph-based Accept / Reject Logic

            x, x0_pred, confidence = graph_accept_reject_blocks(
                x=x,
                x0_pred=x0_pred,
                confidence=confidence,
                draft_blocks_flat=draft_blocks,
                draft_x0=draft_x0,
                draft_conf=draft_conf,
                graph_structure=graph_structure,
                threshold=threshold,
                mask_token_id=mask_token_id,
                block_start=block_start,
                block_end=block_end,
                alg=alg,
                graph_edge_order=graph_edge_order,
                graph_edge_lookup=graph_edge_lookup,
                acceptance_mode=acceptance_mode,
                acceptance_stats=acceptance_feedback,
                                        eos_token_id=eos_token_id,
            )
            if acceptance_mode == "deepest_only" and not acceptance_feedback.pop(
                "_last_deepest_success", False
            ):
                if dual_cache:
                    recovery_input = x[:, block_start:block_end]
                else:
                    recovery_input = x[:, block_start:]
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
                    replace_position if dual_cache else None,
                    replace_indices if dual_cache else None,
                    alg=alg,
                )
                decode_step += 1
                acceptance_feedback["recovery_forward_calls"] = (
                    acceptance_feedback.get("recovery_forward_calls", 0) + 1
                )
            if eos_state.update(x):
                if return_decode_steps:
                    return x, decode_step
                return x


        # EOS early termination

        eos_per_sequence = (x[:, prompt_length:] == eos_token_id).any(dim=1)
        eos_reached = (
            eos_per_sequence.any() if stop_on_any_eos else eos_per_sequence.all()
        )
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
def diffusion_generate_Spiffy(
    model,
    inputs: Optional[torch.Tensor] = None,
    top_p: Optional[int] = None,
    top_k: Optional[int] = None,
    draft_steps: int = 7,
    return_decode_steps: bool = False,
    **kwargs,
):
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
from .context_parallel_opener import context_parallel_opener
