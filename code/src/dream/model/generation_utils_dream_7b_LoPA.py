from src.stopping import EOSState, has_eos, active_token_mask, generation_policy
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple, Union, List
import copy

import torch
import time
try:
    import torch_npu
except ImportError as e:
    pass
import torch.distributions as dists
from torch.nn import functional as F


@dataclass
class Branch:
    """Simple structure to hold branch information"""
    x: torch.Tensor
    score: float = 0.0
    # Optional: track which token was just added for debugging
    last_token_added: Optional[int] = None 


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


def expand_cache(past_key_values: list, n: int):
    """将 Cache 在 Batch 维扩展 n 倍"""
    new_past_key_values = []
    for i in range(len(past_key_values)):
        new_past_key_values.append(())
        for j in range(len(past_key_values[i])):
            # [Batch, ...] -> [Batch*N, ...]
            expanded_tensor = past_key_values[i][j].repeat_interleave(n, dim=0)
            new_past_key_values[i] += (expanded_tensor,)
    
    return new_past_key_values


def local_leap_transfer_index(confidence, threshold=0.9, radius=4, relax_threshold=0.75):
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
    anchor_mask = confidence >= threshold
    anchor_indices = torch.nonzero(anchor_mask, as_tuple=True)
    if len(anchor_indices[0]) == 0:
        return transfer_index
    batch_size, seq_len = confidence.shape
    for batch_idx in range(batch_size):
        batch_anchors = [idx for i, idx in enumerate(anchor_indices[1]) if anchor_indices[0][i] == batch_idx]       
        for anchor_pos in batch_anchors:
            start_pos = max(0, anchor_pos - radius)
            end_pos = min(seq_len - 1, anchor_pos + radius)
            radius_range = slice(start_pos, end_pos + 1)
            neighbor_mask = confidence[batch_idx, radius_range] >= relax_threshold   
            transfer_index[batch_idx, radius_range] = transfer_index[batch_idx, radius_range] | neighbor_mask
    return transfer_index


def topk_branch_lookahead(
    model,
    x,
    cur_x,
    anchor_x,
    mask_index,
    transfer_index,
    confidence,
    x0_pred,
    k_branches,
    block_start,
    block_end,
    block_length,

    # cur_padding_mask,


    cur_attn_mask,
    dual_cache,
    replace_position,

    cur_position_ids,
    past_key_values,
    temperature,
    top_p,
    top_k,
    device,
):
    """Score top-k residual-mask branches, retaining valid anchors per row.

    Completed rows keep an unchanged anchor. An active row with no spatial
    progress must choose a real speculative branch to guarantee progress.
    """
    batch_size = x.shape[0]

    # 检查anchor_x是否有新填充的token
    has_anchor_progress = transfer_index.any(dim=1)  # [batch_size]
    
    # 1. Identify Top-K Unfilled Positions (相对于anchor_x)
    anchor_unfilled_mask = mask_index & (~transfer_index)
    candidate_conf = torch.where(anchor_unfilled_mask, confidence, -torch.inf)
    
    # 确保至少有k_branches个候选位置
    num_candidates_available = anchor_unfilled_mask.sum(dim=1).max().item()
    effective_k = min(k_branches, num_candidates_available)
    
    # 获取top-k位置
    top_k_creation_scores, top_k_indices = torch.topk(candidate_conf, k=effective_k, dim=1)
    
    n_candidates = effective_k + 1
    anchor_offset = 1


    batch_candidates_x = torch.zeros((batch_size, n_candidates, anchor_x.size(-1)), 
                                     device=device, dtype=anchor_x.dtype)
    batch_candidates_mask = torch.zeros((batch_size, n_candidates, anchor_x.size(-1)), 
                                        device=device, dtype=torch.bool)

    batch_creation_conf = torch.zeros((batch_size, n_candidates), 
                                      device=device, dtype=confidence.dtype)
    
    # 3.1 如果包含anchor branch，将其放在索引0
    batch_candidates_x[:, 0, :] = anchor_x
    batch_candidates_mask[:, 0, :] = anchor_unfilled_mask
    anchor_valid = has_anchor_progress | ~mask_index.any(dim=1)
    batch_creation_conf[:, 0] = torch.where(anchor_valid, 1.0, -torch.inf)
    
    # 3.2 构建top-k branches
    branch_pred_tokens = torch.gather(x0_pred, 1, top_k_indices)
    
    for k in range(effective_k):
        branch_idx = k + anchor_offset
        
        # 从anchor_x开始构建
        batch_candidates_x[:, branch_idx, :] = anchor_x.clone()
        batch_candidates_mask[:, branch_idx, :] = anchor_unfilled_mask.clone()
        
        # 填充第k个top-k位置
        cols = top_k_indices[:, k]  # [batch_size]
        vals = torch.where(anchor_unfilled_mask.gather(1, cols[:, None]).squeeze(1),
                           branch_pred_tokens[:, k], anchor_x.gather(1, cols[:, None]).squeeze(1))
        
        batch_candidates_x[:, branch_idx, :].scatter_(1, cols.unsqueeze(1), vals.unsqueeze(1))
        batch_candidates_mask[:, branch_idx, :].scatter_(1, cols.unsqueeze(1), False)
        
        # 记录creation confidence
        batch_creation_conf[:, branch_idx] = top_k_creation_scores[:, k]
    

    draft_input = batch_candidates_x.view(batch_size * n_candidates, -1)


    draft_kv_cache = expand_cache(past_key_values, n_candidates)
    

    draft_attn_mask = cur_attn_mask.repeat_interleave(n_candidates, dim=0)


    draft_pos_ids = cur_position_ids.repeat_interleave(n_candidates, dim=0)
    
    with torch.no_grad():


        if dual_cache:
            draft_replace_position = replace_position.repeat_interleave(n_candidates, dim=0)
            output_draft = model(draft_input, attention_mask=draft_attn_mask, 
                                position_ids=draft_pos_ids, past_key_values=draft_kv_cache, use_cache=True, dual_cache=True, replace_position=draft_replace_position)
        else:
            output_draft = model(draft_input, attention_mask=draft_attn_mask, 
                            position_ids=draft_pos_ids, past_key_values=draft_kv_cache, use_cache=True, dual_cache=False)


        draft_logits = output_draft.logits
        draft_logits = torch.cat([draft_logits[:, :1], draft_logits[:, :-1]], dim=1)
        draft_conf, draft_x0 = sample_tokens(draft_logits, temperature=temperature, top_p=top_p, top_k=top_k)
    

    draft_conf = draft_conf.view(batch_size, n_candidates, -1)
    draft_x0 = draft_x0.view(batch_size, n_candidates, -1)


    next_confidence_list = []
    next_x0_pred_list = []
    
    # 超参数
    alpha = 0.0  # creation confidence的权重
    branch_topp = 1.0  # 考虑未填充位置的比例
    
    for b in range(batch_size):
        best_score = -float('inf')
        best_idx = 0
        
        for i in range(n_candidates):
            mask = batch_candidates_mask[b, i]
            creation_conf = batch_creation_conf[b, i].item()
            current_branch_conf = draft_conf[b, i]
            if creation_conf == -float('inf'):
                continue
            
            # 计算未填充位置的future score
            unfilled_confs = current_branch_conf[mask]
            
            if unfilled_confs.numel() > 0:
                # 取confidence最低的top-p位置的平均值作为future score
                sorted_confs, _ = torch.sort(unfilled_confs, descending=False)
                num_positions = unfilled_confs.shape[0]
                bottom_cnt = max(1, int(num_positions * branch_topp))
                future_score = sorted_confs[:bottom_cnt].mean().item()
            else:
                # 如果所有位置都已填充，future score为1.0
                future_score = 1.0
            
            # 综合评分
            final_score = alpha * creation_conf + (1.0 - alpha) * future_score
            
            if final_score > best_score:
                best_score = final_score
                best_idx = i
        
        # 选择最佳候选

        # x[b, block_start:block_end] = batch_candidates_x[b, best_idx]

        
        x[b, block_start:block_start+batch_candidates_x[b, best_idx].shape[0]] = batch_candidates_x[b, best_idx]


        next_confidence_list.append(draft_conf[b, best_idx])
        next_x0_pred_list.append(draft_x0[b, best_idx])
        

    confidence = torch.stack(next_confidence_list, dim=0)
    x0_pred = torch.stack(next_x0_pred_list, dim=0)
    
    return x, confidence, x0_pred


@generation_policy
@torch.no_grad()
def diffusion_generate_LoPA(
    model,
    inputs: Optional[torch.Tensor] = None,
    top_p: Optional[int] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = 0.9,
    k_branches: int = 3,
    lopa_strategy: str = 'aggressive',  # [NEW] 'aggressive', 'moderate', 'conservative'
    return_decode_steps: bool = False,
    **kwargs,
):
    """
    LoPA策略说明:
    - 'aggressive': 使用base branch + top-k branch的完整lookahead
    - 'moderate': 若base branch已解码token则直接接受,否则使用top-k branch
    - 'conservative': 仅使用top-k branch的lookahead,不使用confidence/local_leap
    """
    block_length = kwargs.pop("block_length", 32)
    attention_mask = kwargs.pop("attention_mask", None)
    alg = kwargs.get("alg", 'origin')
    temperature = kwargs.get("temperature", 0.0)
    mask_token_id = kwargs.get("mask_token_id", None)
    eos_token_id = kwargs.get("eos_token_id", None)
    

    dual_cache = kwargs.get("dual_cache", False)


    assert mask_token_id is not None
    assert eos_token_id is not None
    assert inputs is not None
    assert attention_mask is not None
    assert lopa_strategy in ['aggressive', 'moderate', 'conservative'], \
        "lopa_strategy must be 'aggressive', 'moderate', or 'conservative'"

    input_ids = inputs

    if type(kwargs.get('max_new_tokens', None)) is int:
        max_length = kwargs.get('max_new_tokens') + input_ids.shape[-1]
    else:
        max_length = kwargs.get('max_length')
        if max_length is None:
            raise ValueError("Pass max_new_tokens or max_length")

    prompt_length = input_ids.shape[1]
    if block_length < 1 or max_length < prompt_length or (max_length - prompt_length) % block_length:
        raise ValueError("generation length must be nonnegative and divisible by positive block_length")
    assert (
        (max_length - prompt_length) % block_length == 0
    ), 'The token length cannot be evenly divided by the block length.'
    num_blocks = (max_length - prompt_length) // block_length
    device = model.device
    position_ids = torch.arange(max_length, device=device).unsqueeze(0)
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
    decode_step = 0
    

    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length


        # update cache
        if eos_state.update(x):
            return (x, decode_step) if return_decode_steps else x
        model_output = model(x, attention_mask, position_ids, use_cache=True)
        past_key_values = model_output.past_key_values
        logits = model_output.logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
        confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)

        x[:, block_start] = x0[:, block_start]
        decode_step += 1
        
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
            output = model(cur_x_input,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    dual_cache=True,
                    replace_position=replace_position)
        else:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            output = model(cur_x_input,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    dual_cache=False)


        logits = output.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        decode_step += 1
        confidence, x0_pred = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        
        while True:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x


            if dual_cache:
                cur_x = x[:, block_start:block_end]
            else:
                cur_x = x[:, block_start:]
            mask_index = active_token_mask(cur_x, mask_token_id)
            mask_index &= ~eos_state.finished[:, None]

            if not dual_cache:
                mask_index[:, block_length:] = False


            if mask_index.sum() == 0:
                break
            

            if lopa_strategy == 'conservative':

                num_masks_per_batch = mask_index.sum(dim=1)
                last_step_in_block = (1 >= num_masks_per_batch).all()

                if last_step_in_block:
                    # NOTE: 如果当前block一步就解码完了，那么可以放在block-wise的causal prefilling里面去做
                    cur_x[mask_index] = x0_pred[mask_index]
                    break       
                else:
                    # Conservative: 跳过base branch,直接进入top-k lookahead
                    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
                    anchor_x = cur_x.clone()
                
            else:
                # Aggressive & Moderate: 计算base branch
                confidence_masked = torch.where(mask_index, confidence, -torch.inf)

                if lopa_strategy == 'moderate':
                    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
                else:


                    transfer_index = (F.one_hot(torch.max(confidence_masked, dim=1)[1], num_classes=confidence_masked.size(1)) == 1)


                if alg == 'confidence_threshold':
                    transfer_index |= (confidence_masked > threshold)
                elif alg == "local_leap":
                    transfer_index |= local_leap_transfer_index(confidence_masked)
                transfer_index = transfer_index & mask_index

                num_masks_per_batch = mask_index.sum(dim=1)
                num_new_tokens_per_batch = transfer_index.sum(dim=1)
                last_step_in_block = (num_new_tokens_per_batch >= num_masks_per_batch).all()

                # Moderate策略: 如果base branch已有解码,直接接受
                if lopa_strategy == 'moderate' and num_new_tokens_per_batch.min() > 0:
                    cur_x[transfer_index] = x0_pred[transfer_index]
                    
                    # NOTE: 如果当前block一步就解码完了，那么可以放在block-wise的causal prefilling里面去做
                    if last_step_in_block:
                        break
                    
                    # parallel decoding解码成果，则继续下一轮不进入top-k lookahead
                    anchor_unfilled_mask = mask_index & (~transfer_index)
                    candidate_conf = torch.where(anchor_unfilled_mask, confidence, -torch.inf)
                    
                    draft_input = cur_x
                    draft_kv_cache = past_key_values


                    with torch.no_grad():


                        if dual_cache:
                            if eos_state.update(x):
                                return (x, decode_step) if return_decode_steps else x
                            output_draft = model(draft_input, attention_mask=cur_attn_mask, 
                                                position_ids=cur_position_ids, past_key_values=draft_kv_cache, use_cache=True, dual_cache=True, replace_position=replace_position)
                        else:
                            if eos_state.update(x):
                                return (x, decode_step) if return_decode_steps else x
                            output_draft = model(draft_input, attention_mask=cur_attn_mask, 
                                            position_ids=cur_position_ids, past_key_values=draft_kv_cache, use_cache=True, dual_cache=False)


                        draft_logits = output_draft.logits    
                        draft_logits = torch.cat([draft_logits[:, :1], draft_logits[:, :-1]], dim=1)
                        confidence, x0_pred = sample_tokens(draft_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                        decode_step += 1
                    

                    # x[:, block_start:block_end] = cur_x


                    if dual_cache:
                        x[:, block_start:block_end] = cur_x
                    else:
                        x[:, block_start:] = cur_x


                    continue

                # Aggressive策略: early stopping检查
                # NOTE: 如果当前block一步就解码完了，那么可以放在block-wise的causal prefilling里面去做
                if lopa_strategy == 'aggressive' and last_step_in_block:
                    cur_x[transfer_index] = x0_pred[transfer_index]
                    break

                anchor_x = cur_x.clone()
                anchor_x = torch.where(transfer_index, x0_pred, anchor_x)

            # Every candidate contains the spatial anchor. Commit it before
            # lookahead so an EOS cannot acquire extra speculative tokens.
            cur_x.copy_(anchor_x)
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            mask_index &= ~eos_state.finished[:, None]
            transfer_index &= mask_index
            anchor_x = cur_x.clone()


            x, confidence, x0_pred = topk_branch_lookahead(
                model=model,
                x=x,
                cur_x=cur_x,
                anchor_x=anchor_x,
                mask_index=mask_index,
                transfer_index=transfer_index,
                confidence=confidence,
                x0_pred=x0_pred,
                k_branches=k_branches,
                block_start=block_start,
                block_end=block_end,
                block_length=block_length,


                cur_attn_mask=cur_attn_mask,
                dual_cache=dual_cache,
                replace_position=(replace_position if dual_cache else None),

                cur_position_ids=cur_position_ids,
                past_key_values=past_key_values,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                device=device,
            )
            decode_step += 1

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
