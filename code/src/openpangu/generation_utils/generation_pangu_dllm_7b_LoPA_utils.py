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

from transformers.cache_utils import Cache, DynamicCache


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


class BlockDynamicCache(DynamicCache):
    def __init__(self, _distributed_cache_data: Optional[Iterable] = None) -> None:
        super().__init__(_distributed_cache_data)
        self.skip_cache_update = False
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.skip_cache_update:
            key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return key_cache, value_cache
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

def expand_cache(cache: BlockDynamicCache, n: int):
    """将 Cache 在 Batch 维扩展 n 倍"""
    new_key_cache = []
    new_value_cache = []
    for layer_idx in range(len(cache.key_cache)):
        k = cache.key_cache[layer_idx]
        v = cache.value_cache[layer_idx]
        # [Batch, Num_Heads, Seq_Len, Head_Dim] -> [Batch*N, ...]
        # batch中的不同序列共用同一套kv cache
        k_expanded = k.repeat_interleave(n, dim=0)
        v_expanded = v.repeat_interleave(n, dim=0)
        new_key_cache.append(k_expanded)
        new_value_cache.append(v_expanded)
    
    new_cache = BlockDynamicCache()
    new_cache.key_cache = new_key_cache
    new_cache.value_cache = new_value_cache
    new_cache._seen_tokens = cache._seen_tokens 
    return new_cache

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


@generation_policy
@torch.no_grad()
def diffusion_generate_LoPA(
    model,
    inputs: Optional[torch.Tensor] = None,
    top_p: Optional[int] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = 0.9,
    k_branches: int = 3, # [NEW] Number of branches to guess
    return_decode_steps: bool = False,
    **kwargs,
):
    block_length=kwargs.pop("block_length", 32)
    attention_mask = kwargs.pop("attention_mask", None)
    alg = kwargs.get("alg", 'origin')
    temperature = kwargs.get("temperature", 0.0)
    mask_token_id = kwargs.get("mask_token_id", None)
    eos_token_id = kwargs.get("eos_token_id", None)

    assert mask_token_id is not None
    assert eos_token_id is not None
    assert inputs is not None
    assert attention_mask is not None

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
    # pad input_ids to max_length
    x = F.pad(input_ids, (0, max_length - prompt_length), value=mask_token_id)
    eos_state = EOSState(x, prompt_length, eos_token_id, mask_token_id,
                         kwargs.get("stop_on_any_eos", False))
    if max_length == prompt_length:
        return (x, 0) if return_decode_steps else x

    decode_step = 0
    # Initialize cache for the prompt
    past_key_values = BlockDynamicCache()

    causal_mask = torch.tril(torch.ones(max_length, max_length, device=device, dtype=torch.bool))[None, None, :, :]
    
    padding_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
    position_ids = padding_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(padding_mask == 0, 1)
    # [B, N] --> [B, 1, N, N]
    padding_mask = torch.logical_and(
        padding_mask.unsqueeze(1).unsqueeze(-2),
        padding_mask.unsqueeze(1).unsqueeze(-1),
    )
    attention_mask = padding_mask & causal_mask

    # Prefill stage
    if prompt_length > 0:
        cur_x = x[:, :prompt_length]
        cur_attn_mask = attention_mask[:, :, :prompt_length, :prompt_length]
        cur_position_ids = position_ids[:, :prompt_length]
        output = model(cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True
                        )
        decode_step += 1
        past_key_values = output.past_key_values
    
        logits = output.logits[:, -1:]
        confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        x[:, prompt_length:prompt_length + 1] = x0
    batch_size = x.shape[0]


    start_time = time.time()
    for num_block in range(num_blocks):
        

        # Prepare masks and position ids

        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length
        cur_position_ids = position_ids[:, block_start:block_end]
        
        # cur_padding_mask (Bidirectional): 用于投机 (Drafting & Verification)
        cur_padding_mask = padding_mask[:, :, block_start:block_end, :block_end]
        
        # cur_attn_mask (Causal): 用于更新 Cache
        cur_attn_mask = attention_mask[:, :, block_start:block_end, :block_end]
        
        # 投机过程中先不更新 KV Cache，等 Block 确定了再更新
        past_key_values.skip_cache_update = True
        

        # Initial inference

        cur_x_input = x[:, block_start:block_end]
        if eos_state.update(x):
            return (x, decode_step) if return_decode_steps else x
        output = model(cur_x_input, attention_mask=cur_padding_mask, 
                       position_ids=cur_position_ids, past_key_values=past_key_values, use_cache=True)
        logits = output.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        decode_step += 1
        
        # confidence: [B, L], x0_pred: [B, L]
        confidence, x0_pred = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        
        while True:
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            cur_x = x[:, block_start:block_end]
            mask_index = active_token_mask(cur_x, mask_token_id)
            mask_index &= ~eos_state.finished[:, None]
            if mask_index.sum() == 0:
                break


            # --- Multi-Branch Logic Start ---


            # 1. Generate Anchor Branch (Based on your original logic)
            confidence = torch.where(mask_index, confidence, -torch.inf)
            # Ensure at least one token (max confidence) is selected to guarantee progress
            transfer_index = (F.one_hot(torch.max(confidence, dim=1)[1], num_classes=block_length) == 1)
            
            if alg == 'confidence_threshold':
                transfer_index |= (confidence > threshold)
            elif alg == "local_leap":
                transfer_index |= local_leap_transfer_index(confidence)
            transfer_index = transfer_index & mask_index

            num_masks_per_batch = mask_index.sum(dim=1)
            num_new_tokens_per_batch = transfer_index.sum(dim=1)
            last_step_in_block = (num_new_tokens_per_batch >= num_masks_per_batch).all()

            # early stopping if speculative decoding is in fact not required
            if last_step_in_block:
                cur_x[transfer_index] = x0_pred[transfer_index]
                break


            # Spatial commits are shared by all candidates and may terminate
            # this row before speculative lookahead is allowed to extend it.
            cur_x[transfer_index] = x0_pred[transfer_index]
            if eos_state.update(x):
                return (x, decode_step) if return_decode_steps else x
            mask_index &= ~eos_state.finished[:, None]
            transfer_index &= mask_index

            # --- 2. Identify Top-K Unfilled Positions & Record Creation Conf ---
            anchor_unfilled_mask = mask_index & (~transfer_index)
            candidate_conf = torch.where(anchor_unfilled_mask, confidence, -torch.inf)
            
            # get topk confidence score and positions [B, K]
            top_k_creation_scores, top_k_indices = torch.topk(candidate_conf, k=min(k_branches, block_length), dim=1)


            # --- 3. Construct Candidates Batch ---
            effective_k = top_k_indices.shape[1]
            n_candidates = effective_k + 1

            # Base Anchor: [B, L]
            anchor_x = cur_x.clone()
            anchor_x = torch.where(transfer_index, x0_pred, anchor_x)
            
            # Expand to [B, n_candidates, L]
            batch_candidates_x = anchor_x.unsqueeze(1).repeat(1, n_candidates, 1) # [B, K+1, L]
            batch_candidates_mask = anchor_unfilled_mask.unsqueeze(1).repeat(1, n_candidates, 1) # [B, K+1, L]
            
            # 初始化 Creation Confidence [B, n_candidates]
            # Anchor 的 creation score 默认为 1.0
            batch_creation_conf = torch.ones((batch_size, n_candidates), device=device, dtype=confidence.dtype)
            batch_creation_conf[:, 1:] = top_k_creation_scores

            # 提取预测值: [B, K]
            branch_pred_tokens = torch.gather(x0_pred, 1, top_k_indices) 
            
            for k in range(effective_k):
                cols = top_k_indices[:, k] # [B]
                vals = torch.where(anchor_unfilled_mask.gather(1, cols[:, None]).squeeze(1),
                                   branch_pred_tokens[:, k], anchor_x.gather(1, cols[:, None]).squeeze(1))
                
                # 利用 scatter_ 更新
                # batch_candidates_x[:, k+1, :] 是 [B, L]
                batch_candidates_x[:, k+1, :].scatter_(1, cols.unsqueeze(1), vals.unsqueeze(1))
                batch_candidates_mask[:, k+1, :].scatter_(1, cols.unsqueeze(1), False)

            # --- 4. Batch Verification (Single Forward Pass) ---
            # Flatten [B, n_candidates, L] -> [B * n_candidates, L]
            draft_input = batch_candidates_x.view(-1, block_length)

            draft_kv_cache = expand_cache(past_key_values, n_candidates)
            draft_kv_cache.skip_cache_update = True

            draft_attn_mask = cur_padding_mask.repeat_interleave(n_candidates, dim=0)
            draft_pos_ids = cur_position_ids.repeat_interleave(n_candidates, dim=0)


            with torch.no_grad():
                if eos_state.update(x):
                    return (x, decode_step) if return_decode_steps else x
                output_draft = model(draft_input, attention_mask=draft_attn_mask, 
                                    position_ids=draft_pos_ids, past_key_values=draft_kv_cache, use_cache=True)
                draft_logits = output_draft.logits    
                draft_logits = torch.cat([draft_logits[:, :1], draft_logits[:, :-1]], dim=1)
                # draft_conf: [B * n_cand, L]
                draft_conf, draft_x0 = sample_tokens(draft_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                decode_step += 1

            # --- 5. Selection Logic (Per Batch Item) ---
            # Reshape back to [B, n_candidates, L]
            draft_conf = draft_conf.view(batch_size, n_candidates, block_length)
            draft_x0 = draft_x0.view(batch_size, n_candidates, block_length)

            next_confidence_list = []
            next_x0_pred_list = []

            alpha = 0
            branch_topp = 1

            for b in range(batch_size):
                best_score = -1.0
                best_idx = 0
                
                # 遍历当前样本的所有分支 (Anchor + K branches)
                for i in range(n_candidates):
                    mask = batch_candidates_mask[b, i] # [L] (bool)
                    creation_conf = batch_creation_conf[b, i].item() # scalar
                    current_branch_conf = draft_conf[b, i] # [L]
                    
                    # skip the inf paddings
                    if creation_conf == -float('inf'):
                        continue

                    # --- Logic: Future Mask Confidence (Weak Link) ---
                    unfilled_confs = current_branch_conf[mask] # Shape: [Num_Unfilled]
                    
                    if unfilled_confs.numel() > 0:
                        sorted_confs, _ = torch.sort(unfilled_confs, descending=False)
                        num_positions = unfilled_confs.shape[0]
                        bottom_cnt = max(1, int(num_positions * branch_topp))
                        future_score = sorted_confs[:bottom_cnt].mean().item()
                    else:
                        future_score = 1.0

                    # --- Logic: Final Weighted Score ---
                    final_score = alpha * creation_conf + (1.0 - alpha) * future_score
                    
                    # 更新最佳分支
                    if final_score > best_score:
                        best_score = final_score
                        best_idx = i
                
                # --- Apply Selection for Batch 'b' ---
                x[b, block_start:block_end] = batch_candidates_x[b, best_idx]
                next_confidence_list.append(draft_conf[b, best_idx])
                next_x0_pred_list.append(draft_x0[b, best_idx])

            confidence = torch.stack(next_confidence_list, dim=0) # [B, L]
            x0_pred = torch.stack(next_x0_pred_list, dim=0)     # [B, L]


            # --- Multi-Branch Logic End ---


        # EOS early termination

        if eos_state.update(x):
            total_time = time.time() - start_time
            print(f"Early termination, total decode time:{total_time}")
            print(f"Total Steps: {decode_step}")
            if return_decode_steps:
                return x, decode_step
            else:
                return x
        

        # Causal attention to update kvcache

        past_key_values.skip_cache_update = False
        final_block_input = x[:, block_start:block_end]
        if eos_state.update(x):
            return (x, decode_step) if return_decode_steps else x
        output = model(final_block_input, attention_mask=cur_attn_mask, 
              position_ids=cur_position_ids, past_key_values=past_key_values, use_cache=True)
        decode_step += 1
        past_key_values = output.past_key_values
        
        if num_block < num_blocks - 1:
            logits = output.logits[:, -1:] 
            confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            x[:, block_end:block_end + 1] = x0

    total_time = time.time() - start_time
    print(f"Decode Time: {total_time:.2f}s")
    print(f"Total Steps: {decode_step}")

    if return_decode_steps:
        return x, decode_step
    else:
        return x
