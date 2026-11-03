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


from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Dict, Optional, Tuple, Union

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
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
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
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0


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


def active_mask_to_first_eos(x, prompt_length, eos_token_id, mask_token_id):
    return active_token_mask(x[:, prompt_length:], mask_token_id, eos_token_id)


@generation_policy
@torch.no_grad()
def diffusion_generate(
    model,
    inputs: Optional[torch.Tensor] = None,
    top_p: Optional[int] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = 0.9,
    num_small_blocks: Optional[int] = 1,
    return_decode_steps: bool = False, 
    **kwargs,
):
    block_length=kwargs.pop("block_length", 32)
    attention_mask = kwargs.pop("attention_mask", None)
    alg = kwargs.get("alg", 'origin')
    temperature = kwargs.get("temperature", 0.0)
    mask_token_id = kwargs.get("mask_token_id", None)
    eos_token_id = kwargs.get("eos_token_id", None)
    stop_on_any_eos = kwargs.get("stop_on_any_eos", False)
    

    dual_cache = kwargs.get("dual_cache", False)
    use_context_parallel_opener = kwargs.get("context_parallel_opener", False)


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
    decode_step = 0

    for num_block in range(num_blocks):
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
            cur_x = x[:, block_start:block_end]
            cur_attn_mask = None if attention_mask is None else attention_mask[:, :, block_start:block_end, :]
            cur_position_ids = position_ids[:, block_start:block_end]
        else:
            cur_x = x[:, block_start:]
            cur_attn_mask = None if attention_mask is None else attention_mask[:, :, block_start:, :]
            cur_position_ids = position_ids[:, block_start:]

          
        # Use cache for generation
        small_block_length = block_length // num_small_blocks

        assert block_length % num_small_blocks == 0


        for small_block_idx in range(num_small_blocks):
            small_block_start = small_block_idx * small_block_length
            small_block_end = small_block_start + small_block_length

            while True:
                if eos_state.update(x):
                    return (x, decode_step) if return_decode_steps else x
                active_mask = active_mask_to_first_eos(
                    x, prompt_length, eos_token_id, mask_token_id
                )[:, block_start - prompt_length:block_end - prompt_length]
                sub_mask_index = active_mask[:, small_block_start:small_block_end]
                sub_mask_index &= ~eos_state.finished[:, None]
                if sub_mask_index.sum() == 0:
                    break
                

                if dual_cache:
                    if eos_state.update(x):
                        return (x, decode_step) if return_decode_steps else x
                    output = model(cur_x,
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
                    output = model(cur_x,
                            attention_mask=cur_attn_mask,
                            position_ids=cur_position_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                            dual_cache=False)


                logits = output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits = logits[:, small_block_start:small_block_end]

                confidence, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k,
                                            neg_entropy=(alg in {'entropy', 'entropy_threshold'}), margin_confidence=(alg == 'topk_margin'))
                confidence = torch.where(sub_mask_index, confidence, -torch.inf)
                transfer_index = (F.one_hot(torch.max(confidence, dim=1)[1], num_classes=small_block_length) == 1)

                if alg == 'confidence_threshold':
                    transfer_index |= (confidence > threshold)
                elif alg == 'entropy_threshold':
                    transfer_index |= (confidence > -threshold)
                elif alg == "local_leap":
                    transfer_index |= local_leap_transfer_index(confidence)   
                transfer_index &= sub_mask_index
                cur_x[:, small_block_start:small_block_end][transfer_index] = x0[transfer_index]
                decode_step += 1
                if eos_state.update(x):
                    if return_decode_steps:
                        return x, decode_step
                    return x

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
    print(f"Normal stop, total decode time:{total_time}")
    print(f"decode steps:{decode_step}")

    if return_decode_steps:
        return x, decode_step
    else:
        return x
from .context_parallel_opener import context_parallel_opener
