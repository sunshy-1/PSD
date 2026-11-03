from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .modeling_dream import repeat_kv, rotate_half


def _partition_length(length: int, world_size: int, rank: int) -> Tuple[int, int]:
    return length * rank // world_size, length * (rank + 1) // world_size


def _all_gather_ragged_kv(
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    sequence_length: int,
    group=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    counts = [
        _partition_length(sequence_length, world_size, group_rank)[1]
        - _partition_length(sequence_length, world_size, group_rank)[0]
        for group_rank in range(world_size)
    ]
    max_count = max(counts)
    batch_size, local_count, key_value_size = local_key.shape
    local_pack = torch.zeros(
        (batch_size, max_count, key_value_size, 2),
        dtype=local_key.dtype,
        device=local_key.device,
    )
    local_pack[:, :local_count, :, 0].copy_(local_key)
    local_pack[:, :local_count, :, 1].copy_(local_value)
    gathered_pack = torch.empty(
        (world_size * batch_size, max_count, key_value_size, 2),
        dtype=local_key.dtype,
        device=local_key.device,
    )
    dist.all_gather_into_tensor(gathered_pack, local_pack, group=group)
    gathered_pack = gathered_pack.view(
        world_size, batch_size, max_count, key_value_size, 2
    )
    global_key = torch.cat(
        [gathered_pack[index, :, :count, :, 0] for index, count in enumerate(counts)],
        dim=1,
    )
    global_value = torch.cat(
        [gathered_pack[index, :, :count, :, 1] for index, count in enumerate(counts)],
        dim=1,
    )
    expected_start, expected_end = _partition_length(sequence_length, world_size, rank)
    if local_count != expected_end - expected_start:
        raise RuntimeError("Local context-parallel sequence partition is inconsistent")
    return global_key, global_value


@torch.no_grad()
def context_parallel_opener(
    model,
    input_ids: torch.Tensor,
    logit_index: int,
    group=None,
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    if not dist.is_available() or not dist.is_initialized():
        index = torch.tensor([logit_index], dtype=torch.long, device=input_ids.device)
        output = model(input_ids, use_cache=True, logit_indices=index)
        return output.logits, output.past_key_values

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    sequence_length = input_ids.shape[1]
    sequence_start, sequence_end = _partition_length(
        sequence_length, world_size, rank
    )
    local_input_ids = input_ids[:, sequence_start:sequence_end]
    decoder = model.model
    hidden_states = decoder.embed_tokens(local_input_ids)
    full_position_ids = torch.arange(
        sequence_length, dtype=torch.long, device=input_ids.device
    ).unsqueeze(0)
    cos, sin = decoder.rotary_emb(hidden_states, full_position_ids)
    local_cos = cos[:, sequence_start:sequence_end].unsqueeze(1)
    local_sin = sin[:, sequence_start:sequence_end].unsqueeze(1)
    global_cos = cos.unsqueeze(1)
    global_sin = sin.unsqueeze(1)
    past_key_values = []

    for layer in decoder.layers:
        residual = hidden_states
        normalized_states = layer.input_layernorm(hidden_states)
        query_states, local_key, local_value = layer.self_attn.project_qkv(
            normalized_states
        )
        global_key, global_value = _all_gather_ragged_kv(
            local_key,
            local_value,
            sequence_length,
            group=group,
        )
        past_key_values.append((global_key, global_value))

        batch_size, local_length, _ = query_states.shape
        attention = layer.self_attn
        query_states = query_states.view(
            batch_size, local_length, attention.num_heads, attention.head_dim
        ).transpose(1, 2)
        key_states = global_key.view(
            batch_size,
            sequence_length,
            attention.num_key_value_heads,
            attention.head_dim,
        ).transpose(1, 2)
        value_states = global_value.view(
            batch_size,
            sequence_length,
            attention.num_key_value_heads,
            attention.head_dim,
        ).transpose(1, 2)
        query_states = query_states * local_cos + rotate_half(query_states) * local_sin
        key_states = key_states * global_cos + rotate_half(key_states) * global_sin

        use_native_gqa = attention.num_key_value_groups > 1
        if not use_native_gqa:
            key_states = repeat_kv(key_states, attention.num_key_value_groups)
            value_states = repeat_kv(value_states, attention.num_key_value_groups)
        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=use_native_gqa,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, local_length, attention.hidden_size
        )
        hidden_states = residual + attention.o_proj(attention_output)
        residual = hidden_states
        hidden_states = residual + layer.mlp(
            layer.post_attention_layernorm(hidden_states)
        )

    hidden_states = decoder.norm(hidden_states)
    owner_rank = next(
        group_rank
        for group_rank in range(world_size)
        if _partition_length(sequence_length, world_size, group_rank)[0]
        <= logit_index
        < _partition_length(sequence_length, world_size, group_rank)[1]
    )
    logits = torch.empty(
        (input_ids.shape[0], 1, model.vocab_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    if rank == owner_rank:
        local_index = logit_index - sequence_start
        logits.copy_(model.lm_head(hidden_states[:, local_index:local_index + 1]))
    dist.broadcast(logits, src=owner_rank, group=group)
    return logits, past_key_values
