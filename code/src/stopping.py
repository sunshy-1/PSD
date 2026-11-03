"""Stopping and token accounting shared by all native diffusion decoders.

EOS is terminal only after its complete prefix has no masks. A scoped policy
shares EOS/task-stop rules with existing draft APIs. ContextVar isolates nested
calls and independent threads, without changing the graph or public API.
"""
from contextvars import ContextVar
from functools import wraps

import torch

_POLICY = ContextVar("diffusion_stopping_policy", default=((), ()))
_TEXT_STOP = ContextVar("diffusion_text_stop_state", default=None)


def _eos_ids(value):
    if value is None:
        return ()
    return (value,) if isinstance(value, int) else tuple(value)


def _policy(eos_token_id=None, stop_token_ids=None):
    eos, stops = _POLICY.get()
    return (_eos_ids(eos_token_id) if eos_token_id is not None else eos,
            tuple(tuple(s) for s in stop_token_ids if s) if stop_token_ids is not None else stops)


def generation_policy(function):
    """Scope public generation kwargs to all nested draft/transfer calls."""
    @wraps(function)
    def generate(*args, **kwargs):
        token = _POLICY.set(_policy(kwargs.get("eos_token_id"), kwargs.get("stop_token_ids")))
        text_token = _TEXT_STOP.set(kwargs.get("text_stop_state", _TEXT_STOP.get()))
        try:
            return function(*args, **kwargs)
        finally:
            _POLICY.reset(token)
            _TEXT_STOP.reset(text_token)
    return generate


def terminal_positions(tokens, eos_token_id=None, stop_token_ids=None):
    """Inclusive first terminator end, or length if absent; arbitrary batch dims."""
    eos_ids, stops = _policy(eos_token_id, stop_token_ids)
    length = tokens.shape[-1]
    end = torch.full(tokens.shape[:-1], length, dtype=torch.long, device=tokens.device)
    if not length:
        return end
    positions = torch.arange(length, device=tokens.device)
    for eos in eos_ids:
        end = torch.minimum(end, torch.where(tokens.eq(eos), positions, length).amin(-1))
    for pattern in stops:
        width = len(pattern)
        if width > length:
            continue
        values = torch.as_tensor(pattern, dtype=tokens.dtype, device=tokens.device)
        matches = tokens.unfold(-1, width, 1).eq(values).all(-1)
        ends = torch.arange(width - 1, length, device=tokens.device)
        end = torch.minimum(end, torch.where(matches, ends, length).amin(-1))
    return end


def active_token_mask(tokens, mask_token_id, eos_token_id=None, stop_token_ids=None):
    end = terminal_positions(tokens, eos_token_id, stop_token_ids)
    positions = torch.arange(tokens.shape[-1], device=tokens.device)
    return tokens.eq(mask_token_id) & (positions <= end[..., None])


def terminal_prefix_complete(tokens, mask_token_id, eos_token_id=None, stop_token_ids=None):
    end = terminal_positions(tokens, eos_token_id, stop_token_ids)
    return end.lt(tokens.shape[-1]) & ~active_token_mask(tokens, mask_token_id, eos_token_id, stop_token_ids).any(-1)


def has_eos(x, prompt_length, eos_token_id, stop_on_any_eos=False, mask_token_id=None):
    generated = x[:, prompt_length:]
    if not generated.shape[-1]:
        return False
    done = (terminal_positions(generated, eos_token_id).lt(generated.shape[-1]) if mask_token_id is None
            else terminal_prefix_complete(generated, mask_token_id, eos_token_id))
    return bool((done.any() if stop_on_any_eos else done.all()).item())


class EOSState:
    def __init__(self, x, prompt_length, eos_token_id, mask_token_id, stop_on_any_eos=False):
        self.prompt_length = prompt_length
        self.eos_token_id, self.stop_token_ids = _policy(eos_token_id)
        self.mask_token_id = mask_token_id
        self.stop_on_any_eos = stop_on_any_eos
        self.finished = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
        self.frozen = x.clone()
        self.text_stop = _TEXT_STOP.get()

    def update(self, x):
        x.copy_(torch.where(self.finished[:, None], self.frozen, x))
        generated = x[:, self.prompt_length:]
        if generated.shape[-1] == 0:
            self.finished.fill_(True)
            return True
        resolved = ~active_token_mask(generated, self.mask_token_id, self.eos_token_id,
                                      self.stop_token_ids).any(-1)
        if self.text_stop is not None:
            self.text_stop.update(x, self.prompt_length, self.mask_token_id)
            resolved |= torch.tensor([end is not None for end in self.text_stop.ends], device=x.device)
        newly_finished = resolved & ~self.finished
        self.frozen.copy_(torch.where(newly_finished[:, None], x, self.frozen))
        self.finished |= resolved
        return bool((self.finished.any() if self.stop_on_any_eos else self.finished.all()).item())

    def mask(self, x, start, end):
        generated = x[:, self.prompt_length:]
        boundary = terminal_positions(generated, self.eos_token_id, self.stop_token_ids) + self.prompt_length
        positions = torch.arange(start, end, device=x.device)
        return (x[:, start:end].eq(self.mask_token_id) & ~self.finished[:, None]
                & (positions >= self.prompt_length) & (positions <= boundary[:, None]))


def generated_lengths(output_ids, prompt_length, eos_token_id, mask_token_id, stop_token_ids=(), stop_positions=None):
    """Count the complete prefix including its emitted terminator; never drop holes."""
    generated = output_ids[:, prompt_length:]
    if not generated.shape[-1]:
        return [0] * generated.shape[0], [False] * generated.shape[0]
    end = terminal_positions(generated, eos_token_id, stop_token_ids)
    if stop_positions is not None:
        extra = torch.tensor([p if p is not None else generated.shape[-1] for p in stop_positions], device=generated.device)
        end = torch.minimum(end, extra)
    positions = torch.arange(generated.shape[-1], device=generated.device)
    prefix = positions <= end[:, None]
    if bool((generated.eq(mask_token_id) & prefix).any().item()):
        raise RuntimeError("Generation returned unresolved masks before its stopping boundary")
    hit_eos = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
    for eos in _eos_ids(eos_token_id):
        hit_eos |= (generated.eq(eos) & prefix).any(-1)
    return prefix.sum(-1).tolist(), hit_eos.tolist()


def generated_token_ids(row, prompt_length, eos_token_id, mask_token_id, stop_token_ids=(), stop_position=None):
    generated = row[prompt_length:]
    if not generated.shape[-1]:
        return generated
    end = int(terminal_positions(generated, eos_token_id, stop_token_ids).item())
    if stop_position is not None:
        end = min(end, stop_position)
    generated = generated[:end + 1]
    if bool(generated.eq(mask_token_id).any().item()):
        raise RuntimeError("Cannot decode an incomplete generated prefix")
    return generated
