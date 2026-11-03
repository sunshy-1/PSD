"""Shared native-Torch rules from the PSD paper, Section 3.2 / Algorithm 1.

Graph indices are one-based residual confidence ranks. Zero is a legacy root
marker, never a token rank. ``draft_steps`` counts speculative edges after the
empty root, rather than the number of candidates in the verifier batch.
The default is the original multi-parent DAG. Its nodes and edges are unchanged.
"""
from functools import lru_cache

import torch
from src.stopping import active_token_mask, terminal_prefix_complete


@lru_cache(maxsize=32)
def _graph_nodes(depth, variant):
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("draft_steps must be a positive integer (speculative depth)")
    if variant not in ("chain", "dag", "heuristic_tree"):
        raise ValueError("Unsupported graph_variant: {}".format(variant))
    # At most two candidates per depth; the final level is the full prefix.
    # The paper does not publish its offline-calibrated graph. This deterministic
    # bounded topology supplies depth d without claiming that calibration.
    subsets = [()]
    for level in range(1, depth + 1):
        subsets.append(tuple(range(1, level + 1)))
        if variant != "chain" and level < depth:
            subsets.append(tuple(range(1, level)) + (level + 1,))
    nodes = []
    for index, ranks in enumerate(subsets):
        if index == 0:
            parents = (-1,)
        else:
            parents = tuple(p for p, previous in enumerate(subsets[:index])
                            if len(previous) + 1 == len(ranks)
                            and set(previous).issubset(ranks))
            if variant == "heuristic_tree":
                parents = parents[:1]
        nodes.append(((0,) + ranks, len(ranks) + 1, parents))
    return tuple(nodes)


def get_graph_structure(draft_steps, graph_variant="dag"):
    return [dict(indices=list(ranks), level=level, parents=list(parents))
            for ranks, level, parents in _graph_nodes(draft_steps, graph_variant)]


def rank_metadata(graph, device):
    offsets = tuple(tuple(i - 1 for i in node["indices"] if i > 0) for node in graph)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _rank_metadata(offsets, device)


@lru_cache(maxsize=64)
def _rank_metadata(offsets, device):
    """Reuse read-only rank tensors instead of allocating them per verifier call."""
    width = max(1, max(map(len, offsets)))
    ranks = torch.zeros((len(offsets), width), dtype=torch.long, device=device)
    valid = torch.zeros_like(ranks, dtype=torch.bool)
    for k, values in enumerate(offsets):
        ranks[k, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        valid[k, :len(values)] = True
    return ranks, valid


def draft_blocks(x, predictions, confidence, mask_id, start, end, graph,
                 ranks=None, valid=None):
    """Assemble candidates solely at residual masks, with no model calls."""
    tokens = x[:, start:end]
    batch, length = tokens.shape
    masked = active_token_mask(tokens, mask_id)
    order = torch.argsort(confidence.masked_fill(~masked, -torch.inf),
                          dim=-1, descending=True, stable=True)
    if ranks is None or valid is None:
        ranks, valid = rank_metadata(graph, x.device)
    eligible = (valid[None] & (ranks[None] >= 0)
                & (ranks[None] < masked.sum(-1)[:, None, None]))
    safe_ranks = ranks.clamp(0, length - 1)
    positions = order[:, None].expand(-1, len(graph), -1).gather(
        2, safe_ranks[None].expand(batch, -1, -1))
    transfer = torch.zeros((batch, len(graph), length), dtype=torch.int8,
                           device=x.device)
    transfer.scatter_reduce_(2, positions, eligible.to(torch.int8),
                             reduce="amax", include_self=True)
    transfer = transfer.bool() & masked[:, None]
    drafts = torch.where(transfer, predictions[:, None], tokens[:, None])
    return drafts.reshape(batch * len(graph), length)


def select_accepted(drafts, targets, graph, mask_id, eos_token_id=None):
    """Algorithm 1 lines 17--23: retain all accepted parents; check edge deltas."""
    batch, count, _ = drafts.shape
    if graph[0]["parents"] == [-1] and all(
        node["parents"] == [k - 1] for k, node in enumerate(graph[1:], 1)
    ):
        return _select_chain(drafts, targets, mask_id, eos_token_id)
    parent_key = tuple(tuple(node["parents"]) for node in graph)
    parents, valid, roots, root, depth = _parent_metadata(parent_key, drafts.device)
    # Check every edge together. Padded slots never count as graph edges.
    parent = drafts[:, parents]
    child = drafts[:, :, None]
    parent_filled = parent.ne(mask_id)
    delta = ~parent_filled & child.ne(mask_id)
    root_tokens = drafts[:, root]
    root_filled = root_tokens.ne(mask_id)
    preserves_root = ((drafts == root_tokens[:, None]) | ~root_filled[:, None]).all(-1)
    edges = (((child == parent) | ~parent_filled).all(-1)
             & ((child == targets[:, parents]) | ~delta).all(-1)
             & preserves_root[:, :, None] & valid[None])
    edges &= ~terminal_prefix_complete(parent, mask_id, eos_token_id)
    # Propagate reachability along ALL valid incoming edges. One tensor pass
    # per depth replaces the Python loop and allocations for every node.
    accepted = roots[None].expand(batch, -1)
    for _ in range(depth):
        accepted = accepted | (accepted[:, parents] & edges).any(-1)
    matched = accepted[:, parents] & edges
    secondary = (accepted & ~matched[:, :, 0] & matched[:, :, 1:].any(-1)).any(-1)
    # Count actual speculation: some graph ranks may exceed residual mask count.
    sizes = (drafts.ne(mask_id) & ~root_filled[:, None]).sum(-1)
    scores = sizes.masked_fill(~accepted, -1)
    selected = scores.argmax(-1)  # stable first-node tie break
    deepest = scores.max(-1).values.eq(sizes.max(-1).values)
    return selected, deepest, secondary


@lru_cache(maxsize=64)
def _parent_metadata(parent_key, device):
    """Cache padded parents without deleting branches or secondary parents."""
    roots = [k for k, values in enumerate(parent_key) if -1 in values]
    if len(roots) != 1 or parent_key[roots[0]] != (-1,):
        raise ValueError("A graph must have exactly one root with parents [-1]")
    root = roots[0]
    depths = []
    for k, values in enumerate(parent_key):
        if k == root:
            depths.append(0)
        elif not values or any(p < 0 or p >= k for p in values):
            raise ValueError("Graph parents must precede their child")
        else:
            depths.append(1 + max(depths[p] for p in values))
    width = max(map(len, parent_key))
    padded = [[0 if p == -1 else p for p in values] + [0] * (width - len(values))
              for values in parent_key]
    valid = [[p != -1 for p in values] + [False] * (width - len(values))
             for values in parent_key]
    return (torch.tensor(padded, dtype=torch.long, device=device),
            torch.tensor(valid, dtype=torch.bool, device=device),
            torch.arange(len(parent_key), device=device).eq(root), root, max(depths))


def _select_chain(drafts, targets, mask_id, eos_token_id):
    """Verify every chain edge in one tensor operation, preserving EOS and root."""
    parent, child = drafts[:, :-1], drafts[:, 1:]
    parent_filled = parent.ne(mask_id)
    delta = ~parent_filled & child.ne(mask_id)
    root = drafts[:, :1]
    root_filled = root.ne(mask_id)
    edges = (((child == parent) | ~parent_filled).all(-1)
             & ((child == targets[:, :-1]) | ~delta).all(-1)
             & ((child == root) | ~root_filled).all(-1))
    edges &= ~terminal_prefix_complete(parent, mask_id, eos_token_id)
    accepted = torch.cat((torch.ones_like(drafts[:, :1, 0], dtype=torch.bool),
                          edges.to(torch.int32).cumprod(-1).bool()), dim=1)
    sizes = (drafts.ne(mask_id) & ~root_filled).sum(-1)
    scores = sizes.masked_fill(~accepted, -1)
    selected = scores.argmax(-1)
    deepest = scores.max(-1).values.eq(sizes.max(-1).values)
    return selected, deepest, torch.zeros_like(deepest)


def accept_blocks(x, predictions, confidence, flat_drafts, draft_predictions,
                  draft_confidence, graph, threshold, mask_id, start, end, alg,
                  transfer_step, acceptance_mode="hierarchical", stats=None,
                  eos_token_id=None):
    batch = x.shape[0]
    count = len(graph)
    length = end - start
    targets, _ = transfer_step(flat_drafts, draft_predictions, draft_confidence,
                               threshold, mask_id, 0, length, alg)
    drafts = flat_drafts.reshape(batch, count, length)
    targets = targets.reshape(batch, count, length)
    selected, deepest, secondary = select_accepted(drafts, targets, graph, mask_id,
                                                  eos_token_id)
    if acceptance_mode == "hierarchical":
        use = torch.ones_like(deepest)
    elif acceptance_mode == "deepest_only":
        use = deepest
    else:
        raise ValueError("Unsupported acceptance_mode: {}".format(acceptance_mode))
    rows = torch.arange(batch, device=x.device)
    flat_indices = rows * count + selected
    x[:, start:end] = torch.where(use[:, None], drafts[rows, selected], x[:, start:end])
    predictions = torch.where(use[:, None], draft_predictions[flat_indices], predictions)
    confidence = torch.where(use[:, None], draft_confidence[flat_indices], confidence)
    if stats is not None:
        stats["_last_deepest_success"] = bool(deepest.all().item())
        values = dict(verification_calls=1, sequences_verified=batch,
                      deepest_success_sequences=int(deepest.sum().item()),
                      hierarchical_recovery_sequences=int((~deepest).sum().item()),
                      secondary_parent_recovery_sequences=int(secondary.sum().item()))
        for key, value in values.items():
            stats[key] = stats.get(key, 0) + value
        levels = torch.tensor([node["level"] for node in graph], device=x.device)[selected]
        for key, values in (("selected_level_counts", levels),
                            ("accepted_level_counts", torch.where(use, levels, 0))):
            counts = stats.setdefault(key, {})
            for value in values.tolist():
                counts[str(value)] = counts.get(str(value), 0) + 1
    return x, predictions, confidence


def accept_full(x, predictions, confidence, flat_drafts, draft_predictions,
                draft_confidence, graph, threshold, mask_id, start, end, alg,
                transfer_step, eos_token_id=None):
    return accept_blocks(x, predictions, confidence, flat_drafts[:, start:end],
                         draft_predictions, draft_confidence, graph, threshold,
                         mask_id, start, end, alg, transfer_step,
                         eos_token_id=eos_token_id)
