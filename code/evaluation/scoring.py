"""Deterministic answer extraction; raw outputs are always retained.

GSM8K primary accuracy follows lm-evaluation-harness v0.4.8 flexible-extract.
Code normalization removes response containers, never repairs a program or
chooses an answer using reference tests.
"""
import ast
import io
import re
import tokenize

TASK_STOPS = {
    "gsm8k": ["Question:", "</s>", "<|im_end|>"],
    "humaneval": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
    "mbpp": ["[DONE]"],
}
EXTRACTION_VERSION = "resolved-prefix-v2"


def truncate_stops(text, stops):
    return text[:min([text.find(s) for s in stops if s and s in text] or [len(text)])]


def generation_stops(dataset):
    return TASK_STOPS[dataset]


def encode_stop_strings(tokenizer, stops):
    return [list(v) for v in dict.fromkeys(tuple(tokenizer.encode(s, add_special_tokens=False)) for s in stops) if v]


def _string_lines(text):
    occupied = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.STRING:
                occupied.update(range(token.start[0], token.end[0] + 1))
    except tokenize.TokenError as exc:
        if 'multi-line string' in str(exc) and len(exc.args) > 1:
            occupied.update(range(exc.args[1][0], len(text.splitlines()) + 2))
    except (IndentationError, SyntaxError):
        pass
    return occupied


def _string_spans(text):
    """Exact character ranges; a quoted value does not protect its whole line."""
    starts=[0]
    starts.extend(match.end() for match in re.finditer('\n',text))
    def offset(position):
        row,column=position
        return starts[min(row-1,len(starts)-1)]+column
    spans=[]
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type==tokenize.STRING:spans.append((offset(token.start),offset(token.end)))
    except tokenize.TokenError as error:
        if 'multi-line string' in str(error) and len(error.args)>1:
            spans.append((offset(error.args[1]),len(text)))
    except (IndentationError,SyntaxError):
        pass
    return spans


def unwrap_code(text):
    """Remove Markdown wrappers without deleting fences inside Python strings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    occupied = _string_lines(text)
    matches = []
    for match in re.finditer(r"(?m)^[ \t]*```([^\n`]*)[ \t]*(?:\n|$)", text):
        if text.count("\n", 0, match.start()) + 1 not in occupied:
            matches.append(match)
    if not matches:
        return text, "plain"
    first = matches[0]
    # An answer already starting as Python uses its first empty fence as a
    # closing boundary; do not select a later explanation/example block.
    prefix = text[:first.start()]
    starts_as_code = bool(re.search(r"(?m)^[ \t]*(?:def |async def |class |import |from |return |raise |if |for |while |[A-Za-z_]\w*\s*=)", prefix))
    language = first.group(1).strip().lower()
    if prefix.strip() and starts_as_code and not language:
        return prefix, "closing_fence"
    if language in ("", "python", "python3", "py"):
        end = next((m.start() for m in matches[1:] if not m.group(1).strip()), len(text))
        return text[first.end():end], "markdown_block"
    return text, "plain"


def clean_completion(text, dataset):
    if dataset == "gsm8k":
        return truncate_stops(text, TASK_STOPS[dataset]), "task_stop"
    code, rule = unwrap_code(text)
    return code[:text_boundary(code, dataset)], rule


def text_boundary(text, dataset, entry_point=None):
    """First task delimiter outside Python literals, including closing fences.

    An opening Markdown fence is not terminal. A repeated complete HumanEval
    entry definition is protected; later definitions follow the task boundary.
    """
    if dataset == 'gsm8k':
        return len(truncate_stops(text, TASK_STOPS[dataset]))
    strings = _string_spans(text)
    def in_string(position):return any(start<=position<end for start,end in strings)
    boundaries = []
    protected = re.search(r'(?m)^(?:async\s+)?def\s+' + re.escape(entry_point) + r'\s*\(', text) if entry_point else None
    for stop in TASK_STOPS[dataset]:
        for match in re.finditer(re.escape(stop), text):
            if in_string(match.end()-1):
                continue
            if stop == '\ndef' and protected and match.start() + 1 == protected.start():
                continue
            if stop == '\ndef' and entry_point:
                before = text[:match.start()]
                fragment = re.match(r'def\s*(\w*)', text[match.start()+1:])
                body_started = re.search(r'(?m)^[ \t]*(return |raise |def |class )', before)
                if not body_started and fragment and entry_point.startswith(fragment.group(1)):
                    continue
            boundaries.append(match.start()); break
    fences = [m for m in re.finditer(r'(?m)^[ \t]*```([^\n`]*)[ \t]*(?:\n|$)', text)
              if not in_string(m.start())]
    if fences:
        first = fences[0]
        if first.group(1).strip():
            closing = next((m for m in fences[1:] if not m.group(1).strip()), None)
        elif re.search(r'(?m)^[ \t]*(def |class |return |import |from |[A-Za-z_]\w*\s*=)', text[:first.start()]):
            closing = first
        else:
            closing = next((m for m in fences[1:] if not m.group(1).strip()), None)
        if closing:
            boundaries.append(closing.start())
    return min(boundaries or [len(text)])


class TextStopState:
    """Check only the stable, unmasked prefix; independent of BPE boundaries."""
    def __init__(self, tokenizer, dataset, prompts):
        self.tokenizer, self.dataset = tokenizer, dataset
        self.ends = [None] * len(prompts)
        self.last_lengths = [-1] * len(prompts)
        self.entries = []
        for prompt in prompts:
            functions = re.findall(r'(?m)^def\s+(\w+)\s*\(', prompt)
            self.entries.append(functions[-1] if dataset == 'humaneval' and functions else None)

    def update(self, x, prompt_length, mask_token_id):
        for i, ids in enumerate(x[:, prompt_length:].detach().cpu().tolist()):
            if self.ends[i] is not None:
                continue
            length = ids.index(mask_token_id) if mask_token_id in ids else len(ids)
            if length == self.last_lengths[i]:
                continue
            self.last_lengths[i] = length
            prefix = ids[:length]
            def stopped(n):
                text = self.tokenizer.decode(prefix[:n], skip_special_tokens=True)
                return text_boundary(text, self.dataset, self.entries[i]) < len(text)
            if stopped(length):
                low, high = 1, length
                while low < high:
                    mid = (low + high) // 2
                    if stopped(mid):high = mid
                    else:low = mid + 1
                self.ends[i] = low - 1


def _normalize_math(value):
    for pattern in (",", r"\$", r"(?s).*#### ", r"\.$"):
        value = re.sub(pattern, "", value)
    return value.strip().lower()


def score_math(text, reference):
    completion, _ = clean_completion(text, "gsm8k")
    strict = re.findall(r"#### (\-?[0-9\.\,]+)", completion)
    flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", completion)
    answer = next((s for s in flexible[-1] if s), "[invalid]") if flexible else "[invalid]"
    strict_answer = strict[0] if strict else "[invalid]"
    gold = _normalize_math(reference)
    return dict(correct=_normalize_math(answer) == gold, accuracy_metric="gsm8k_flexible_exact_match",
                extracted_answer=answer, reference_answer=gold, strict_answer=strict_answer,
                strict_correct=_normalize_math(strict_answer) == gold)


def prepare_code(text, case, dataset):
    raw = truncate_stops(text, TASK_STOPS[dataset])
    code, rule = unwrap_code(text)
    if dataset == "humaneval":
        entry = case["entry_point"]
        full = re.search(r"(?m)^(?:async\s+)?def\s+" + re.escape(entry) + r"\s*\(", code)
        if full:
            # Preserve the first entry definition if a fenced answer repeats
            # the complete function, rather than treating it as body text.
            head = case["prompt"].split("def " + entry, 1)[0]
            candidate = head + truncate_stops(code, ["\n```", "\nif __name__"])
            rule += "+full_function"
        else:
            code = code[:text_boundary(code, dataset)]
            candidate = case["prompt"] + code
        raw_candidate = case["prompt"] + raw
        tests = case["test"] + "\ncheck(" + entry + ")"
    else:
        candidate = code[:text_boundary(code, dataset)]
        raw_candidate = raw
        tests = case.get("test_setup_code", "") + "\n" + "\n".join(case["test_list"])
    return dict(candidate_code=candidate, raw_candidate_code=raw_candidate, test_code=tests,
                extraction_rule=rule, extraction_version=EXTRACTION_VERSION)


def infer_dataset(until):
    if "[DONE]" in until:
        return "mbpp"
    if "\ndef" in until or "\nclass" in until:
        return "humaneval"
    return "gsm8k"
