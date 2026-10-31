"""Model loading, greedy generation, hidden-state extraction and steering hooks.

Written against the PyTorch + Hugging Face transformers APIs (AutoModelForCausalLM,
apply_chat_template, output_hidden_states, forward hooks). APIs change between versions:
run scripts/00_smoke_test.py on your GPU first, and record your versions with
`pip freeze > requirements.lock.txt`.

Layer indexing convention used everywhere in this repo
------------------------------------------------------
hidden_states returned by transformers has n_layers + 1 entries:
    index 0      = token embeddings ("layer 0", the embedding-layer baseline)
    index l >= 1 = output of decoder block l-1
"Probe layer l" means hidden_states[l]. Steering "at layer l" adds the vector to the output
of decoder block l-1, which is exactly what hidden_states[l] reads. steer() takes care of it.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import numpy as np

MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
}


PRECISION_SUFFIXES = ("4bit", "8bit", "float16", "bfloat16")


def parse_model_name(name: str) -> tuple[str, str | None]:
    """'qwen7b-4bit' -> ('qwen7b', '4bit'); 'qwen7b' -> ('qwen7b', None).

    Put the precision in the model name you pass to every script. Then every result lands in
    its own folder (results/qwen7b-4bit/ vs results/qwen7b-bfloat16/), so 4-bit and full-precision
    runs never mix, and resuming never skips work done at a different precision.
    """
    base, _, suf = name.rpartition("-")
    if base and suf in PRECISION_SUFFIXES:
        return base, suf
    return name, None


def set_seed(seed: int = 0) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class LM:
    name: str
    model: object
    tok: object

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def blocks(self):
        # Qwen2 and Llama both keep decoder blocks at model.model.layers
        return self.model.model.layers


def load(name: str, dtype: str | None = None) -> LM:
    """Load a model and tokenizer.

    Precision comes from `dtype` or $IRREV_DTYPE:
      bfloat16  - what the brief assumes (A100/H100/L4/L40; Colab Pro "L4" or "A100")
      float16   - GPUs without fast bf16 (V100, T4) when the model fits (a 7B model does NOT fit a 15 GB T4)
      8bit      - bitsandbytes LLM.int8 (7B ~ 8 GB). Closest to full precision on a T4, but slow
      4bit      - bitsandbytes NF4 with fp16 compute (7B ~ 5.5 GB). Fastest option on a free Colab T4
      auto      - bf16 if the GPU supports it and the model fits, else fp16 if it fits, else 4bit
    Use ONE precision per model for every experiment, and report it in the paper.
    $IRREV_MODEL_DIR overrides the Hub id with a local folder (e.g. a pre-downloaded copy).
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base, suffix = parse_model_name(name)
    dtype = dtype or suffix or os.environ.get("IRREV_DTYPE", "bfloat16")
    path = MODELS.get(base, base)
    local = os.environ.get("IRREV_MODEL_DIR")
    if local and os.path.isdir(os.path.join(local, path.split("/")[-1])):
        path = os.path.join(local, path.split("/")[-1])
    if dtype == "auto":
        dtype = _auto_precision(base)
        print(f"[irrev] auto precision -> {dtype}")
    tok = AutoTokenizer.from_pretrained(path)
    kw = dict(device_map="auto")
    if dtype in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        if dtype == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        else:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kw["torch_dtype"] = torch.float16
    else:
        kw["torch_dtype"] = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(path, **kw)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # needed for batched generation
    lm = LM(name, model, tok)
    lm.precision = dtype
    return lm


APPROX_PARAMS_B = {"qwen7b": 7.6, "llama8b": 8.0, "qwen14b": 14.8}


def _auto_precision(name: str) -> str:
    import torch
    free = torch.cuda.mem_get_info()[0] / 1e9
    need16 = APPROX_PARAMS_B.get(name, 8.0) * 2 * 1.15      # weights + activations headroom
    if torch.cuda.is_bf16_supported() and free > need16:
        return "bfloat16"
    if free > need16:
        return "float16"
    return "4bit"


def render(lm: LM, messages: list[dict], add_generation_prompt: bool) -> str:
    return lm.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


# ------------------------------------------------------------------ hidden states (Step 2)
def action_token_span(lm: LM, text: str, action: str) -> tuple[list[int], int, int]:
    """Tokenise text and return (input_ids, first_action_token, last_action_token)."""
    start_char = text.rfind(action)
    assert start_char >= 0, "action text not found in rendered prompt"
    end_char = start_char + len(action)
    enc = lm.tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    toks = [i for i, (s, e) in enumerate(offs) if e > start_char and s < end_char and e > s]
    assert toks, "no tokens overlap the action span"
    return ids, toks[0], toks[-1]


def extract_hidden(lm: LM, messages: list[dict], action: str, layers: list[int] | None = None):
    """One forward pass. Returns (last_token [L+1, d], mean_over_action [L+1, d]) as float32 numpy.

    The prompt is rendered *without* a generation prompt because the action is already the
    assistant's turn; the end-of-turn token after the action is ignored (we index the action span).
    """
    import torch
    text = render(lm, messages, add_generation_prompt=False)
    ids, a0, a1 = action_token_span(lm, text, action)
    with torch.no_grad():
        out = lm.model(torch.tensor([ids], device=lm.model.device), output_hidden_states=True)
    hs = out.hidden_states                       # tuple of L+1 tensors [1, T, d]
    sel = range(len(hs)) if layers is None else layers
    last = torch.stack([hs[l][0, a1] for l in sel]).float().cpu().numpy()
    mean = torch.stack([hs[l][0, a0:a1 + 1].mean(0) for l in sel]).float().cpu().numpy()
    return last, mean, len(ids)


# ------------------------------------------------------------------ generation
def generate(lm: LM, messages: list[dict], max_new_tokens: int = 128, return_scores: bool = False):
    """Greedy decoding (temperature 0), as the brief requires for all behaviour experiments."""
    import torch
    text = render(lm, messages, add_generation_prompt=True)
    enc = lm.tok(text, return_tensors="pt", add_special_tokens=False).to(lm.model.device)
    with torch.no_grad():
        out = lm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                temperature=None, top_p=None, top_k=None,
                                pad_token_id=lm.tok.pad_token_id,
                                output_scores=return_scores, return_dict_in_generate=True)
    new = out.sequences[0, enc["input_ids"].shape[1]:]
    s = lm.tok.decode(new, skip_special_tokens=True)
    if not return_scores:
        return s
    # mean negative log-likelihood of the generated tokens under the model (RQ5)
    nll = []
    for step, tok_id in enumerate(new.tolist()):
        if step >= len(out.scores):
            break
        logp = torch.log_softmax(out.scores[step][0].float(), -1)[tok_id].item()
        nll.append(-logp)
        if tok_id == lm.tok.eos_token_id:
            break
    return s, float(np.mean(nll)) if nll else math.nan, len(nll)


def generate_batch(lm: LM, batch: list[list[dict]], max_new_tokens: int = 128) -> list[str]:
    """Greedy decoding for several conversations at once (left padding). Much faster on Colab.

    Batched and unbatched greedy outputs can differ in rare cases because padding changes the
    floating-point order of operations. Keep the batch size fixed within an experiment and
    record it (it is saved in every result file).
    """
    import torch
    if not batch:
        return []
    texts = [render(lm, m, add_generation_prompt=True) for m in batch]
    enc = lm.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(lm.model.device)
    with torch.no_grad():
        out = lm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                temperature=None, top_p=None, top_k=None,
                                pad_token_id=lm.tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [lm.tok.decode(row, skip_special_tokens=True) for row in new]


def continuation_nll(lm: LM, messages: list[dict], continuation: str) -> float:
    """Mean NLL of a given assistant continuation (teacher forcing)."""
    import torch
    prompt = render(lm, messages, add_generation_prompt=True)
    p_ids = lm.tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = lm.tok(continuation, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p_ids + c_ids], device=lm.model.device)
    with torch.no_grad():
        logits = lm.model(ids).logits[0, len(p_ids) - 1:-1].float()
    logp = torch.log_softmax(logits, -1).gather(1, torch.tensor(c_ids, device=logits.device)[:, None])
    return float(-logp.mean().item())


# ------------------------------------------------------------------ steering (Step 6, RQ4)
@contextlib.contextmanager
def steer(lm: LM, layer: int, vector: np.ndarray | None, strength: float):
    """Add strength * vector to the output of decoder block (layer - 1), at every position.

    layer uses the probe convention (1..n_layers). strength 0 or vector None = no-op.
    Handles both block return types (tensor, or tuple whose first element is the tensor).
    """
    if vector is None or strength == 0:
        yield
        return
    import torch
    assert 1 <= layer <= lm.n_layers, "steering layer must be a decoder block output (>= 1)"
    block = lm.blocks[layer - 1]
    v32 = torch.tensor(vector, dtype=torch.float32) * float(strength)
    cache = {}

    def add(h):
        key = (h.dtype, h.device)
        if key not in cache:                      # cast once to the activation dtype/device
            cache[key] = v32.to(device=h.device, dtype=h.dtype)
        return h + cache[key]

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (add(output[0]),) + tuple(output[1:])
        return add(output)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
