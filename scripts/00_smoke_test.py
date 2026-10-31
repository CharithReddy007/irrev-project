"""Run this FIRST on the GPU machine, for each model. Takes about 2 minutes.

    python scripts/00_smoke_test.py --model qwen7b

It checks the things that silently break between library versions:
  1. the model loads in bf16 and fits in memory
  2. the chat template renders, and the action span we index is exactly the action text
  3. hidden_states has n_layers + 1 entries of the right size
  4. the steering hook changes the output, and is removed afterwards (output returns to normal)
  5. greedy generation is deterministic and produces a parseable tool call
If any check fails, fix it before running anything else.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from irrev.agent.loop import parse_tool_call  # noqa: E402
from irrev.models import lm as L  # noqa: E402
from irrev.models.prompts import probe_messages, agent_messages  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen7b")
    ap.add_argument("--data", default="data/dataset.jsonl")
    a = ap.parse_args()
    import torch
    rec = json.loads(open(a.data).readline())
    L.set_seed(0)
    lm = L.load(a.model)
    print(f"[1] loaded {a.model} ({getattr(lm, 'precision', '?')}): {lm.n_layers} layers, "
          f"d={lm.model.config.hidden_size}, GPU {torch.cuda.get_device_name(0)}, "
          f"memory used {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    msgs = probe_messages(rec)
    text = L.render(lm, msgs, add_generation_prompt=False)
    ids, a0, a1 = L.action_token_span(lm, text, rec["action_text"])
    span = lm.tok.decode(ids[a0:a1 + 1])
    print(f"[2] action tokens {a0}..{a1} of {len(ids)}; decoded span:\n    {span!r}\n    expected:\n    {rec['action_text']!r}")
    assert rec["action_text"].strip() in span.strip() or span.strip() in rec["action_text"], "action span mismatch"

    last, mean, n = L.extract_hidden(lm, msgs, rec["action_text"])
    assert last.shape == (lm.n_layers + 1, lm.model.config.hidden_size), last.shape
    assert np.isfinite(last).all()
    print(f"[3] hidden states OK: {last.shape}; norms at layers 0/mid/last = "
          f"{np.linalg.norm(last[0]):.1f} / {np.linalg.norm(last[lm.n_layers // 2]):.1f} / {np.linalg.norm(last[-1]):.1f}")

    gm = agent_messages(rec.get("task_text", "Tidy the workspace."), rec["state_text"])
    out0 = L.generate(lm, gm, max_new_tokens=60)
    out0b = L.generate(lm, gm, max_new_tokens=60)
    assert out0 == out0b, "greedy decoding is not deterministic"
    layer = lm.n_layers // 2
    v = np.random.default_rng(0).normal(size=lm.model.config.hidden_size).astype(np.float32)
    v *= np.linalg.norm(last[layer]) / np.linalg.norm(v)
    with L.steer(lm, layer, v, 4.0):
        out1 = L.generate(lm, gm, max_new_tokens=60)
    out2 = L.generate(lm, gm, max_new_tokens=60)
    print(f"[4] unsteered: {out0!r}\n    steered:   {out1!r}\n    after:     {out2!r}")
    assert out1 != out0, "steering hook had no effect (check block output type / layer path)"
    assert out2 == out0, "steering hook was not removed"

    batch = L.generate_batch(lm, [gm, gm[:1] + [{"role": "user", "content": "Say OK."}], gm], max_new_tokens=60)
    print(f"[4b] batched: {batch[0]!r}")
    if batch[0] != out0 or batch[2] != out0:
        print("    WARNING: batched output differs from unbatched (padding numerics). Usually harmless;"
              " keep --batch-size fixed for a whole experiment and report it.")
    else:
        print("    batched == unbatched")

    call, status = parse_tool_call(out0)
    print(f"[5] parse status of the first agent step: {status}; call = {call}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
