#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for PPO generations with persona+act conditioning.

Unlike plain inference, this script ALWAYS injects:
  [BPER] persona triplets [EPER] [BACT] act [EACT]
before the [BRES] marker, so that evaluation metrics like PA_Consistency
can correctly check act/persona fidelity.

Usage:
python3 scripts/infer_ppo.py \
  --model_dir runs/ppo_debug_cpu \
  --test_jsonl pref_data/final_val_pref.jsonl \
  --out_jsonl he_outputs/ppo.jsonl \
  --force_cpu --no_sample
"""

import os, json, argparse, torch
from typing import Dict, Any, List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

SPECIAL_TOKENS = ["[BOS]","[EOS]","[BPER]","[EPER]",
                  "[BACT]","[EACT]","[BRES]","[ERES]"]

def add_special_tokens(tok):
    add = [t for t in SPECIAL_TOKENS if t not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    tok.padding_side = "left"
    return tok

# Add this helper near the top (after imports)
def get_ctx_limit(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("n_positions", "max_position_embeddings", "max_sequence_length"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    # sensible default for GPT-2 class
    return 1024

def build_gen_kwargs(tok, eres_id, args, do_sample: bool):
    # Only include sampling flags when do_sample=True to avoid warnings
    kw = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.pad_token_id,
        eos_token_id=[tok.eos_token_id, eres_id] if eres_id else tok.eos_token_id,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        num_beams=args.num_beams if not do_sample else 1,
    )
    if do_sample:
        kw.update(dict(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        ))
    return kw

def serialize_pa(persona_list: List[str], act_label: str) -> str:
    persona_str = " ".join(p.strip() for p in (persona_list or []))
    act_label = (act_label or "general_response").strip()
    return f"[BPER] {persona_str} [EPER] [BACT] {act_label} [EACT]"

def ensure_bres(s: str) -> str:
    s = (s or "").rstrip()
    return s + ("\n" if not s.endswith("\n") else "") + "[BRES]"

def build_prompt_pa(row: Dict[str, Any]) -> str:
    x = row.get("x", "")
    pa = row.get("pa_plus") or {}
    persona = pa.get("persona") or []
    act = pa.get("act", "")
    header = serialize_pa(persona, act)
    return ensure_bres(f"{x}\n\n{header}\n")

def clean_span(text: str) -> str:
    s = text.find("[BRES]")
    e = text.find("[ERES]", s + 6) if s != -1 else -1
    core = text[s+6:e].strip() if (s != -1 and e != -1) else text.strip()
    core = core.strip().strip('"').strip()
    return f"[BRES] {core} [ERES]"

def load_model(model_dir: str, force_cpu=False):
    use_cuda = torch.cuda.is_available() and not force_cpu
    device_map = "auto" if use_cuda else None
    dtype = (torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported()
             else (torch.float16 if use_cuda else torch.float32))
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    tok = add_special_tokens(tok)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map=device_map, torch_dtype=dtype, trust_remote_code=True
    )
    if model.get_input_embeddings().num_embeddings != len(tok):
        model.resize_token_embeddings(len(tok))
    model.eval()
    return tok, model

def run_infer(args):
    ds = load_dataset("json", data_files=args.test_jsonl, split="train")
    tok, model = load_model(args.model_dir, force_cpu=args.force_cpu)

    # Context window of the base LM (GPT-2 usually 1024)
    ctx_limit = get_ctx_limit(model)

    # We will dynamically cap the prompt length per example so that
    # prompt_len + max_new_tokens <= ctx_limit
    def safe_prompt_max_len(requested_prompt_max: int, max_new_tokens: int) -> int:
        # leave a tiny cushion
        return max(16, min(requested_prompt_max, ctx_limit - max_new_tokens - 1))

    eres_id = tok.convert_tokens_to_ids("[ERES]")

    do_sample = not args.no_sample
    gen_kwargs_template = build_gen_kwargs(tok, eres_id, args, do_sample=do_sample)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)
    wrote = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for row in ds:
            prompt = build_prompt_pa(row)

            # First, tokenize to see current prompt tokens if we used requested max
            # We’ll re-encode with a safe cap next.
            # Compute a *per-row* safe cap so we never exceed the LM context.
            safe_cap = safe_prompt_max_len(args.max_prompt_length, args.max_new_tokens)

            # If the prompt is still too long after truncation, we may also need to
            # clamp max_new_tokens to avoid overflow.
            # Encode once without truncation to estimate true length:
            tmp_ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            prompt_len = tmp_ids.shape[-1]
            if prompt_len + args.max_new_tokens >= ctx_limit:
                # Reduce generation length so total fits; keep at least 8 new tokens.
                new_max_new = max(8, ctx_limit - 1 - min(prompt_len, safe_cap))
                # Update generation kwargs accordingly (copy to avoid mutating template)
                gen_kwargs = dict(gen_kwargs_template)
                gen_kwargs["max_new_tokens"] = new_max_new
            else:
                gen_kwargs = gen_kwargs_template

            # Now do the final, safe encode with truncation
            enc = tok(prompt, add_special_tokens=False, truncation=True,
                      max_length=safe_cap, return_tensors="pt")
            enc = {k: v.to(model.device) for k,v in enc.items()}

            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)

            tail = out[0, enc["input_ids"].shape[-1]:]
            raw = tok.decode(tail, skip_special_tokens=False)
            resp = clean_span(raw)

            rec = {k: row[k] for k in row.keys()}
            rec["PromptUsed"] = prompt
            rec["RawTail"] = raw
            rec["Response"] = resp
            rec["Meta"] = {
                "ctx_limit": ctx_limit,
                "safe_cap": safe_cap,
                "used_max_new_tokens": gen_kwargs["max_new_tokens"],
                "do_sample": do_sample,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1

            if wrote % 5 == 0:
                print(f"→ {wrote} examples processed... (ctx={ctx_limit}, safe_cap={safe_cap})")

    print(f"✅ Wrote {wrote} generations → {args.out_jsonl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)

    ap.add_argument("--max_prompt_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--no_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)

    ap.add_argument("--force_cpu", action="store_true")

    args = ap.parse_args()
    run_infer(args)

if __name__ == "__main__":
    main()
