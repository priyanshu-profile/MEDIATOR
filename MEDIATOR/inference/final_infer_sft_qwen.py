#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for SFT-trained Qwen models.

- Loads a fine-tuned CausalLM model (e.g., Qwen2.5-3B-SFT).
- Builds prompts with persona+act conditioning (like PPO inference).
- Generates responses with [BRES] ... [ERES] spans.
- Saves JSONL output aligned with input rows.

Usage:
python3 scripts/infer_sft_qwen.py \
  --model_dir final_qwen_sft/best_merged \
  --test_jsonl pref_data/final_test_pref.jsonl \
  --out_jsonl final_outputs/qwen_sft_preds.jsonl \
  --force_cpu --no_sample
"""

import os, json, argparse, torch
from typing import Dict, Any, List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

SPECIAL_TOKENS = ["[BOS]","[EOS]","[BPER]","[EPER]",
                  "[BACT]","[EACT]","[BRES]","[ERES]"]

# ---------------- utility helpers ----------------
def add_special_tokens(tok):
    add = [t for t in SPECIAL_TOKENS if t not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    tok.padding_side = "left"
    return tok

def serialize_pa(persona_list: List[str], act_label: str) -> str:
    persona_str = " ".join(p.strip() for p in (persona_list or []))
    act_label = (act_label or "general_response").strip()
    return f"[BPER] {persona_str} [EPER] [BACT] {act_label} [EACT]"

def ensure_bres(s: str) -> str:
    s = (s or "").rstrip()
    return s + ("\n" if not s.endswith("\n") else "") + "[BRES]"

def build_prompt_pa(row: Dict[str, Any]) -> str:
    """Build persona+act-conditioned prompt for SFT model."""
    x = row.get("x", "")
    pa = row.get("pa_plus") or {}
    persona = pa.get("persona") or []
    act = pa.get("act", "")
    header = serialize_pa(persona, act)
    return ensure_bres(f"{x}\n\n{header}\n")

def clean_span(text: str) -> str:
    """Extract span between [BRES] ... [ERES]."""
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

# ---------------- main inference ----------------
def run_infer(args):
    ds = load_dataset("json", data_files=args.test_jsonl, split="train")
    tok, model = load_model(args.model_dir, force_cpu=args.force_cpu)

    eres_id = tok.convert_tokens_to_ids("[ERES]")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams if args.no_sample else 1,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tok.pad_token_id,
        eos_token_id=[tok.eos_token_id, eres_id] if eres_id else tok.eos_token_id,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)
    wrote = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for row in ds:
            prompt = build_prompt_pa(row)
            enc = tok(prompt, add_special_tokens=False, truncation=True,
                      max_length=args.max_prompt_length, return_tensors="pt")
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
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1

            if wrote % 5 == 0:
                print(f"→ {wrote} examples processed...")

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
