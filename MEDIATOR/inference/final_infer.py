#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Selective multi-model inference.

Runs only the models you specify and writes a separate JSONL for each one.
- PA-DPO uses PA-conditioned prompting:  x + serialize_pa(pa_plus) + [BRES]
- DPO/KTO use plain prompting by default: x + [BRES]

Example:
python3 scripts/final_infer.py \
  --test_jsonl pref_data/final_test_pref.jsonl \
  --pa_dpo_dir final_runs/pa_dpo_qwen2.5-3b/epoch_1 --pa_dpo_out final_outputs/pa_dpo.jsonl \
  --dpo_dir    runs/vanilla_dpo_qwen25_3b/epoch_1 --dpo_out    outputs/dpo.jsonl \
  --kto_dir    runs/vanilla_kto_qwen25_3b/epoch_1 --kto_out    outputs/kto.jsonl \
  --max_prompt_length 1400 --max_new_tokens 196 --no_sample
"""

import os, json, argparse
from typing import Dict, Any, List
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- special tokens ----------
SPECIAL_TOKENS = ["[BOS]","[EOS]","[BPER]","[EPER]","[BACT]","[EACT]","[BRES]","[ERES]"]

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
    x = row.get("x", "")
    pa = row.get("pa_plus") or {}
    persona = pa.get("persona") or []
    act = pa.get("act", "")
    header = serialize_pa(persona, act)
    return ensure_bres(f"{x}\n\n{header}\n")

def build_prompt_plain(row: Dict[str, Any]) -> str:
    return ensure_bres(row.get("x", ""))

def clean_span(text: str) -> str:
    s = text.find("[BRES]")
    e = text.find("[ERES]", s + 6) if s != -1 else -1
    core = text[s+6:e].strip() if (s != -1 and e != -1) else text.strip()
    core = core.strip().strip('"').strip()
    return f"[BRES] {core} [ERES]"

def build_bad_words_ids(tokenizer):
    # Forbid scaffolding tokens (allow [ERES] so the span can close)
    ban_strings = [
        "vinfos", "<|im_start|>", "<|im_end|>", "<tool_call>", "general_response",
        "[BOS]","[EOS]","[BPER]","[EPER]","[BACT]","[EACT]","[BRES]"
    ]
    ids = tokenizer(ban_strings, add_special_tokens=False).input_ids
    return ids

def load_model(model_dir: str):
    use_cuda = torch.cuda.is_available()
    device_map = "auto" if use_cuda else None
    dtype = (torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported()
             else (torch.float16 if use_cuda else torch.float32))
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    tok = add_special_tokens(tok)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map=device_map, torch_dtype=dtype, trust_remote_code=True
    )
    # keep tokenizer/model in sync
    try:
        if model.get_input_embeddings().num_embeddings != len(tok):
            model.resize_token_embeddings(len(tok))
    except Exception:
        pass
    model.eval()
    return tok, model

def generate(model, tok, prompt: str, max_prompt_length: int, gen_cfg: Dict[str, Any]) -> Dict[str, str]:
    enc = tok(prompt, add_special_tokens=False, truncation=True,
              max_length=max_prompt_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    eres_id = tok.convert_tokens_to_ids("[ERES]")
    bad_words_ids = build_bad_words_ids(tok)

    kwargs = dict(
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        temperature=(gen_cfg["temperature"] if gen_cfg["do_sample"] else None),
        top_p=(gen_cfg["top_p"] if gen_cfg["do_sample"] else None),
        top_k=(gen_cfg["top_k"] if gen_cfg["do_sample"] else None),
        num_beams=(gen_cfg["num_beams"] if not gen_cfg["do_sample"] else 1),
        repetition_penalty=gen_cfg["repetition_penalty"],
        no_repeat_ngram_size=gen_cfg["no_repeat_ngram_size"],
        renormalize_logits=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=[tok.eos_token_id, eres_id] if eres_id is not None else tok.eos_token_id,
        bad_words_ids=bad_words_ids if bad_words_ids else None,
        use_cache=True,
    )

    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                             **{k:v for k,v in kwargs.items() if v is not None})
    tail = out[0, input_ids.shape[-1]:]
    raw = tok.decode(tail, skip_special_tokens=False)
    return {"RawTail": raw, "Response": clean_span(raw)}

def run_one_model(name: str, model_dir: str, out_path: str,
                  ds, prompt_mode: str, max_prompt_length: int, gen_cfg: Dict[str, Any]):
    print(f"[{name}] loading: {model_dir}")
    tok, mdl = load_model(model_dir)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    build_prompt = build_prompt_pa if prompt_mode == "pa" else build_prompt_plain

    wrote = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in ds:
            prompt = build_prompt(row)
            gen = generate(mdl, tok, prompt, max_prompt_length, gen_cfg)
            rec = {k: row[k] for k in row.keys()}  # keep original fields
            rec["PromptUsed"] = prompt
            rec["RawTail"] = gen["RawTail"]
            rec["Response"] = gen["Response"]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1
    print(f"[{name}] wrote {wrote} rows → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", required=True)

    # Optional models (run only those provided)
    ap.add_argument("--pa_dpo_dir", type=str, default=None)
    ap.add_argument("--pa_dpo_out", type=str, default=None)
    ap.add_argument("--dpo_dir", type=str, default=None)
    ap.add_argument("--dpo_out", type=str, default=None)
    ap.add_argument("--kto_dir", type=str, default=None)
    ap.add_argument("--kto_out", type=str, default=None)

    # Prompt & decoding settings
    ap.add_argument("--max_prompt_length", type=int, default=1400)
    ap.add_argument("--max_new_tokens",    type=int, default=196)
    ap.add_argument("--no_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p",      type=float, default=0.9)
    ap.add_argument("--top_k",      type=int,   default=50)
    ap.add_argument("--num_beams",  type=int,   default=1)
    ap.add_argument("--repetition_penalty", type=float, default=1.12)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    # Optional: force prompt modes for baselines
    ap.add_argument("--dpo_mode", choices=["plain","pa"], default="plain")
    ap.add_argument("--kto_mode", choices=["plain","pa"], default="plain")

    args = ap.parse_args()
    ds = load_dataset("json", data_files=args.test_jsonl, split="train")

    gen_cfg = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    any_run = False

    if args.pa_dpo_dir and args.pa_dpo_out:
        run_one_model(
            name="PA-DPO",
            model_dir=args.pa_dpo_dir,
            out_path=args.pa_dpo_out,
            ds=ds,
            prompt_mode="pa",               # PA-conditioned
            max_prompt_length=args.max_prompt_length,
            gen_cfg=gen_cfg,
        )
        any_run = True

    if args.dpo_dir and args.dpo_out:
        run_one_model(
            name="DPO",
            model_dir=args.dpo_dir,
            out_path=args.dpo_out,
            ds=ds,
            prompt_mode=args.dpo_mode,      # default plain; can set --dpo_mode pa
            max_prompt_length=args.max_prompt_length,
            gen_cfg=gen_cfg,
        )
        any_run = True

    if args.kto_dir and args.kto_out:
        run_one_model(
            name="KTO",
            model_dir=args.kto_dir,
            out_path=args.kto_out,
            ds=ds,
            prompt_mode=args.kto_mode,      # default plain; can set --kto_mode pa
            max_prompt_length=args.max_prompt_length,
            gen_cfg=gen_cfg,
        )
        any_run = True

    if not any_run:
        raise SystemExit("No models selected. Provide *_dir and *_out for at least one model.")

if __name__ == "__main__":
    main()
