#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NVML guard must be set before importing torch (keeps your 460 driver happy)
import os
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

import re
import json
import argparse
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

SPECIAL_TOKENS = ["[BOS]","[EOS]","[BPER]","[EPER]","[BACT]","[EACT]","[BRES]","[ERES]"]

def add_special_tokens(tok):
    add = [t for t in SPECIAL_TOKENS if t not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def ensure_bres(s: str) -> str:
    s = (s or "").rstrip()
    if not s.endswith("[BRES]"):
        s = s + ("\n" if not s.endswith("\n") else "") + "[BRES]"
    return s

def serialize_pa(persona_list: List[str], act_label: str) -> str:
    persona_str = " ".join(p.strip() for p in (persona_list or []))
    act_label = (act_label or "general_response").strip()
    return f"[BPER] {persona_str} [EPER] [BACT] {act_label} [EACT]"

def build_prompt(row: Dict[str, Any]) -> str:
    # x + PA+ header + [BRES]
    x = row["x"]
    pa = row.get("pa_plus") or {}
    persona = pa.get("persona") or []
    act = pa.get("act") or ""
    header = serialize_pa(persona, act)
    return ensure_bres(f"{x}\n\n{header}\n")

def clean_span(text: str) -> str:
    # keep only the span inside [BRES]...[ERES]
    s = text.find("[BRES]")
    e = text.find("[ERES]", s+6) if s != -1 else -1
    core = text[s+6:e].strip() if (s != -1 and e != -1) else text.strip()
    # strip mismatched quotes and trailing junk spaces
    core = core.strip().strip('"').strip()
    # optional: drop non-ascii leftovers (toggle by uncommenting)
    # core = re.sub(r'[^\x00-\x7F]+', ' ', core)
    return f"[BRES] {core} [ERES]"

def build_bad_words_ids(tokenizer, ban_cjk=True):
    # ban obvious junk & meta tokens
    ban_strings = [
        "vinfos", "<|im_start|>", "<|im_end|>", "<tool_call>", "general_response",
        "[BOS]","[EOS]","[BPER]","[EPER]","[BACT]","[EACT]","[BRES]"  # allow only [ERES] in the tail
    ]
    bad = tokenizer(ban_strings, add_special_tokens=False).input_ids

    if ban_cjk:
        # also ban any vocab piece containing CJK/Japanese/Korean chars
        pat = re.compile(r'[\u3400-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]')
        for tok, tid in tokenizer.get_vocab().items():
            if pat.search(tok):
                bad.append([tid])
    return bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_prompt_length", type=int, default=1400)
    ap.add_argument("--max_new_tokens", type=int, default=196)
    ap.add_argument("--repetition_penalty", type=float, default=1.12)
    ap.add_argument("--no_cuda", action="store_true")
    ap.add_argument("--ban_cjk", action="store_true")  # on by default
    args = ap.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"[DEVICE] CUDA={'yes' if use_cuda else 'no'} | dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
    tok = add_special_tokens(tok)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map=("auto" if use_cuda else None),
        torch_dtype=(torch.float16 if use_cuda else torch.float32),
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    ds = load_dataset("json", data_files=args.test_jsonl, split="train")
    bad_words_ids = build_bad_words_ids(tok, ban_cjk=args.ban_cjk)
    eres_id = tok.convert_tokens_to_ids("[ERES]")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)) or ".", exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for row in ds:
            prompt = build_prompt(row)

            enc = tok(
                prompt, add_special_tokens=False, truncation=True,
                max_length=args.max_prompt_length, return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=False,                # deterministic
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=4,
                renormalize_logits=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=[tok.eos_token_id, eres_id] if eres_id is not None else tok.eos_token_id,
                bad_words_ids=bad_words_ids,
                use_cache=True,
            )

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **{k: v for k, v in gen_kwargs.items() if v is not None}
                )

            tail = out[0, input_ids.shape[-1]:]
            full = tok.decode(tail, skip_special_tokens=False)
            final_msg = clean_span(full)

            rec = dict(row)
            rec["PromptUsed"] = prompt
            rec["RawTail"] = full
            rec["Response"] = final_msg
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved generations to {args.out_jsonl}")

if __name__ == "__main__":
    main()
