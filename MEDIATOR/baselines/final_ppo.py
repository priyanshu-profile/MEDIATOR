#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast Debug PPO (TRL 0.11.3) — CPU-friendly

- Uses distilgpt2 (tiny) instead of Qwen2.5-3B.
- Very small batch/epoch settings.
- Short prompt/response lengths to keep runtime short.
- Runs end-to-end PPO loop in <5 minutes on CPU.

Run:
python3 scripts/final_ppo_cpu_debug.py \
  --train_jsonl pref_data/final_train_pref.jsonl \
  --output_dir runs/ppo_debug_cpu \
  --use_sbert --do_sample
"""

import os, json, argparse, random, re
from typing import Dict, Any, List

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 🔒 Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda : False
torch.set_default_device("cpu")

# SBERT
_HAS_SBERT = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    _HAS_SBERT = False

SPECIAL_TOKENS = ["[BOS]","[EOS]","[BPER]","[EPER]","[BACT]","[EACT]","[BRES]","[ERES]"]

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

def add_special_tokens(tok):
    add = [t for t in SPECIAL_TOKENS if t not in tok.get_vocab()]
    if add:
        tok.add_special_tokens({"additional_special_tokens": add})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    tok.padding_side = "left"
    return tok

def span_inner(text: str) -> str:
    t = (text or "").strip()
    s = t.find("[BRES]")
    e = t.find("[ERES]", s + 6) if s != -1 else -1
    core = t[s+6:e].strip() if (s != -1 and e != -1) else t
    return core.strip().strip('"').strip()

def ensure_prompt_has_bres(prompt: str) -> str:
    s = (prompt or "").rstrip()
    if not s.endswith("[BRES]"):
        s = s + ("\n" if not s.endswith("\n") else "") + "[BRES]"
    return s

def cut_at_eres(decoded: str) -> str:
    i = decoded.find("[ERES]")
    return decoded[: i + len("[ERES]")] if i != -1 else decoded

def load_pref_jsonl(path: str, limit:int=20) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            r = json.loads(line)
            x  = str(r.get("x","")).strip()
            yp = span_inner(r.get("y_plus",""))
            ym = span_inner(r.get("y_minus",""))
            if not x or not yp or not ym:
                continue
            rows.append({"query": x, "y_plus": yp, "y_minus": ym})
    return rows

# -------------------------
# Reward
# -------------------------
class RewardComputer:
    def __init__(self, use_sbert: bool = True, sbert_model: str = "all-MiniLM-L6-v2"):
        self.use_sbert = use_sbert and _HAS_SBERT
        self.device = "cpu"
        if self.use_sbert:
            self.sbert = SentenceTransformer(sbert_model, device=self.device)
        else:
            self.sbert = None

    def pairwise_reward(self, gen: List[str], pos: List[str], neg: List[str]) -> torch.Tensor:
        vals = []
        if self.sbert:
            with torch.no_grad():
                eg = self.sbert.encode(gen, normalize_embeddings=True, convert_to_tensor=True)
                ep = self.sbert.encode(pos, normalize_embeddings=True, convert_to_tensor=True)
                en = self.sbert.encode(neg, normalize_embeddings=True, convert_to_tensor=True)
                sim_pos = (eg * ep).sum(dim=1)
                sim_neg = (eg * en).sum(dim=1)
                return (sim_pos - sim_neg).to(torch.float32)
        else:
            for g, p, n in zip(gen, pos, neg):
                sp = len(set(g.split()) & set(p.split())) / max(1, len(set(g.split())))
                sn = len(set(g.split()) & set(n.split())) / max(1, len(set(g.split())))
                vals.append(sp - sn)
            return torch.tensor(vals, dtype=torch.float32)

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_model_dir", default="distilgpt2", help="use distilgpt2 for CPU debug")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--mini_batch_size", type=int, default=1)
    ap.add_argument("--ppo_epochs", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--target_kl", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_prompt_length", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--use_sbert", action="store_true")
    return ap.parse_args()

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data (limit for CPU speed)
    data = load_pref_jsonl(args.train_jsonl, limit=20)
    if not data:
        raise RuntimeError("No valid rows found in --train_jsonl")
    ds = Dataset.from_list(data)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.sft_model_dir, use_fast=False)
    tok = add_special_tokens(tok)
    eres_id = tok.convert_tokens_to_ids("[ERES]")

    # Policy
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(args.sft_model_dir)
    policy.to("cpu")
    policy.pretrained_model.resize_token_embeddings(len(tok))
    policy.pretrained_model.config.use_cache = False

    # PPO config
    ppo_cfg = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        target_kl=args.target_kl,
        log_with=None,
        seed=args.seed,
    )

    trainer = PPOTrainer(
        config=ppo_cfg,
        model=policy,
        ref_model=None,
        tokenizer=tok,
    )

    rewarder = RewardComputer(use_sbert=args.use_sbert)
    device = "cpu"
    policy.eval()

    def build_prompts(qs: List[str]) -> List[str]:
        return [ensure_prompt_has_bres(q) for q in qs]

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "eos_token_id": [tok.eos_token_id, eres_id] if eres_id is not None else tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
    }

    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.ppo_epochs):
        for batch in train_loader:
            queries   = batch["query"]
            y_pluses  = batch["y_plus"]
            y_minuses = batch["y_minus"]

            prompts = build_prompts(queries)
            enc = tok(prompts, add_special_tokens=False, truncation=True,
                      max_length=args.max_prompt_length, padding=True, return_tensors="pt")
            query_tensors = [enc["input_ids"][i].to(device) for i in range(enc["input_ids"].size(0))]

            response_tensors = trainer.generate(query_tensors, **gen_kwargs)

            decoded_responses = [cut_at_eres(tok.decode(resp, skip_special_tokens=False)) for resp in response_tensors]
            decoded_responses = [span_inner(txt) for txt in decoded_responses]

            rewards = rewarder.pairwise_reward(decoded_responses, y_pluses, y_minuses)
            rewards_list = [torch.tensor([r], device=device, dtype=torch.float32) for r in rewards]

            stats = trainer.step(query_tensors, response_tensors, rewards_list)
            print(f"[Epoch {epoch}] Stats:", stats)

    policy.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("✅ PPO debug training complete on CPU →", args.output_dir)


if __name__ == "__main__":
    main()

