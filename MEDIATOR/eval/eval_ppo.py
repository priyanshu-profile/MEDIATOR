#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for ABN generations.

Computes:
- BLEU-4
- Distinct-3
- BERTScore-F1
- Perplexity (mean)

Input: JSONL with keys "Response" (model output) and "y_plus" (gold)
"""

import os, json, argparse
import math
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import bert_score

# -------------------------
# Utils
# -------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "Response" in r and "y_plus" in r:
                data.append(r)
    return data

def tokenize_for_bleu(s):
    return s.replace('"', "").replace("'", "").split()

def calc_bleu4(data):
    refs = [[tokenize_for_bleu(d["y_plus"])] for d in data]
    hyps = [tokenize_for_bleu(d["Response"]) for d in data]
    smoothie = SmoothingFunction().method4
    score = corpus_bleu(refs, hyps, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)
    return score

def calc_distinct_n(texts, n=3):
    all_ngrams = []
    for txt in texts:
        toks = txt.split()
        ngrams = list(zip(*[toks[i:] for i in range(n)]))
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

def calc_bertscore(data, lang="en", model="microsoft/deberta-xlarge-mnli"):
    refs = [d["y_plus"] for d in data]
    hyps = [d["Response"] for d in data]
    P, R, F1 = bert_score.score(hyps, refs, lang=lang, model_type=model, verbose=True)
    return float(F1.mean().item())

def calc_ppl(data, model_name="gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    mdl.eval()

    ppl_list = []
    with torch.no_grad():
        for d in tqdm(data, desc="PPL"):
            enc = tok(d["Response"], return_tensors="pt").to(device)
            out = mdl(**enc, labels=enc["input_ids"])
            loss = out.loss.item()
            ppl = math.exp(loss) if loss < 20 else float("inf")
            ppl_list.append(ppl)
    return float(np.mean(ppl_list))

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True, help="Model predictions JSONL with Response and y_plus")
    ap.add_argument("--out_path", type=str, default=None, help="Optional save results JSON")
    ap.add_argument("--ppl_model", type=str, default="gpt2", help="Model for perplexity (default gpt2)")
    args = ap.parse_args()

    data = load_jsonl(args.pred_jsonl)
    print(f"Loaded {len(data)} examples")

    bleu4 = calc_bleu4(data)
    distinct3 = calc_distinct_n([d["Response"] for d in data], n=3)
    bert_f1 = calc_bertscore(data)
    ppl = calc_ppl(data, model_name=args.ppl_model)

    results = {
        "BLEU-4": bleu4,
        "Distinct-3": distinct3,
        "BERTScore-F1": bert_f1,
        "PPL-Mean": ppl
    }

    print("==== Evaluation Results ====")
    for k,v in results.items():
        print(f"{k:15s}: {v:.4f}")

    if args.out_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved → {args.out_path}")

if __name__ == "__main__":
    main()
