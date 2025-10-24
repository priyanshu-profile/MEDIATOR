# eval_pa_hal_rlen.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, math, argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt', quiet=True)
BRES, ERES = "[BRES]", "[ERES]"

# ---------------- utilities ----------------

def extract_response_span(text: str) -> str:
    s = text.find(BRES); e = text.find(ERES, s+6) if s != -1 else -1
    core = text[s+6:e].strip() if (s != -1 and e != -1) else text.strip()
    return core.strip().strip('"')

def split_sentences(text: str) -> List[str]:
    sents = sent_tokenize(text)
    return [re.sub(r"\s+", " ", s).strip() for s in sents if s and s.strip()]

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def verbalize_triplet(t: str) -> str:
    m = re.match(r"\(?\s*([^,]+)\s*,\s*([^,]+)\s*,\s*(.+?)\)?\s*$", t or "")
    if not m: return (t or "").strip()
    s, p, o = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    s = s.lower()
    if not s.startswith("the "): s = "the " + s
    p = p.replace("_", " ").strip()
    return f"{s} {p} {o}."

def persona_sentences(persona_list: List[str]) -> List[str]:
    return [verbalize_triplet(p) for p in persona_list or [] if str(p).strip()]

# --------------- NLI wrapper ----------------

class NLIScorer:
    def __init__(self, model_name="roberta-large-mnli", device=None):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = {0:"contradict", 1:"neutral", 2:"entail"}

    @torch.no_grad()
    def nli(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        enc = self.tok(premise, hypothesis, truncation=True, max_length=384, return_tensors="pt")
        enc = {k:v.to(self.device) for k,v in enc.items()}
        out = self.model(**enc)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0)
        lab = self.id2label[int(torch.argmax(probs))]
        return lab, float(torch.max(probs))

# --------------- Act classifier wrapper ---------------

class ActClassifier:
    def __init__(self, ckpt_dir: str):
        self.tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, float]:
        enc = self.tok(text, truncation=True, max_length=384, return_tensors="pt")
        enc = {k:v.to(self.device) for k,v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pid = int(torch.argmax(probs))
        return self.id2label[pid], float(probs[pid])

# --------------- Metrics ----------------

def pa_consistency(
    gold_act: str,
    persona_sents: List[str],
    gen_text: str,
    act_clf: ActClassifier,
    nli: NLIScorer,
    allow_k_contradictions: int = 0,
    act_prob_threshold: float = 0.5,
    soft: bool = False,
    lambda_soft: float = 1.0
):
    pred_act, prob = act_clf.predict(gen_text)
    act_ok = (pred_act == gold_act) and (prob >= act_prob_threshold)

    contradictions = 0
    for p in persona_sents:
        label, _ = nli.nli(p, gen_text)
        if label == "contradict":
            contradictions += 1

    persona_ok = (contradictions <= allow_k_contradictions)

    if not soft:
        score = 1.0 if (act_ok and persona_ok) else 0.0
    else:
        score = (prob if (pred_act == gold_act) else 0.0) * math.exp(-lambda_soft * contradictions)

    return score, {"pred_act": pred_act, "act_prob": prob, "act_ok": act_ok, "persona_contradictions": contradictions}

def hal_sent(gen_text: str, support_sents: List[str], nli: NLIScorer) -> Tuple[float, dict]:
    sents = split_sentences(gen_text)
    if not sents:
        return 0.0, {"num_sents": 0, "unsupported": 0}
    unsupported = 0
    for s in sents:
        entailed = False
        for prem in support_sents:
            lab, _ = nli.nli(prem, s)
            if lab == "entail":
                entailed = True; break
        if not entailed:
            unsupported += 1
    val = 100.0 * (unsupported / len(sents))
    return val, {"num_sents": len(sents), "unsupported": unsupported}

def asr(gold_act: str, gen_text: str, act_clf: ActClassifier, prob_threshold: float=0.5) -> Tuple[float, dict]:
    pred, prob = act_clf.predict(gen_text)
    ok = (pred == gold_act) and (prob >= prob_threshold)
    return (100.0 if ok else 0.0), {"pred_act": pred, "act_prob": prob, "ok": ok}

def r_len(gen_text: str) -> int:
    return len(gen_text.split())

# --------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_jsonl", required=True, help="Your model generations JSONL")
    ap.add_argument("--resp_field", default="Response", help="Field with generated text")
    ap.add_argument("--act_ckpt", required=True, help="Act classifier ckpt dir")
    ap.add_argument("--nli_model", default="roberta-large-mnli")
    ap.add_argument("--alpha", type=float, default=0.7, help="alpha for HAL_composite")
    ap.add_argument("--allow_k_contradictions", type=int, default=0)
    ap.add_argument("--act_prob_threshold", type=float, default=0.5)
    ap.add_argument("--use_soft_pa", action="store_true")
    ap.add_argument("--lambda_soft", type=float, default=1.0)
    ap.add_argument("--use_history_support", action="store_true", help="also add x (history) to support pool")
    args = ap.parse_args()

    rows = load_jsonl(args.gen_jsonl)
    act_clf = ActClassifier(args.act_ckpt)
    nli = NLIScorer(args.nli_model)

    pa_hard, pa_soft, hal_sents, asrs, r_lens = [], [], [], [], []
    dbg = []

    for i, r in enumerate(rows):
        pa = r.get("pa_plus", {}) or {}
        gold_act = str(pa.get("act","")).strip()
        persona = pa.get("persona", []) or []

        persona_sents = persona_sentences(persona)
        support = persona_sents.copy()
        if args.use_history_support:
            support += split_sentences(str(r.get("x", "")))

        gen_raw = str(r.get(args.resp_field, "")).strip()
        gen = extract_response_span(gen_raw)

        # PA-Consistency
        pa_score, pa_info = pa_consistency(
            gold_act=gold_act,
            persona_sents=persona_sents,
            gen_text=gen,
            act_clf=act_clf,
            nli=nli,
            allow_k_contradictions=args.allow_k_contradictions,
            act_prob_threshold=args.act_prob_threshold,
            soft=False
        )
        pa_hard.append(pa_score)

        if args.use_soft_pa:
            s_score, _ = pa_consistency(
                gold_act=gold_act,
                persona_sents=persona_sents,
                gen_text=gen,
                act_clf=act_clf,
                nli=nli,
                allow_k_contradictions=999999,
                act_prob_threshold=0.0,
                soft=True,
                lambda_soft=args.lambda_soft
            )
            pa_soft.append(s_score)

        # HAL_sent (content)
        hal_s, hal_info = hal_sent(gen, support, nli)
        hal_sents.append(hal_s)

        # ASR (act support rate)
        asr_val, asr_info = asr(gold_act, gen, act_clf, prob_threshold=args.act_prob_threshold)
        asrs.append(asr_val)

        # R-LEN
        r_lens.append(r_len(gen))

        dbg.append({
            "idx": i,
            "gold_act": gold_act,
            "pred_act": pa_info["pred_act"],
            "act_prob": round(pa_info["act_prob"], 4),
            "act_ok": pa_info["act_ok"],
            "persona_contradictions": pa_info["persona_contradictions"],
            "HAL_sent": round(hal_s, 2),
            "ASR": asr_val,
            "R_LEN": r_lens[-1]
        })

    out = {
        "Count": len(rows),
        "PA_Consistency": float(np.mean(pa_hard)) if pa_hard else 0.0,
        **({"PA_Consistency_soft": float(np.mean(pa_soft))} if pa_soft else {}),
        "HAL_sent_mean": float(np.mean(hal_sents)) if hal_sents else 0.0,
        "ASR_mean": float(np.mean(asrs)) if asrs else 0.0,
        "HAL_composite(alpha={:.2f})".format(args.alpha):
            float(args.alpha * (np.mean(hal_sents) if hal_sents else 0.0) +
                  (1-args.alpha) * (100.0 - (np.mean(asrs) if asrs else 0.0))),
        "R_LEN_mean": float(np.mean(r_lens)) if r_lens else 0.0
    }

    print(json.dumps(out, indent=2))
    dbg_path = os.path.splitext(args.gen_jsonl)[0] + ".pa_hal_dbg.jsonl"
    with open(dbg_path, "w", encoding="utf-8") as f:
        for d in dbg:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("Per-item diagnostics:", dbg_path)

if __name__ == "__main__":
    main()
