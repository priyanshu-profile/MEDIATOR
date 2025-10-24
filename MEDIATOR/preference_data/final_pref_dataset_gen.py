#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rDPO preference dataset builder (RE-ALIGN adapted to dialogues; bulletproof)

- Gold = SFT(Context + [PERSONA]) reply between [BRES]...[ERES]
- Rejected = persona-conditioned REWRITE of Gold (no masks)
- Sanitization: English-only, no HTML/placeholders/|/MASK/meta, trim to 1–2 sentences
- Repetition guard + self-repair
"""

import os, re, ast, json, random, argparse, unicodedata
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Config
# ----------------------------
SPECIAL_TOKENS = ["[BOS]","[EOS]","[BPER]","[EPER]","[BACT]","[EACT]","[BRES]","[ERES]"]
NEUTRAL_AGENT_ACT = "general_response"
INST = ("You are a helpful travel booking assistant. Given the dialogue history between an agent and a traveler, "
        "and the traveler's current turn, produce: (i) the traveler's relevant persona triplets, "
        "(ii) the agent's dialogue act, and (iii) the agent's next response. Use the special tokens exactly as specified.")

EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K_DEFAULT = 5
SIM_THRESHOLD_DEFAULT = 0.95
SEED = 42

# Generation
MAX_NEW_TOKENS = 256
MIN_NEW_TOKENS = 48
TEMPERATURE = 0.6          # slightly cooler to reduce babble
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.05

# Quality thresholds
MIN_CHARS = 40
REGEN_TRIES = 1
MAX_REPLY_CHARS = 600
STRICT_ENGLISH_RATIO = 0.8

# Persona cleaning
MAX_PERSONA_ITEMS = 8
MAX_TOKEN_REPEAT = 3

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ----------------------------
# CSV & conversation helpers
# ----------------------------
def safe_parse_list(x):
    if pd.isna(x): return []
    if isinstance(x, list): return x
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list): return v
    except Exception: pass
    return []

def build_conversations(df: pd.DataFrame, group_key: str) -> Dict[Any, List[Dict[str, Any]]]:
    if group_key not in df.columns:
        if "conv_ids" in df.columns: group_key = "conv_ids"
        elif "conv_id" in df.columns: group_key = "conv_id"
        else: raise KeyError(f"Missing conversation id column '{group_key}'. Found: {list(df.columns)}")
    grouped = {}
    for cid, sub in df.groupby(group_key, sort=True):
        grouped[cid] = sub.to_dict("records")
    return grouped

def select_first_with_persona(grouped: Dict[Any, List[Dict[str, Any]]], limit: int) -> List[Any]:
    out = []
    for cid in sorted(grouped.keys()):
        items = grouped[cid]
        if not items: continue
        first = items[0]
        pe = safe_parse_list(first.get("persona_evidences", None))
        pt = safe_parse_list(first.get("persona_triplets", None))
        if pe or pt:
            out.append(cid)
            if limit > 0 and len(out) >= limit: break
    return out

def format_turn(agent_act: Optional[str], agent_utt: Optional[str], traveler_utt: Optional[str]) -> str:
    if traveler_utt is not None:
        return traveler_utt.strip()
    act = agent_act.strip() if (agent_act and agent_act.strip()) else NEUTRAL_AGENT_ACT
    utt = (agent_utt or "").strip()
    return f"[BACT] {act} [EACT] {utt}".strip()

def build_context_samples_for_conv(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return {"Context": X} for each Traveler turn followed by an Agent response."""
    if not records: return []
    turns = []
    for r in records:
        spk = int(r.get("Speaker"))
        utt = str(r.get("Utterance", "")).strip()
        turns.append({"role": "agent" if spk==1 else "traveler", "text": utt})

    history, samples = [], []
    for i, t in enumerate(turns):
        if t["role"] == "agent":
            history.append(format_turn(NEUTRAL_AGENT_ACT, t["text"], None))
        else:
            if i + 1 < len(turns) and turns[i + 1]["role"] == "agent":
                H = "\n".join(history) if history else ""
                T_t = t["text"]
                X = INST + "\n\n" + (H + "\n" if H else "") + T_t
                samples.append({"Context": X})
            history.append(format_turn(None, None, t["text"]))
    return samples

# ----------------------------
# Persona cleaning & signature
# ----------------------------
def _squash_repeats(text: str, max_repeat=MAX_TOKEN_REPEAT) -> str:
    toks = text.split()
    out, last, run = [], None, 0
    for t in toks:
        if t == last:
            run += 1
            if run <= max_repeat: out.append(t)
        else:
            last, run = t, 1
            out.append(t)
    return " ".join(out)

def _strip_weird_chars(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-_/()[]{}$€₹%&'\"|")
    return "".join(ch for ch in text if ch in allowed or ch.isspace())

def clean_persona_triplets(triplets: List[str], evidences: List[str], max_items=MAX_PERSONA_ITEMS) -> str:
    pool = triplets if triplets else evidences
    cleaned, seen = [], set()
    for t in pool:
        t = unicodedata.normalize("NFKC", str(t))
        t = _squash_repeats(t)
        t = _strip_weird_chars(t)
        t = re.sub(r"\s+", " ", t).strip()
        if not t or len(t) < 6: continue
        if t.lower().startswith(("retrieved_persona","persona","mask")): continue
        key = t.lower()
        if key in seen: continue
        seen.add(key); cleaned.append(t)
        if len(cleaned) >= max_items: break
    return " | ".join(cleaned)

def persona_signature(triplets: List[str], evidences: List[str]) -> str:
    return clean_persona_triplets(triplets, evidences, max_items=16)

# ----------------------------
# Retrieval KB
# ----------------------------
def build_persona_index(grouped, embedder) -> Tuple[List[str], torch.Tensor, List[int]]:
    sigs, ids = [], []
    for cid in sorted(grouped.keys()):
        first = grouped[cid][0]
        pt = safe_parse_list(first.get("persona_triplets", []))
        pe = safe_parse_list(first.get("persona_evidences", []))
        sig = persona_signature(pt, pe)
        if sig.strip(): sigs.append(sig); ids.append(cid)
    if not sigs: return [], torch.empty(0), []
    with torch.no_grad():
        embs = embedder.encode(sigs, convert_to_tensor=True, normalize_embeddings=True)
    return sigs, embs, ids

def retrieve_similar(sig_query: str, embedder, kb_embs: torch.Tensor, kb_sigs: List[str], kb_ids: List[int],
                     exclude_id: int, top_k: int) -> List[Tuple[int,str,float]]:
    if not sig_query.strip() or kb_embs.numel()==0: return []
    q = embedder.encode(sig_query, convert_to_tensor=True, normalize_embeddings=True)
    cos = util.cos_sim(q, kb_embs).cpu().numpy()[0]
    order = np.argsort(-cos)
    out = []
    for idx in order:
        cid = kb_ids[idx]
        if cid == exclude_id: continue
        out.append((cid, kb_sigs[idx], float(cos[idx])))
        if len(out) >= top_k: break
    return out

# ----------------------------
# Generation utils
# ----------------------------
class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_sequences_ids: List[List[int]]): self.stop_sequences_ids = stop_sequences_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen = input_ids[0].tolist()
        for seq in self.stop_sequences_ids:
            if len(gen) >= len(seq) and gen[-len(seq):] == seq: return True
        return False

def _make_stoppers(tokenizer) -> StoppingCriteriaList:
    stop_ids = []
    for s in ["[ERES]", "\n[PERSONA]", "\n[BACT]", "\n[BPER]", "###", "<|im_end|>"]:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if ids: stop_ids.append(ids)
    return StoppingCriteriaList([StopOnSequences(stop_ids)])

def _decode_continuation(tokenizer, input_ids, gen_ids) -> str:
    in_len = input_ids.shape[1]
    cont = gen_ids[:, in_len:]
    return tokenizer.decode(cont[0], skip_special_tokens=False)

def _extract_bres(decoded_cont: str) -> str:
    s = decoded_cont.find("[BRES]")
    if s != -1:
        s += len("[BRES]")
        e = decoded_cont.find("[ERES]", s)
        if e == -1: e = len(decoded_cont)
        return decoded_cont[s:e].strip()
    return decoded_cont.replace("[ERES]", "").strip()

# --- sanitizers & validators ---
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
HTML_RE = re.compile(r"<[^>]+>")
MASK_RE = re.compile(r"\[MASK_\d+\]", re.IGNORECASE)
META_PAREN_RE = re.compile(r"\((?:[^)]*?(response|generation|strategy|placeholder|mask)[^)]*?)\)", re.I)
BANNED_TOKENS = {"generationstrategy","placeholder","template","tool_call","im_end","mask_","preservation","structure"}

def _strip_meta(s: str) -> str:
    s = META_PAREN_RE.sub(" ", s)
    s = re.sub(r"\s*\([^)]{0,80}\)$", " ", s)   # trailing small parens
    return s

def _sentence_trim(s: str, max_sent=2) -> str:
    parts = re.split(r'(?<=[.!?])\s+', s)
    kept = []
    for p in parts:
        p = p.strip()
        if len(p) < 5: continue
        kept.append(p)
        if len(kept) >= max_sent: break
    return " ".join(kept) if kept else s

def _is_repetitive(s: str) -> bool:
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", s)]
    if len(toks) < 6: return False
    from collections import Counter
    c = Counter(toks)
    most = c.most_common(3)
    total = sum(c.values())
    return (most[0][1] / total) > 0.4 or (sum(x[1] for x in most)/total) > 0.7

def _clean_reply(s: str) -> str:
    s = s.replace("\x00", " ")
    s = HTML_RE.sub(" ", s)
    s = re.sub(r"\[/?(BRES|ERES|BPER|EPER|BACT|EACT|BOS|EOS)\]", " ", s, flags=re.I)
    s = s.replace("general_response", " ")
    s = re.sub(r"\(Traveler,.*?\)", " ", s)
    s = s.replace("|", " ")
    s = MASK_RE.sub(" ", s)
    s = NON_ASCII_RE.sub(" ", s)  # enforce English-only by stripping non-ASCII
    s = _strip_meta(s)
    s = re.sub(r"\s+", " ", s).strip().strip('\'" ')
    s = _sentence_trim(s, max_sent=2)
    if len(s) > MAX_REPLY_CHARS: s = s[:MAX_REPLY_CHARS].rsplit(" ",1)[0]
    return s

def _looks_english(s: str) -> bool:
    if not s: return False
    letters = re.findall(r"[A-Za-z]", s)
    ratio = len(letters) / max(1, len(s))
    return ratio >= STRICT_ENGLISH_RATIO

def _bad_keywords(s: str) -> bool:
    low = s.lower()
    return any(k in low for k in BANNED_TOKENS)

def _finalize_reply(text: str) -> str:
    text = _clean_reply(text)
    # collapse short token runs
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        out.append(toks[i])
        run = 1
        while i + run < len(toks) and toks[i + run] == toks[i] and run < 4:
            run += 1
        i += run
    return " ".join(out)

def _good(s: str) -> bool:
    return bool(s and len(s) >= MIN_CHARS and _looks_english(s) and not _bad_keywords(s) and not _is_repetitive(s))

def repair_to_english(model, tokenizer, bad_reply: str, device: str) -> str:
    prompt = (
        "Rewrite the following agent message into clean, fluent ENGLISH ONLY addressing the traveler. "
        "Remove any placeholders, masks, HTML, boilerplate, or meta commentary. "
        "Return ONLY the message between [BRES] and [ERES]. Do NOT include the words 'strategy', 'generation', 'mask', 'placeholder', 'structure'.\n\n"
        f"[BRES]{bad_reply} [ERES]\n\n[BRES]"
    )
    fixed = generate_reply(model, tokenizer, prompt, device=device, tries=1)
    return _finalize_reply(fixed)

# ----------------------------
# Prompts (SFT-consistent)
# ----------------------------
def make_prompt_for_gold(context: str, persona_v: str) -> str:
    persona_block = f"\n\n[PERSONA]\n{persona_v}" if persona_v.strip() else ""
    return (
        f"{context}{persona_block}\n\n"
        "Reply in ENGLISH ONLY. Output ONLY the agent's message to the traveler, with no labels, persona, HTML, or meta commentary. "
        "Put the message strictly between [BRES] and [ERES].\n[BRES]"
    )

def make_rewrite_prompt(context: str, gold_reply: str, retrieved_persona: str) -> str:
    return (
        f"{context}\n\n[PERSONA]\n{retrieved_persona}\n\n"
        "Rewrite the agent's next message so it strongly reflects the above persona (preferences, exclusions, budget stance). "
        "ENGLISH ONLY. No labels, persona, HTML, or meta commentary. "
        "Return ONLY the final message between [BRES] and [ERES]. "
        "Do NOT include the words 'strategy', 'generation', 'mask', 'placeholder', or 'structure'.\n[BRES]"
        + gold_reply
    )

# ----------------------------
# Forced divergence (last resort)
# ----------------------------
PRICE_SUB_RE = re.compile(r"(?:USD|US\$|\$|INR|₹|EUR|€)\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)
def force_divergence(gold: str) -> str:
    m = PRICE_SUB_RE.search(gold)
    if m:
        orig = m.group(0)
        num = re.sub(r"[^\d.]", "", orig)
        try:
            val = float(num.replace(",", ""))
            rep_val = int(val * random.choice([0.85, 0.9, 1.1, 1.15]))
            rep = re.sub(r"\d[\d,]*", f"{rep_val:,}", orig)
            out = gold.replace(orig, rep, 1)
            return _finalize_reply(out)
        except: pass
    suffixes = [
        " If you prefer, we can exclude the photography tour to optimize the budget.",
        " We can also include a guided photography add-on if that excites you.",
        " To keep costs tighter, we’ll remove two extras you didn’t prioritize."
    ]
    g = gold.rstrip()
    if not g.endswith((".", "!", "?")): g += "."
    return _finalize_reply(g + " " + random.choice(suffixes))

# ----------------------------
# Generation (continuation-only, [ERES] stoppers)
# ----------------------------
class StopOnSeqs(StoppingCriteria):
    def __init__(self, ids): self.ids = ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen = input_ids[0].tolist()
        for seq in self.ids:
            if len(gen) >= len(seq) and gen[-len(seq):] == seq: return True
        return False

def _stoppers(tokenizer) -> StoppingCriteriaList:
    stop_ids = []
    for s in ["[ERES]", "\n[PERSONA]", "\n[BACT]", "\n[BPER]", "###", "<|im_end|>"]:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if ids: stop_ids.append(ids)
    return StoppingCriteriaList([StopOnSeqs(stop_ids)])

def _decode_cont(tokenizer, input_ids, gen_ids) -> str:
    in_len = input_ids.shape[1]
    cont = gen_ids[:, in_len:]
    return tokenizer.decode(cont[0], skip_special_tokens=False)

def generate_reply(model, tokenizer, prompt: str, device: str, tries=REGEN_TRIES) -> str:
    stoppers = _stoppers(tokenizer)
    best = ""
    for attempt in range(tries+1):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS if attempt==0 else 0,
                do_sample=True if attempt==0 else False,
                top_p=TOP_P, top_k=TOP_K, temperature=TEMPERATURE if attempt==0 else 1.0,
                repetition_penalty=REPETITION_PENALTY,
                stopping_criteria=stoppers,
            )
        dec = _decode_cont(tokenizer, inputs["input_ids"], gen)
        s = dec
        # extract between [BRES]... [ERES] if present
        i = s.find("[BRES]")
        if i != -1:
            i += len("[BRES]")
            j = s.find("[ERES]", i)
            if j == -1: j = len(s)
            s = s[i:j]
        s = _finalize_reply(s)
        if _good(s): return s
        best = s
    return best

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--group_key", type=str, default="conv_id", help="conv_id or conv_ids")
    ap.add_argument("--sft_model_dir", type=str, required=True)
    ap.add_argument("--out_rdpo", type=str, default="./rdpo_pairs.jsonl")
    ap.add_argument("--out_dpo",  type=str, default="./dpo_pairs.jsonl")
    ap.add_argument("--top_k", type=int, default=TOP_K_DEFAULT)
    ap.add_argument("--sim_threshold", type=float, default=SIM_THRESHOLD_DEFAULT)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--limit_persona_convs", type=int, default=5, help="-1 for all")
    ap.add_argument("--max_pairs", type=int, default=100000)
    ap.add_argument("--accept_if_not_equal", action="store_true",
                    help="Accept a non-empty candidate if it differs from Gold (ignores similarity).")
    args = ap.parse_args()

    # Load CSV & group
    df = pd.read_csv(args.csv_path, low_memory=False,
                     dtype={"persona_evidences":"string","persona_triplets":"string"})
    grouped = build_conversations(df, args.group_key)

    # Embeddings & KB
    embedder = SentenceTransformer(EMB_MODEL)
    kb_sigs, kb_embs, kb_ids = build_persona_index(grouped, embedder)

    # SFT model
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_dir, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens":[t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]})
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if (args.device=="auto" and torch.cuda.is_available()) else (args.device if args.device!="auto" else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_dir,
        torch_dtype=(torch.float16 if device=="cuda" else torch.float32),
        device_map="auto" if device=="cuda" else None,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    f_rdpo = open(args.out_rdpo, "w", encoding="utf-8")
    f_dpo  = open(args.out_dpo,  "w", encoding="utf-8")

    # Select persona conversations
    limit = args.limit_persona_convs
    selected = select_first_with_persona(grouped, limit if limit != -1 else 10**9)
    if not selected:
        print("No conversations with persona found.")
        f_rdpo.close(); f_dpo.close(); return

    c_written = 0
    dbg = dict(contexts_total=0, gold_repaired=0, rej_repaired=0, forced_divergence=0)

    for cid in tqdm(selected, desc="Building rDPO/DPO pairs"):
        recs = grouped[cid]
        first = recs[0]
        raw_trip = safe_parse_list(first.get("persona_triplets", []))
        raw_evid = safe_parse_list(first.get("persona_evidences", []))

        persona_v_clean = clean_persona_triplets(raw_trip, raw_evid, max_items=MAX_PERSONA_ITEMS)
        persona_sig_v   = persona_signature(raw_trip, raw_evid)

        contexts = build_context_samples_for_conv(recs)
        dbg["contexts_total"] += len(contexts)

        retrieved = retrieve_similar(persona_sig_v, embedder, kb_embs, kb_sigs, kb_ids,
                                     exclude_id=cid, top_k=args.top_k)

        for ex in contexts:
            if c_written >= args.max_pairs: break
            context = ex["Context"]

            # 1) Gold
            prompt_v = make_prompt_for_gold(context, persona_v_clean)
            gold = generate_reply(model, tokenizer, prompt_v, device=device)
            if not _good(gold):
                gold2 = repair_to_english(model, tokenizer, gold, device=device)
                if _good(gold2): gold = gold2; dbg["gold_repaired"] += 1
                else: continue

            # 2) Negative via persona-conditioned REWRITE
            rejected, persona_vl, cos_val = None, None, None
            used_retrieval = False

            for rid, r_sig_clean, sim_orig in retrieved or []:
                prompt_vl = make_rewrite_prompt(context, gold, r_sig_clean)
                cand = generate_reply(model, tokenizer, prompt_vl, device=device)
                if not _good(cand):
                    cand = repair_to_english(model, tokenizer, cand, device=device)
                    if _good(cand): dbg["rej_repaired"] += 1
                if not _good(cand):  # still bad
                    continue

                if not args.accept_if_not_equal:
                    with torch.no_grad():
                        e1 = embedder.encode([gold], convert_to_tensor=True, normalize_embeddings=True)
                        e2 = embedder.encode([cand], convert_to_tensor=True, normalize_embeddings=True)
                        cos_val = util.cos_sim(e1, e2).item()
                    if cand.strip() != gold.strip() and cos_val < args.sim_threshold:
                        rejected = cand; persona_vl = r_sig_clean; used_retrieval = True; break
                else:
                    if cand.strip() != gold.strip():
                        rejected = cand; persona_vl = r_sig_clean; used_retrieval = True; break

            # fallback: forced divergence
            if rejected is None:
                alt = force_divergence(gold)
                if alt.strip() != gold.strip() and _good(alt):
                    rejected = alt
                    persona_vl = "(fallback-perturbation)"
                    dbg["forced_divergence"] += 1

            if not rejected:  # give up this context
                continue

            # Final safety
            gold = _finalize_reply(gold)
            rejected = _finalize_reply(rejected)
            if not (_good(gold) and _good(rejected) and gold.strip() != rejected.strip()):
                continue

            # rDPO prompts
            prompt_vl = (f"{context}\n\n[PERSONA]\n{persona_vl}\n[BRES]"
                         if persona_vl and persona_vl != "(fallback-perturbation)"
                         else f"{context}\n\n[PERSONA]\n{persona_vl}")

            # Write
            f_rdpo.write(json.dumps({
                "Context": context,
                "Persona_v": persona_v_clean,
                "Persona_vl": persona_vl,
                "Prompt_v": prompt_v,
                "Prompt_vl": prompt_vl,
                "Gold": gold,
                "Rejected": rejected,
                "_meta": {"conv_id": cid, "used_retrieval": used_retrieval, "gold_vs_cand_sim": cos_val}
            }, ensure_ascii=False) + "\n")

            f_dpo.write(json.dumps({
                "Context": context,
                "Gold Response": gold,
                "Rejected Response": rejected
            }, ensure_ascii=False) + "\n")

            c_written += 1

    f_rdpo.close(); f_dpo.close()
    print(f"Done. Wrote {c_written} rDPO rows to {args.out_rdpo}")
    print(f"Also wrote {c_written} DPO rows to {args.out_dpo}")
    print("[debug]", dbg)

if __name__ == "__main__":
    main()
