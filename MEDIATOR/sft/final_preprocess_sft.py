#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, ast, argparse
import pandas as pd
from typing import Any, Dict, List, Optional

SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[BPER]", "[EPER]", "[BACT]", "[EACT]", "[BRES]", "[ERES]"]
NEUTRAL_AGENT_ACT = "general_response"

def safe_parse_list(x):
    if pd.isna(x): return []
    if isinstance(x, list): return x
    try:
        val = ast.literal_eval(x)
        return val if isinstance(val, list) else []
    except Exception:
        return []

def norm_act(x: Optional[str]) -> str:
    x = (x or "").strip()
    return x if x else NEUTRAL_AGENT_ACT

def turn_str(act, text):
    return f"[BACT] {norm_act(act)} [EACT] {str(text).strip()}"

def build_samples_for_conv(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build (X, Y) pairs where Y = [BPER] p... [EPER] [BACT] a_t [EACT] [BRES] A_t [ERES]."""
    if not records: return []
    first = records[0]
    persona_triplets = safe_parse_list(first.get("persona_triplets", []))
    per_block = " ".join(persona_triplets) if persona_triplets else ""

    # Typed, acted turns
    turns = []
    for r in records:
        role = "agent" if int(r["Speaker"]) == 1 else "traveler"
        turns.append({
            "role": role,
            "text": str(r["Utterance"]).strip(),
            "act": str(r.get("Intent","")).strip()
        })

    samples, H_tagged = [], []

    for i, t in enumerate(turns):
        # append the current turn (tagged) to history
        H_tagged.append(turn_str(t["act"], t["text"]))

        # create a sample when current is traveler and next is agent
        if t["role"] == "traveler" and i + 1 < len(turns) and turns[i+1]["role"] == "agent":
            H = "\n".join(H_tagged[:-1])  # up to U_t-1
            U_t = H_tagged[-1]            # current traveler turn
            INST = (
                "You are a helpful travel booking assistant. Given the dialogue history between an agent and a traveler, "
                "and the traveler's current turn, produce: (i) the traveler's relevant persona triplets, (ii) the agent's dialogue act, "
                "and (iii) the agent's next response. Use the special tokens exactly as specified."
            )
            X = f"{INST}\n\n{H}\n{U_t}" if H else f"{INST}\n\n{U_t}"

            next_agent = turns[i+1]
            a_t = norm_act(next_agent["act"])
            A_t = next_agent["text"]

            Y = (
                "[BOS] [BPER] " + (per_block + " " if per_block else "") +
                "[EPER] [BACT] " + a_t + " [EACT] [BRES] " + A_t + " [ERES] [EOS]"
            )
            samples.append({"input": X.strip(), "output": Y.strip()})
    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", required=True, help="Path to original dataset CSV")
    ap.add_argument("--out_dir", required=True, help="Output folder for the 10-conv splits")
    ap.add_argument("--num_convs", type=int, default=10, help="How many conv_ids to include")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read CSV (use explicit dtype for persona cols)
    df = pd.read_csv(
        args.raw_csv,
        low_memory=False,
        dtype={
            "persona_evidences": "string",
            "persona_triplets": "string"
        }
    )

    req = {
        "conv_id","Preference Profile","Buyer Profile","Buyer Argument Profile","Seller Argument Profile",
        "Negotiation Strategy","Speaker","Intent","Utterance","Stage","Aspect","Offer","Removed Amenity",
        "Removed Cost","Added Amenity","Added Cost","Argument","persona_evidences","persona_triplets"
    }
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Choose conv_ids that actually have persona_triplets on the first row
    conv_ids = []
    for cid, sub in df.groupby("conv_id", sort=True):
        first = sub.sort_index().iloc[0]
        pts = safe_parse_list(first.get("persona_triplets", None))
        if pts:
            conv_ids.append(cid)
    conv_ids = conv_ids[:args.num_convs]
    if not conv_ids:
        raise ValueError("No conversations with persona_triplets found!")

    # Build samples for each selected conv_id
    all_samples = []
    by_conv = {}
    for cid in conv_ids:
        recs = df[df["conv_id"] == cid].sort_index().to_dict("records")
        samples = build_samples_for_conv(recs)
        by_conv[cid] = samples
        all_samples.extend(samples)

    # Split conv-wise: 8/1/1 (train/val/test)
    train_c = conv_ids[:2171]
    val_c   = conv_ids[2171:3256]
    test_c  = conv_ids[3256:4343]

    def collect(cids):
        out = []
        for cid in cids:
            out.extend(by_conv[cid])
        return out

    train_samples = collect(train_c)
    val_samples   = collect(val_c)
    test_samples  = collect(test_c)

    # Save CSVs
    pd.DataFrame(train_samples).to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    pd.DataFrame(val_samples).to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    pd.DataFrame(test_samples).to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    # Quick stats
    print("✅ Prepared 10-conv subset with tagged history and correct targets")
    print(f"  conv_ids: {conv_ids}")
    print(f"  train: {len(train_samples)} samples from {len(train_c)} convs")
    print(f"  val:   {len(val_samples)} samples from {len(val_c)} convs")
    print(f"  test:  {len(test_samples)} samples from {len(test_c)} convs")

if __name__ == "__main__":
    main()
