# build_act_data_from_pact.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, random, csv
from collections import defaultdict

"""
Input: PACT CSV with columns:
conv_id, Preference Profile, Buyer Profile, Buyer Argument Profile, Seller Argument Profile,
Negotiation Strategy, Speaker, Intent, Utterance, Stage, Aspect, Offer, Removed Amenity,
Removed Cost, Added Amenity, Added Cost, Argument, persona_evidences, persona_triplets

Notes:
- Speaker: "1" = Agent, "0" = Traveler (based on your example). Adjust with --agent_value if needed.
- We create a simple act classification dataset:
    text = agent utterance (Utterance)
    label = Intent (dialogue act)
- If a conversation begins, persona_* columns are usually only present in the *first* row for that conv.
  We optionally emit a sidecar JSON with the persona sentences per conv (can be useful later).
"""

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pact_csv", required=True)
    ap.add_argument("--out_dir", default="pact_act_data")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--agent_value", type=str, default="1", help="Value of Speaker that denotes Agent")
    return ap.parse_args()

def clean_text(s):
    s = (s or "").strip()
    # remove leading/trailing quotes if the CSV has doubled quotes
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1]
    return s

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with open(args.pact_csv, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)

    # Build act examples from agent utterances
    examples = []
    conv_persona = {}  # conv_id -> list of persona triplets as strings
    for r in rows:
        conv_id = str(r.get("conv_id", "")).strip()
        spk      = str(r.get("Speaker", "")).strip()
        intent   = clean_text(r.get("Intent", ""))
        utt      = clean_text(r.get("Utterance", ""))

        # persona_triplets usually only in first row per conversation
        if conv_id and conv_id not in conv_persona:
            trip_str = r.get("persona_triplets", "")
            if trip_str:
                # expect something like: "['(Traveler, seeks, local shopping...)', ...]"
                try:
                    trip_list = eval(trip_str)  # trusted source; otherwise JSON-sanitize first
                    conv_persona[conv_id] = [t.strip() for t in trip_list if isinstance(t, str)]
                except Exception:
                    conv_persona[conv_id] = []

        if spk == args.agent_value and utt and intent:
            examples.append({"text": utt, "label": intent, "conv_id": conv_id})

    random.seed(args.seed)
    random.shuffle(examples)

    n = len(examples)
    n_val = max(1, int(n * args.val_ratio))
    val, train = examples[:n_val], examples[n_val:]

    # Save JSONL
    def dump(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    dump(os.path.join(args.out_dir, "train.jsonl"), train)
    dump(os.path.join(args.out_dir, "valid.jsonl"), val)

    # Sidecar persona by conversation (optional/debug)
    with open(os.path.join(args.out_dir, "persona_by_conv.json"), "w", encoding="utf-8") as f:
        json.dump(conv_persona, f, ensure_ascii=False, indent=2)

    print(f"✅ Built act dataset in {args.out_dir}")
    print(f"   train: {len(train)} | valid: {len(val)} | convs with persona: {len(conv_persona)}")

if __name__ == "__main__":
    main()
