import re
from collections import Counter
import pandas as pd
import os

def parse_m2(filepath):
    edits = []
    with open(filepath, 'r') as f:
        sent_id = 0
        for line in f:
            line = line.strip()
            if line.startswith("A "):
                parts = line.split("|||")
                span = parts[0].split()
                start, end = int(span[1]), int(span[2])
                edit_type = parts[1]
                cor = parts[2]
                edits.append((sent_id, start, end, edit_type, cor))
            elif line == "":
                sent_id += 1
    return edits

# === File paths (replace if needed) ===
pred_m2_path = "/Predictions/fine_tuned_full/pred.m2"
ref_m2_path = "/Predictions/fine_tuned_full/ref.m2"

# === Main logic ===
if os.path.exists(pred_m2_path) and os.path.exists(ref_m2_path):
    pred_edits = parse_m2(pred_m2_path)
    ref_edits = parse_m2(ref_m2_path)

    pred_set = set((sid, s, e, t) for sid, s, e, t, _ in pred_edits)
    ref_set = set((sid, s, e, t) for sid, s, e, t, _ in ref_edits)

    tp_edits = pred_set & ref_set
    fp_edits = pred_set - ref_set
    fn_edits = ref_set - pred_set

    def count_types(edit_set):
        counts = Counter()
        for _, _, _, etype in edit_set:
            counts[etype] += 1
        return counts

    tp_counts = count_types(tp_edits)
    fp_counts = count_types(fp_edits)
    fn_counts = count_types(fn_edits)

    all_types = sorted(set(tp_counts) | set(fp_counts) | set(fn_counts))
    df = pd.DataFrame({
        "EditType": all_types,
        "TruePositive": [tp_counts.get(t, 0) for t in all_types],
        "FalsePositive": [fp_counts.get(t, 0) for t in all_types],
        "FalseNegative": [fn_counts.get(t, 0) for t in all_types],
    })

    df["TotalErrors"] = df["FalsePositive"] + df["FalseNegative"]
    df = df.sort_values(by="TotalErrors", ascending=False)

    # === Save pretty output only ===
    with open("error_analysis.txt", "w") as f:
        f.write(df.to_string(index=False))

else:
    print("❌ ERROR: One or both input .m2 files are missing.")
    print(f"  pred.m2: {pred_m2_path} exists? {os.path.exists(pred_m2_path)}")
    print(f"  ref.m2:  {ref_m2_path} exists? {os.path.exists(ref_m2_path)}")
