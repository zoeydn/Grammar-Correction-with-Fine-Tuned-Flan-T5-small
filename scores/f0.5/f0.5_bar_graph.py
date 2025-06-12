import os
import matplotlib.pyplot as plt
import pandas as pd

score_files = {
    "baseline": "score_baseline.txt",
    "full": "score_full.txt",
    "lora_adapter": "score_lora_adapter.txt",
    "lora_checkpoint_11000": "score_lora_11000.txt"
}

f05_scores = {}

for model, filename in score_files.items():
    with open(filename, 'r') as f:
        line4 = f.readlines()[3].strip()  
        f05 = float(line4.split()[-1])  
        f05_scores[model] = f05
        print(f"{model}: {f05}")

valid_scores = {m: s for m, s in f05_scores.items() if s is not None}

if valid_scores:
    models = list(valid_scores.keys())
    scores = list(valid_scores.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, scores, color='skyblue', edgecolor='black')

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.3f}", 
                 ha='center', va='bottom', fontsize=10)

    plt.title("F0.5 Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("F0.5 Score")
    plt.ylim(0, max(scores) + 0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("f05_scores_bargraph.png", dpi=300)
    plt.show()
else:
    print("No valid F0.5 scores found.")

df = pd.DataFrame.from_dict(f05_scores, orient='index', columns=['F0.5 Score'])
df.index.name = 'Model'
print("\nExtracted F0.5 Scores:")
print(df)
