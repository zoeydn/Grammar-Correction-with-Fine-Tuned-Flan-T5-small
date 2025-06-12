import os
import matplotlib.pyplot as plt
import pandas as pd

gleu_files = {
    "base": "gleu_result_base.txt",
    "full": "gleu_result_full.txt",
    "adapter": "gleu_result_adapter.txt",
    "checkpoint_11000": "gleu_result_lora_11000.txt"
}

gleu_scores = {}

for model, filename in gleu_files.items():
    print(f"Reading {filename}...")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            line = f.readline().strip()
            gleu = float(line.split()[-1])
            gleu_scores[model] = gleu
            print(f"Parsed GLEU for {model}: {gleu}")
        

valid_scores = {m: s for m, s in gleu_scores.items() if s is not None}

if valid_scores:
    models = list(valid_scores.keys())
    scores = list(valid_scores.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, scores, color='mediumseagreen', edgecolor='black')

    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{score:.3f}", ha='center', va='bottom')

    plt.title("Comparison of GLEU Scores Across Models", fontsize=14)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("GLEU Score", fontsize=12)
    plt.ylim(0, max(scores) + 0.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig("gleu_scores_bargraph.png", dpi=300)
    plt.show()
else:
    print("No valid GLEU scores found.")

# Show table
df = pd.DataFrame.from_dict(gleu_scores, orient='index', columns=['GLEU Score'])
df.index.name = 'Model'
print("\nExtracted GLEU Scores:")
print(df)
