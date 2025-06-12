import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

txt_path = "lora_error_analysis.txt"

with open(txt_path, "r") as f:
    lines = f.readlines()

data_str = "".join(lines)
df = pd.read_fwf(StringIO(data_str))
df.columns = df.columns.str.strip()

df["TotalModelCorrections"] = df["TruePositive"] + df["FalsePositive"]

top_corrected = df.sort_values(by="TotalModelCorrections", ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.bar(top_corrected["EditType"], top_corrected["TruePositive"], label="Correct Edits (TP)", alpha=0.8)
plt.bar(top_corrected["EditType"], top_corrected["FalsePositive"], bottom=top_corrected["TruePositive"], label="Incorrect Edits (FP)", alpha=0.6)
#plt.xticks(rotation=45, ha='right')
plt.ylabel("Edit Count")
plt.title("Top 10 Grammar Edits Made by Model (Correct + Incorrect)")
plt.legend()
plt.tight_layout()

plt.savefig("model_most_common_corrections.png", dpi=300)
plt.show()
