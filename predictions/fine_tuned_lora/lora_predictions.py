#Produces base model predictions
import json, argparse, torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel     

parser = argparse.ArgumentParser()
parser.add_argument("--input_file",  required=True)
parser.add_argument("--output_file", required=True)
parser.add_argument("--checkpoint",  required=True,
                    help="Dir of best *full* checkpoint OR LoRA adapter")
parser.add_argument("--use_lora",    action="store_true",
                    help="Set if --checkpoint is adapter-only")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_name = "google/flan-t5-small"
tok   = AutoTokenizer.from_pretrained(base_model_name)

if args.use_lora:
    base  = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)
    model = PeftModel.from_pretrained(base, args.checkpoint).to(device)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)

model.eval()

ds = load_dataset("json", data_files={"data": args.input_file})["data"]

preds = []
for ex in tqdm(ds, desc="Generating"):
    prompt  = f"{ex['instruction']} {ex['input']}"
    inputs  = tok(prompt, return_tensors="pt",
                  truncation=True, padding=True).to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            num_beams=5,
            max_new_tokens=128,
            early_stopping=True
        )
    hyp = tok.decode(gen_ids[0], skip_special_tokens=True)
    preds.append({
        "input":      ex["input"],
        "reference":  ex.get("output", ""),
        "prediction": hyp
    })

with open(args.output_file, "w") as f:
    for r in preds:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f" wrote {len(preds)} lines → {args.output_file}")
