#Produces base model predictions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import json
import torch
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", required=True, help="Path to dev JSONL file")
parser.add_argument("--output_file", required=True, help="Path to save predictions")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

dataset = load_dataset("json", data_files={"dev": args.input_file})["dev"]

predictions = []
for example in tqdm(dataset, desc="Generating corrections"):
    prompt = f"{example['instruction']} {example['input']}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=5,
        early_stopping=True
    )
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append({
        "input": example["input"],
        "reference": example["output"],
        "prediction": pred
    })

with open(args.output_file, "w") as f:
    for entry in predictions:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
