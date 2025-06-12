import os
import json
from datasets import Dataset
from tqdm import tqdm
from transformers import TrainerCallback
from transformers import EarlyStoppingCallback

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

class TqdmCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps
        self.pbar = tqdm(total=total_steps, desc="Training Progress", dynamic_ncols=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pbar:
            self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", required=True, help="Path to training JSONL file")
parser.add_argument("--dev_path", required=True, help="Path to validation JSONL file")
parser.add_argument("--output_dir", required=True, help="Directory to save LoRA adapter and checkpoints")
args = parser.parse_args()

model_name = "google/flan-t5-small"
max_input_length = 256
max_target_length = 256
batch_size = 4
num_train_epochs = 3
learning_rate = 5e-4

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(base_model, lora_config)
print(model.print_trainable_parameters())

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

train_raw = load_jsonl(train_path)
dev_raw = load_jsonl(dev_path)

train_dataset = Dataset.from_list(train_raw)
eval_dataset = Dataset.from_list(dev_raw)

def preprocess(example):
    prompt   = example["instruction"] + " " + example["input"]
    response = example["output"]

    model_inputs = tokenizer(
        prompt,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    
    labels = tokenizer(
        text_target=response,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )["input_ids"]

    first_non_pad = next((i for i, tid in enumerate(labels) if tid != tokenizer.pad_token_id), None)
    for i, tid in enumerate(labels):
        if i > first_non_pad and tid == tokenizer.pad_token_id:
            labels[i] = -100

    model_inputs["labels"] = labels
    return model_inputs


tokenized_train = train_dataset.map(
    preprocess, remove_columns=train_dataset.column_names)
tokenized_eval  = eval_dataset.map(
    preprocess, remove_columns=eval_dataset.column_names)

ex = tokenized_train[0]

wanted_keys = {"input_ids", "attention_mask", "labels"}
batch = {k: torch.tensor([ex[k]]).to(model.device) for k in wanted_keys}

print("Non-masked label count:", sum(t != -100 for t in ex["labels"]))
print("Surviving label IDs:", set(ex["labels"]) - {-100})

with torch.no_grad():
    print("Single-batch loss:", model(**batch).loss.item())


training_args = Seq2SeqTrainingArguments(
    logging_first_step=True,
    output_dir=output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-4,
    num_train_epochs=3,
    warmup_steps=10,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=False,
    report_to="none",
    remove_unused_columns=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(output_dir, "lora_adapter"))
