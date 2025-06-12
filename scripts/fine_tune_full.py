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

model_name = "google/flan-t5-small"
train_path = "/gscratch/scrubbed/dinuoz/fine_tune_grammar/wi+locness/m2/ABC.train.gold.bea19.jsonl" 
dev_path = "/gscratch/scrubbed/dinuoz/fine_tune_grammar/wi+locness/m2/new_dev.jsonl"
output_dir = "./flan_t5_grammar_full"
max_input_length = 256
max_target_length = 256
batch_size = 4
num_train_epochs = 3
learning_rate = 5e-4

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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

trainer.save_model(os.path.join(output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))