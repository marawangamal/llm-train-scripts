"""
Fine-tune GPT-2 on ELI5 dataset.

Resources: https://huggingface.co/docs/transformers/tasks/language_modeling


Two options for data loading:

Given a dataset of sequences of different length {s1, s2, ..., s2}, we have two options for dataloading

1. Simple (preprocess_simple)
    - Convert each sequence to be of length `max_len` via padding or trunction 

2. Advanced (preprocess_function & group texts)
    - Combine to sinlge length string s = [s_1, s_2, ..., s_b], then split into chunks of size `max_len`. This is less 
    - Less wastefulness from truncation


"""

import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
)
from transformers import DataCollatorForLanguageModeling, get_scheduler


def preprocess_eli5(examples):
    return {"text": [" ".join(x) for x in examples["answers.text"]]}


def tokenize(examples, tokenizer):
    return tokenizer(examples["text"])


def preprocess_function_simple(examples, tokenizer, block_size):
    return tokenizer(
        [" ".join(x) for x in examples["answers.text"]],
        padding="max_length",
        truncation=True,
        max_length=block_size,
    )


def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_eli5_data(tokenizer, block_size):
    eli5 = load_dataset("eli5_category", split="train[:5000]")
    eli5 = eli5.train_test_split(test_size=0.2)  # type: ignore
    eli5 = eli5.flatten()
    eli5_processed = eli5.map(
        preprocess_eli5,
        batched=True,
    )
    eli5_tokenized = eli5_processed.map(
        lambda examples: tokenize(examples, tokenizer),
        batched=True,
        num_proc=4,
        remove_columns=[*eli5["train"].column_names, "text"],
    )
    # Each sample is now of length `block_size`
    eli5_regrouped = eli5_tokenized.map(
        lambda examples: group_texts(examples, block_size),
        batched=True,
        num_proc=4,
    )
    return eli5_regrouped


def get_test_sample(
    model, tokenizer, prompt="Somatic hypermutation allows the immune system to"
):
    # Inference
    cpu_device = "cpu"
    model.to(cpu_device)  # type: ignore
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Train function
def train(model, train_dataloader, eval_dataloader, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # type: ignore
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        losses = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(loss.item())

        eval_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch + 1}, Perplexity: {math.exp(eval_loss):.2f}")


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    block_size = 128

    lm_dataset = load_eli5_data(tokenizer, block_size)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Data loaders
    train_dataloader = DataLoader(
        lm_dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator  # type: ignore
    )
    eval_dataloader = DataLoader(
        lm_dataset["test"], batch_size=8, collate_fn=data_collator  # type: ignore
    )

    # (Option A) Train from scratch
    # # Load configuration from pretrained model and create a new model from scratch
    # config = GPT2Config.from_pretrained("distilbert/distilgpt2")
    # model = GPT2LMHeadModel(config)

    # (Option B) Pretrained
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

    train(model, train_dataloader, eval_dataloader)

    # Generate a test sample
    sample_output = get_test_sample(model, tokenizer)
    print(sample_output)
