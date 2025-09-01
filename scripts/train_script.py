import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from scripts.test_model import model, tokenizer


dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    report_to="wandb",     # log to W&B
    run_name="llama-wikitext2-ft",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_steps=100,
    fp16=True,   # or bf16 if GPU supports
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=collator,
    tokenizer=tokenizer,
)

if __name__ = "__main__":
    
    wandb.login()
    wandb.init(project="llama-wikitext2", name="llama-ft-run")
    
    trainer.train()

    wandb.finish()