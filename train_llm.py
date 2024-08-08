from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from rouge_score import rouge_scorer
import numpy as np
import torch

if not torch.cuda.is_available():
    raise EnvironmentError(
        "GPU is not available. Ensure you have a compatible GPU and the necessary drivers installed.")

# Load dataset
dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')

# Load language model for text generation
models = [
    # 8B
    "meta-llama/Meta-Llama-3-8B-Instruct",  # yes, Dauer: Oke, 0
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # yes, dauer: gut, 1

    # 7B
    "Qwen/Qwen2-7B-Instruct",  # yes, Dauer: Oke, 2
    "Qwen/Qwen-7B-Chat",  # yes, dauer: gut, 3
    "mistralai/Mistral-7B-Instruct-v0.3",  # yes, Dauer, gut, 4
    # "meta-llama/Llama-2-7b-chat-hf", # yes, Dauer: sehr lange
    # "internlm/internlm2_5-7b", # yes, Dauer: sehr lange
    # "tiiuae/falcon-7b-instruct",# yes, Dauer: sehr lange

    # 4B
    "microsoft/Phi-3-mini-128k-instruct",  # yes Dauer: Gut , 5
    "Qwen/Qwen1.5-4B-Chat",  # yes Dauer: Gut, 6

    # 0B - 2B,
    "internlm/internlm2-chat-1_8b",  # yes Dauer: Gut, 7
    "Qwen/Qwen2-1.5B-Instruct",  # yes Dauer: Gut, 8
    "Qwen/Qwen1.5-0.5B-Chat",  # yes Dauer: Gut, 9
    "Qwen/Qwen2-0.5B-Instruct",  # yes Dauer: Gut, 10

    # ?
    "Qwen/Qwen1.5-1.8B-Chat", # yes Dauer: Gut, 11
    "stabilityai/stablelm-2-1_6b-chat",# ? Dauer: ?, 12
    "HuggingFaceTB/SmolLM-1.7B-Instruct",# ? Dauer: ?, 13
    "microsoft/Phi-3-small-128k-instruct",# ? Dauer: ?, 14
]

model_index = 13  # 0 - 10
print(models[model_index])

# Load model and tokenizer
model_name = models[model_index]
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized


print(dataset["train"][:5])
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=['text'])

# Set training arguments
training_args = TrainingArguments(
    output_dir=f'./results/{models[model_index]}_finetuned',

    eval_strategy='epoch',

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=3, # More epochs can lead to better performance but may cause overfitting
    learning_rate=5e-5, # # Initial learning rate for the optimizer, Lower learning rates are more stable but training can be slower
    warmup_steps=500, # Warmup helps to stabilize training at the beginning
    weight_decay=0.01, # Helps to prevent overfitting by penalizing large weights

    logging_dir=f'./logs/{models[model_index]}_finetuned',
    logging_steps=100,

    save_strategy='epoch',
    save_total_limit=3,

    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,  # Higer rougeL is better

    fp16=True
)


# LoRA configuration
lora_config = LoraConfig(
    r=4,  # Rank of the low-rank matrices,  4, 8, 16 smaller => fever additional parameters
    lora_alpha=32,  # Scaling factor, 16, 32, 64 => efficiency vs stability
    lora_dropout=0.1,  # Dropout rate for LoRA layers, 0.0, 0.1, 0.2 => smaller dataset higher dropout to counter overfit
    target_modules=['q_proj', 'v_proj'],  # Target modules to apply LoRA
    # target_modules=['wqkv'],  # Target modules to apply LoRA

    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA/QLoRA to the model
model = get_peft_model(model, lora_config)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Train model
trainer.train()

# Save the trained model and tokenizer
model_save_path = f'./trained_model/{models[model_index]}_finetuned'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# nohup python train_llm.py &