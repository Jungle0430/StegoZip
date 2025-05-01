import os
import torch
import wandb
import numpy as np
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def clean_text(text):
    """Clean whitespace and newlines in text"""
    return " ".join(text.strip().split())

def generate_prompt_qwen(instruction, input_text, output_text, end_marker="[END]"):
    input_text = clean_text(input_text)
    output_text = clean_text(output_text)
    
    return f"""<|im_start|>system\n{instruction}<|im_end|>
<|im_start|>user\n{input_text}<|im_end|>
<|im_start|>assistant\n{output_text}{end_marker}<|im_end|>"""

def generate_prompt_deepseek(instruction, input_text, output_text, end_marker="[END]"):
    input_text = clean_text(input_text)
    output_text = clean_text(output_text)
    
    return f"""System:{instruction}\n\nUser:{input_text}\n\nAssistant:{output_text}{end_marker}"""

def tokenize(prompt, tokenizer, cutoff_len=512, response_start_pos=0, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
        add_special_tokens=True
    )
    
    if add_eos_token and result["input_ids"][-1] != tokenizer.eos_token_id:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    
    import copy    
    labels = copy.deepcopy(result["input_ids"])
    labels[:response_start_pos] = [-100] * response_start_pos
    
    return {
        "input_ids": result["input_ids"][:cutoff_len],
        "attention_mask": result["attention_mask"][:cutoff_len],
        "labels": labels[:cutoff_len]
    }

def generate_and_tokenize_prompt(tokenizer, data_point, train_settings):
    if "Qwen" in train_settings['model_name']:
        full_prompt = generate_prompt_qwen(train_settings['instruction'],
                                           data_point["compressed_text"],
                                           data_point["original_text"],
                                           train_settings['end_marker'])
        prompt_without_response = full_prompt.split("assistant\n")[0] + "assistant\n"
        tokenized_prompt = tokenizer(prompt_without_response, add_special_tokens=False)
        response_start_pos = len(tokenized_prompt["input_ids"])
        
    elif "deepseek" in train_settings['model_name']:
        full_prompt = generate_prompt_deepseek(train_settings['instruction'],
                                               data_point["compressed_text"],
                                               data_point["original_text"],
                                               train_settings['end_marker'])
        prompt_without_response = full_prompt.split("Assistant:")[0] + "Assistant:"
        tokenized_prompt = tokenizer(prompt_without_response, add_special_tokens=False)
        response_start_pos = len(tokenized_prompt["input_ids"])
        
    else:
        raise ValueError(f"Unsupported model name: {train_settings['model_name']}")
    
    tokenized_full = tokenize(full_prompt, tokenizer, train_settings['cutoff_len'], response_start_pos)
    
    return tokenized_full

def finetune_model(base_model,
                   tokenizer,
                   train_data,
                   output_dir,
                   train_settings,
                   use_wandb=True):
    if os.path.exists(output_dir):
        raise FileExistsError(f"Output directory {output_dir} already exists! \nIf you want to refinetune the model, please delete the directory first!")
    
    if use_wandb:
        wandb.init(project="stegozip", name=f"{train_settings['model_name']}_finetune", config=train_settings)
    
    model = prepare_model_for_kbit_training(base_model)
    """Fine-tune a language model on text restoration task"""
    lora_config = LoraConfig(
        r=train_settings['lora_r'],
        lora_alpha=train_settings['lora_alpha'],
        target_modules=train_settings['target_modules'],
        lora_dropout=train_settings['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    wandb.watch(
        model, 
        log="gradients",
        log_freq=10,
        log_graph=True
    )
    
    if train_settings['val_ratio'] > 0:
        train_val = train_data.train_test_split(
            test_size=train_settings['val_ratio'],
            shuffle=True
        )
        train_data = train_val["train"].shuffle().map(
            lambda x: generate_and_tokenize_prompt(tokenizer, x,  train_settings)
        )
        val_data = train_val["test"].shuffle().map(
            lambda x: generate_and_tokenize_prompt(tokenizer, x, train_settings)
        )
        print(f"Training set size: {len(train_data)}, Validation set size: {len(val_data)}")
    else:
        val_data = None
        train_data = train_data.shuffle().map(
            lambda x: generate_and_tokenize_prompt(tokenizer, x, train_settings)
        )
        print(f"Training set size: {len(train_data)}, Validation set size: 0")
        
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=train_settings['micro_batch_size'],
        per_device_eval_batch_size=train_settings['micro_batch_size'],
        gradient_accumulation_steps=train_settings['gradient_accumulation_steps'],
        warmup_ratio=train_settings['warmup_ratio'],
        num_train_epochs=train_settings['epochs'],
        learning_rate=train_settings['learning_rate'],
        # max_grad_norm=1.0,
        # fp16=True,
        logging_steps=train_settings['logging_steps'],
        evaluation_strategy="steps" if train_settings['val_ratio'] > 0 else "no",
        save_strategy="steps",
        eval_steps=train_settings['eval_steps'],
        save_steps=train_settings['save_steps'],
        output_dir=output_dir,
        save_total_limit=train_settings['save_total_limit'],
        load_best_model_at_end=True if  train_settings['val_ratio'] > 0 else False,
        ddp_find_unused_parameters=False,
        report_to="wandb",
        run_name=f"{train_settings['model_name']}_finetune",
        optim="adamw_torch",
    )
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    model.config.use_cache = False
    print("Starting training...")
    with torch.autograd.detect_anomaly():
        trainer.train()
    model.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    file_path = "../data/compress/Qwen_ag_news_train_3_10.json"
    instruction = "You are a text restoration expert."
    
    with open(file_path, "r") as f:
        test_dataset = json.load(f)
        
    for data_point in test_dataset:
        text = generate_prompt_qwen(instruction,
                                    data_point["compressed_text"],
                                    data_point["original_text"])
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokens = tokenize(text, tokenizer, 256)
        
        print(f"text")
        print(f"\n{tokens}\n")
        