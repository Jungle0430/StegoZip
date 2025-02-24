import os
import torch
import numpy as np
import transformers
from peft import LoraConfig, get_peft_model

def generate_prompt_qwen(instruction, input, output, end_marker="[END]"):
    return f"""### Instruction: {instruction} \n### Input: {input} \n### Response: {output}{end_marker}"""

def generate_prompt_vicuna(instruction, input, output, end_marker="[END]"):
    return f"""SYSTEM: {instruction} \nUSER: {input} \nASSISTANT: {output}{end_marker}"""

def clean_text(text):
    """Clean whitespace and newlines in text"""
    return " ".join(text.strip().split())

def tokenize(prompt, tokenizer, cutoff_len=512, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
    )
    
    if add_eos_token and result["input_ids"][-1] != tokenizer.eos_token_id:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    
    return {
        "input_ids": result["input_ids"][:cutoff_len],
        "attention_mask": result["attention_mask"][:cutoff_len],
    }

def generate_and_tokenize_prompt(tokenizer, data_point, train_settings):
    if "Qwen" in train_settings['model_name']:
        full_prompt = generate_prompt_qwen(train_settings['instruction'],
                                           data_point["compressed_text"],
                                           data_point["original_text"],
                                           train_settings['end_marker'])
    elif "vicuna" in train_settings['model_name']:
        full_prompt = generate_prompt_vicuna(train_settings['instruction'],
                                              data_point["compressed_text"],
                                              data_point["original_text"],
                                              train_settings['end_marker'])
    else:
        raise ValueError(f"Unsupported model name: {train_settings['model_name']}")
    
    tokenized = tokenize(full_prompt, tokenizer, train_settings['cutoff_len'])
    return tokenized

def finetune_model(base_model,
                   tokenizer,
                   train_data,
                   output_dir,
                   train_settings):
    if os.path.exists(output_dir):
        raise FileExistsError(f"Output directory {output_dir} already exists!")
    
    """Fine-tune a language model on text restoration task"""
    lora_config = LoraConfig(
    r=train_settings['lora_r'],
    lora_alpha=train_settings['lora_alpha'],
    target_modules=train_settings['target_modules'],
    lora_dropout=train_settings['lora_dropout'],
    bias="none",
    task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
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
    fp16=True,
    logging_steps=train_settings['logging_steps'],
    evaluation_strategy="steps" if train_settings['val_ratio'] > 0 else "no",
    save_strategy="steps",
    eval_steps=train_settings['eval_steps'],
    save_steps=train_settings['save_steps'],
    output_dir=output_dir,
    save_total_limit=train_settings['save_total_limit'],
    load_best_model_at_end=True if  train_settings['val_ratio'] > 0 else False,
    ddp_find_unused_parameters=False,
    report_to="none",
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
    with torch.enable_grad():
        trainer.train()
    model.save_pretrained(output_dir)
    print("Training complete!")
    

    
    
    
    

