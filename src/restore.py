import time
import json
import torch
import random
from tqdm import tqdm

def clean_text(text):
    """Clean whitespace and newlines in text"""
    return " ".join(text.strip().split())

def generate_prompt_qwen(instruction, input_text):
    input_text = clean_text(input_text)
    
    return f"""<|im_start|>system\n{instruction}<|im_end|>
<|im_start|>user\n{input_text}<|im_end|>
<|im_start|>assistant\n"""

def generate_prompt_vicuna(instruction, input_text):
    input_text = clean_text(input_text)
    
    return f"""SYSTEM: {instruction}\nUSER: {input}\nASSISTANT: """

def generate_prompt(instruction, compressed_text, model_name):
    if "Qwen" in model_name:
        return generate_prompt_qwen(instruction, compressed_text)
    elif "vicuna" in model_name:
        return generate_prompt_vicuna(instruction, compressed_text)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def check_message(model, tokenizer, data, test_setting):
    start_time = time.time()
    prompt = generate_prompt(test_setting["instruction"], data["compressed_text"], test_setting["model_name"])
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=test_setting["max_new_token"],
        truncation=True,
        padding=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=test_setting["max_new_token"],
            temperature=test_setting["temperature"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        
    # Decode and process output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Qwen" in test_setting["model_name"]:
        response_start = full_output.find("assistant\n") + len("assistant\n")
    elif "vicuna" in test_setting["model_name"]:
        response_start = full_output.find("ASSISTANT: ") + len("ASSISTANT: ")
    else:
        raise ValueError(f"Unsupported model name: {test_setting['model_name']}")
    
    restored_text = full_output[response_start:].strip()
    
    # Truncate at end marker
    end_pos = restored_text.find(test_setting["end_marker"])
    if end_pos != -1:
        restored_text = restored_text[:end_pos].strip()
        
    # TODO: Evaluation of word meanings can further enhance the load rate.
    original_words = data["original_text"].split()
    restored_words = restored_text.split()
    lack_original_words_with_index = [(index, word) for index, word in enumerate(original_words) if word.startswith("[") and word.endswith("]")]
    lack_restored_words_with_index = [(index, word) for index, word in enumerate(restored_words) if word.startswith("[") and word.endswith("]")]
    
    # Check if the restored text is the same as the original text
    if len(lack_original_words_with_index) != len(lack_restored_words_with_index):
        data["compressed_text"] = data["content"]
    else:
        compressed_words = data["compressed_text"].split()
        
        for i in range(len(lack_original_words_with_index)):
            if lack_original_words_with_index[i][1] != lack_restored_words_with_index[i][1]:
                compressed_words[lack_original_words_with_index[i][0]] = lack_original_words_with_index[i][1].replace("[", "").replace("]", "")
        
        data["compressed_text"] = " ".join(compressed_words)
    end_time = time.time()
    
    data["check_time"] = round(end_time - start_time, 6)
    
def restore_message(model, tokenizer, test_data, test_setting, output_file):
    sample_size = min(test_setting["test_size"], len(test_data))
    print(f"Restoring {sample_size} samples")
    sample_data = random.sample(test_data, sample_size)

    # Process samples
    for data in tqdm(sample_data, desc="Processing samples"):
        # Generate restored text
        start_time = time.time()
        prompt = generate_prompt(test_setting["instruction"], data["compressed_text"], test_setting["model_name"])
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=test_setting["max_new_token"],
            truncation=True,
            padding=True
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=test_setting["max_new_token"],
                temperature=test_setting["temperature"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode and process output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Qwen" in test_setting["model_name"]:
            response_start = full_output.find("assistant\n") + len("assistant\n")
        elif "vicuna" in test_setting["model_name"]:
            response_start = full_output.find("ASSISTANT: ") + len("ASSISTANT: ")
        else:
            raise ValueError(f"Unsupported model name: {test_setting['model_name']}")
        
        restored_text = full_output[response_start:].strip()
        
        # Truncate at end marker
        end_pos = restored_text.find(test_setting["end_marker"])
        if end_pos != -1:
            restored_text = restored_text[:end_pos].strip()
        end_time = time.time()
        
        data["restored_text"] = restored_text
        data["restore_time_cost"] = round(end_time - start_time, 6)
    
    print(f"=================== Restore Data Complete ===================")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=4, ensure_ascii=False)
    
    return sample_data