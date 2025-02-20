import time
import json
import torch
import random
from tqdm import tqdm

def generate_prompt_qwen(instruction, input):
    return f"""### Instruction: {instruction} \n### Input: {input} \n### Response: """

def generate_prompt_vicuna(instruction, input):
    return f"""SYSTEM: {instruction}\nUSER: {input}\nASSISTANT: """

def generate_prompt(instruction, compressed_text, model_name):
    if "Qwen" in model_name:
        return generate_prompt_qwen(instruction, compressed_text)
    elif "vicuna" in model_name:
        return generate_prompt_vicuna(instruction, compressed_text)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def restore_message(model, tokenizer, test_data, test_setting, output_file):
    sample_size = min(test_setting["test_size"], len(test_data))
    print(f"Restoring {sample_size} samples")
    sampled_data = random.sample(test_data, sample_size)

    # Process samples
    for data in tqdm(sampled_data, desc="Processing samples"):
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
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and process output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = time.time()
        
        if "Qwen" in test_setting["model_name"]:
            response_start = full_output.find("### Response: ") + len("### Response: ")
        elif "vicuna" in test_setting["model_name"]:
            response_start = full_output.find("ASSISTANT: ") + len("ASSISTANT: ")
        else:
            raise ValueError(f"Unsupported model name: {test_setting['model_name']}")
        
        restored_text = full_output[response_start:].strip()
        
        # Truncate at end marker
        end_pos = restored_text.find(test_setting["end_marker"])
        if end_pos != -1:
            restored_text = restored_text[:end_pos].strip()
        
        data["restored_text"] = restored_text
        data["restore_time_cost"] = round(end_time - start_time, 6)
    
    print(f"=================== Restore Data Complete ===================")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=4, ensure_ascii=False)