import time
import json
import torch
import random
from tqdm import tqdm
from transformers.generation import StoppingCriteria, StoppingCriteriaList

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self,
                 end_marker_ids):
        self.end_marker_ids = end_marker_ids
        self.end_len = len(self.end_marker_ids)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq_length = input_ids.shape[-1]
        if seq_length >= self.end_len:
            last_tokens = input_ids[0, -self.end_len:].tolist()
            if last_tokens == self.end_marker_ids:
                return True
        
        return False

def clean_text(text):
    """Clean whitespace and newlines in text"""
    return " ".join(text.strip().split())

def generate_prompt_qwen(instruction, input_text):
    input_text = clean_text(input_text)
    
    return f"""<|im_start|>system\n{instruction}<|im_end|>
<|im_start|>user\n{input_text}<|im_end|>
<|im_start|>assistant\n"""

def generate_prompt_deepseek(instruction, input_text):
    input_text = clean_text(input_text)
    return f"""System:{instruction}\n\nUser:{input_text}\n\nAssistant:"""

def generate_prompt(instruction, compressed_text, model_name):
    if "Qwen" in model_name:
        return generate_prompt_qwen(instruction, compressed_text)
    elif "deepseek" in model_name:
        return generate_prompt_deepseek(instruction, compressed_text)
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
    
    eos_token_id = tokenizer.encode(test_setting["end_marker"], add_special_tokens=False)[-2:]
    stopping_criteria = CustomStoppingCriteria(eos_token_id)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=test_setting["max_new_token"],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([stopping_criteria])
        )
        
    # Decode and process output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Qwen" in test_setting["model_name"]:
        response_start = full_output.find("assistant\n") + len("assistant\n")
    elif "deepseek" in test_setting["model_name"]:
        response_start = full_output.find("Assistant:") + len("Assistant:")
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
                compressed_words[lack_original_words_with_index[i][0]] = lack_original_words_with_index[i][1]
        
        data["compressed_text"] = " ".join(compressed_words)
    end_time = time.time()
    
    data["check_time"] = round(end_time - start_time, 6)

def before_check(compressed_text):
    words = compressed_text.split()
    record_list = []
    result_words = []

    for index, word in enumerate(words, start=1):
        if word.startswith("[") and word.endswith("]"):
            content = word[1:-1]
            if content:
                record_list.append((index, content))
                result_words.append("[]")
            else:
                result_words.append(word)
        else:
            result_words.append(word)

    result_sentence = " ".join(result_words)
    return record_list, result_sentence

def after_restore(record_list, result_sentence, compressed_sentence):
    result_words = result_sentence.split()
    compressed_words = compressed_sentence.split()
    lack_result_words_with_index = [(index, word) for index, word in enumerate(result_words) if word.startswith("[") and word.endswith("]")]
    lack_compressed_words_with_index = [(index, word) for index, word in enumerate(compressed_words) if word.startswith("[") and word.endswith("]")]
    
    if len(lack_result_words_with_index) != len(lack_compressed_words_with_index):
        return " ".join(compressed_words)
        
    for i in range(len(lack_result_words_with_index)):
        compressed_words[lack_compressed_words_with_index[i][0]] = lack_result_words_with_index[i][1]
    
    if record_list == []:
        return " ".join(compressed_words)
    
    for position, content in record_list:
        compressed_words[position - 1] = f"[{content}]"
    restored_sentence = " ".join(compressed_words)
    return restored_sentence

def restore_message(model, tokenizer, test_data, test_setting, output_file):
    sample_size = min(test_setting["test_size"], len(test_data))
    print(f"Restoring {sample_size} samples")
    sample_data = random.sample(test_data, sample_size)

    # Process samples
    for data in tqdm(sample_data, desc="Processing samples"):
        # Generate restored text
        start_time = time.time()
        record_list, result_sentence = before_check(data["compressed_text"])
        prompt = generate_prompt(test_setting["instruction"], result_sentence, test_setting["model_name"])
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=test_setting["max_new_token"],
            truncation=True,
            padding=True
        ).to(model.device)
        
        eos_token_id = tokenizer.encode(test_setting["end_marker"], add_special_tokens=False)[-2:]
        stopping_criteria = CustomStoppingCriteria(eos_token_id)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=test_setting["max_new_token"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([stopping_criteria])
            )
        
        # Decode and process output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Qwen" in test_setting["model_name"]:
            response_start = full_output.find("assistant\n") + len("assistant\n")
        elif "deepseek" in test_setting["model_name"]:
            response_start = full_output.find("Assistant:") + len("Assistant:")
        else:
            raise ValueError(f"Unsupported model name: {test_setting['model_name']}")
        
        restored_text = full_output[response_start:].strip()
        
        # Truncate at end marker
        end_pos = restored_text.find(test_setting["end_marker"])
        if end_pos != -1:
            restored_text = restored_text[:end_pos].strip()
        
        restored_text = after_restore(record_list, restored_text, data["compressed_text"])
        end_time = time.time()
        
        data["restored_text"] = restored_text
        data["restore_time_cost"] = round(end_time - start_time, 6)
    
    print(f"=================== Restore Data Complete ===================")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=4, ensure_ascii=False)
    
    return sample_data
    
if __name__ == "__main__":
    import time
    import torch
    import random
    import numpy as np
    from peft import PeftModel
    
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    setup_seed(42)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # lora_path = "../checkpoint/DeepSeek-R1-Distill-Llama-8B_ag_news_business_30"
    model_name = "Qwen/Qwen2.5-7B"
    lora_path = "../checkpoint/Qwen2.5-7B_ag_news_business_30"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload().eval()
    
    data = {"content": "VoIP becomes new option Jackson is one of the first to take advantage of Time Warner Cable #39;s new venture into Voice over Internet Provider (VoIP) telephone service here and she says it works great.",
            "original_text": "VoIP becomes [new] [option] Jackson is one of [the] first to [take] advantage of [Time] Warner Cable #39;s [new] [venture] into Voice over Internet Provider (VoIP) telephone [service] here [and] she says [it] works great.",
            "compressed_text": "VoIP becomes [] [] Jackson is one of [] first to [] advantage of [] Warner Cable #39;s [] [] into Voice over Internet Provider (VoIP) telephone [] here [] she says [] works great."
        }
    
    test_setting = {
        "instruction": "You are a text restoration specialist. Your task is to ONLY fill in the missing content within square brackets [] in the input text. Requirements:\n1. Strictly preserve all existing text and punctuation outside brackets.\n2. Maintain original text structure and formatting.",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "max_new_token": 1024,
        "end_marker": "[END]"
    }

    check_message(model, tokenizer, data, test_setting)
    print(f"original_text: {data['original_text']}")
    
    restore_message(model, tokenizer, data, test_setting)
    print(f"restored_text: {data['restored_text']}")