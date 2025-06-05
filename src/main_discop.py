import sys
import time
import json
import random
from tqdm import tqdm

sys.path.append('src/stega')
from model import load_model
from utils import SingleExampleOutput
from config import Settings, text_default_settings_discop

def read_random_binary_from_file(file_path, length):
    with open(file_path, 'r') as file:
        content = file.read()
    
    if len(content) < length:
        raise ValueError("key stream is too short!")
    
    start_index = random.randint(0, len(content) - length)
    random_binary = content[start_index:start_index + length]
    
    return random_binary

def text_encode(settings: Settings, model, tokenizer, message, prompt, key):
    if settings.algo == 'sample':
        from random_sample_cy import encode_text
    elif settings.algo in ['Discop', 'Discop_baseline']:
        from stega_cy import encode_text
    else:
        raise NotImplementedError("encode algorithm must belong to {'Discop', 'Discop_baseline', 'sample'}!")

    settings.seed = key
    single_example_output: SingleExampleOutput = encode_text(model, tokenizer, message, prompt, settings)
    # print(f"single_example_output.stego_object: {single_example_output.stego_object[:10]}")
    # print(f"single_example_output.ids: {single_example_output.generated_ids}")
    
    return single_example_output.stego_object, single_example_output.generated_ids

def text_decode(settings: Settings, model, tokenizer, stego_ids, prompt, key):
    # choose algorithm
    if settings.algo in ['Discop', 'Discop_baseline']:
        from stega_cy import decode_text
    else:
        raise NotImplementedError("decode algorithm must belong to {'Discop', 'Discop_baseline'}!")
    settings.seed = key

    # stego_ids = tokenizer(stego, return_tensors='pt')['input_ids'][0].tolist()
    # stego_ids = stego_ids[1:]
    # print(f"{settings.algo} stego_ids2: {stego_ids}")
    
    message_decoded = decode_text(model, tokenizer, stego_ids, prompt, settings)
    
    return message_decoded
        
def discop_stego(test_data, stego_prompt, seed, output_file, temperature, use_lora=True):
    settings = text_default_settings_discop
    settings.temp = temperature

    model, tokenizer = load_model(settings)
    result_data = []

    for i, sample in tqdm(enumerate(test_data)):
        message_binary = sample[f'ranks_code_{use_lora}']
        
        key_stream_file = "src/key_stream.txt"
        random_binary = read_random_binary_from_file(key_stream_file, len(message_binary))
        
        start1 = time.time()
        xor_result = ''.join(str(int(a) ^ int(b)) for a, b in zip(message_binary, random_binary))
        stego_text, stego_ids = text_encode(settings, model, tokenizer, xor_result, stego_prompt[i], seed)
        end1 = time.time()
        
        start2 = time.time()
        xor_result_decoded = text_decode(settings, model, tokenizer, stego_ids, stego_prompt[i], seed)
        message_binary_decoded = ''.join(str(int(a) ^ int(b)) for a, b in zip(xor_result_decoded, random_binary))
        end2 = time.time()
        
        if message_binary_decoded == "":
            continue
        
        sample['stego'] = {
            'stego_text': stego_text,
            'encode_time': end1 - start1,
            'decode_time': end2 - start2,
            'message_binary_decoded': message_binary_decoded
        }
        
        result_data.append(sample)
        
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
        
    return result_data