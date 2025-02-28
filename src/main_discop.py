import sys
import time
import json
import random
from tqdm import tqdm

sys.path.append('src/stego')
from model import load_model
from stego_utils import *
from config import Settings, text_default_settings_discop, text_default_settings_sample

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
    
    return single_example_output.stego_object

def text_decode(settings: Settings, model, tokenizer, stego, prompt, key, flag=False):
    # choose algorithm
    if settings.algo in ['Discop', 'Discop_baseline']:
        from stega_cy import decode_text
    else:
        raise NotImplementedError("decode algorithm must belong to {'Discop', 'Discop_baseline'}!")
    settings.seed = key
    
    # stego_ids = tokenizer(stego, return_tensors='pt')['input_ids'][0].tolist()
    # print(f"stego_ids: {stego_ids[:10]}")
    
    stego = stego.replace('<s>', ' <s>')
    stego = stego.replace('</s>', ' </s>')
    if flag:
        stego = '<s>' + stego

    stego_ids = tokenizer(stego, return_tensors='pt')['input_ids'][0].tolist()
    
    if flag:
        stego_ids = stego_ids[1:]
    if stego_ids[0] == 1:
        stego_ids = stego_ids[1:]
    if stego_ids[0] == 29871:
        stego_ids = stego_ids[1:]
         
    # print(f"{settings.algo} stego_ids2: {stego_ids}")
    
    message_decoded = decode_text(model, tokenizer, stego_ids, prompt, settings)
    
    return message_decoded
        
def discop_stego(test_data, stego_prompt, seed, output_file):
    settings = text_default_settings_discop
    settings_sample = text_default_settings_sample

    model, tokenizer = load_model(settings)

    for i, sample in tqdm(enumerate(test_data)):
        message_binary = sample['ranks_code_True']
        
        random_binary = ''.join(random.choice('01') for _ in range(len(message_binary)))
        xor_result = ''.join(str(int(a) ^ int(b)) for a, b in zip(message_binary, random_binary))
        
        start1 = time.time()
        stego_text = text_encode(settings, model, tokenizer, xor_result, stego_prompt[i], seed)
        end1 = time.time()
        
        start2 = time.time()
        message_binary_decoded = text_decode(settings, model, tokenizer, stego_text, stego_prompt[i], seed)
        end2 = time.time()
        
        sample['stego'] = {
            'stego_text': stego_text,
            'encode_time': end1 - start1,
            'decode_time': end2 - start2,
        }
        
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)