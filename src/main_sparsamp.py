import sys
import time
import json
import torch
import random
from tqdm import tqdm
from math import ceil

sys.path.append('src/stega')
from model import load_model
from utils import func_mrn, dec2bin, get_lower_upper_bound, get_probs_indices_past
from config import Settings, text_default_settings_sparsamp

def read_random_binary_from_file(file_path, length):
    with open(file_path, 'r') as file:
        content = file.read()
    
    if len(content) < length:
        raise ValueError("key stream is too short!")
    
    start_index = random.randint(0, len(content) - length)
    random_binary = content[start_index:start_index + length]
    
    return random_binary

def encode_step(probs, n_m, k_m):
    r = random.random()
    cumulative_probs = probs.cumsum(0)
    r_i_m = func_mrn(k_m, n_m, r)
    token_index = (cumulative_probs > r_i_m).nonzero()[0].item()

    SE = get_lower_upper_bound(cumulative_probs, token_index)
    temp0 = ceil((SE[0] - r) * n_m)
    temp1 = ceil((SE[1] - r) * n_m)

    if k_m + r * n_m >= n_m:
        k_m = k_m - n_m - temp0
    else:
        k_m = k_m - temp0
    n_m = temp1 - temp0
    return token_index, n_m, k_m

def pad_message_bits(message_bits, block_size):
    remainder = len(message_bits) % block_size
    if remainder != 0:
        message_bits += '0' * (block_size - remainder)
    return message_bits

@torch.no_grad()
def encode_spar(model, context, message_bits, settings, random_seed=42, block_size=32):
    context = torch.tensor(context[-1022:], device=settings.device, dtype=torch.long)

    generated_ids = []
    m_index = 0
    k_m = int(message_bits[:block_size], 2)
    n_m = 2**block_size
    token_num_generated = 0
    random.seed(random_seed)
    encoded_message = []
    past = None
    prev = context

    while True:
        probs, indices, past = get_probs_indices_past(model=model,
                                                      prev=prev,
                                                      past=past,
                                                      settings=settings)

        probs = probs.to(torch.float64)
        token_index, n_m, k_m = encode_step(probs=probs,
                                            n_m=n_m,
                                            k_m=k_m)
        tokenID = indices[token_index]
        token_num_generated += 1
        if token_num_generated < settings.length:
            if n_m == 1:
                encoded_message.append(message_bits[m_index:m_index + block_size])
                m_index += block_size
                if m_index + block_size > len(message_bits):
                    generated_ids.append(tokenID.item())
                    break
                n_m = 2 ** block_size
                k_m = int(message_bits[m_index:m_index + block_size], 2)
        else:
            if n_m == 1:
                encoded_message.append(message_bits[m_index:m_index + block_size])
                m_index += block_size
                generated_ids.append(tokenID.item())
                break
            if token_num_generated > 12000:
                print(f"We have generated more than 12000 tokens, but this block message still not embedded over. This context seems have problem. let's skip it.")
                raise Exception("This context seems have problem. let's skip it.")
        generated_ids.append(tokenID.item())
        prev = torch.tensor([tokenID], device=settings.device, dtype=torch.long).unsqueeze(0)

    return generated_ids


@torch.no_grad()
def decode_spar(model, context, generated_ids, settings, random_seed=42, block_size=32):
    context = torch.tensor(context[-1022:], device=settings.device, dtype=torch.long)

    random.seed(random_seed)
    message = []
    n_m = 2 ** block_size
    k_m = 0
    n_m_arr = []
    temp0_arr = []
    temp1_arr = []
    past = None
    prev = context

    for tokenID in generated_ids:
        r = random.random()
        probs, indices, past = get_probs_indices_past(model=model,
                                                      prev=prev,
                                                      past=past,
                                                      settings=settings)
        probs = probs.to(torch.float64)
        cumulative_probs = probs.cumsum(0)

        token_index = torch.where(indices == tokenID)[0]
        SE = get_lower_upper_bound(cumulative_probs, token_index)

        temp0 = ceil((SE[0] - r) * n_m)
        temp1 = ceil((SE[1] - r) * n_m)

        n_m = temp1 - temp0
        temp0_arr.append(temp0)
        temp1_arr.append(temp1)
        n_m_arr.append(n_m)

        if n_m == 1:
            count = len(temp0_arr) - 2
            k_m = temp0_arr[count + 1]
            while count >= 0:
                n_m_new = n_m_arr[count]
                k_m = temp0_arr[count] + ((k_m + n_m_new) % n_m_new)
                count -= 1
            k_m = (k_m + 2 ** block_size) % 2 ** block_size
            temp0_arr = []
            temp1_arr = []
            n_m_arr = []
            message.append(dec2bin(k_m, block_size))
            n_m = 2 ** block_size
        prev = torch.tensor([tokenID], device=settings.device, dtype=torch.long).unsqueeze(0)

    return message

def sparsamp_stego(test_data, stego_prompt, seed, output_file, temperature, use_lora=True, block_size=32):
    print("Running SparSamp ……")
    settings = text_default_settings_sparsamp
    settings.temp = temperature
    
    model, tokenizer = load_model(settings)
    result_data = []
    
    for i, sample in tqdm(enumerate(test_data)):
        message_binary = sample[f'ranks_code_{use_lora}']
        message_bits = pad_message_bits(message_binary, block_size)
        
        key_stream_file = "src/key_stream.txt"
        random_binary = read_random_binary_from_file(key_stream_file, len(message_binary))
        xor_result = ''.join(str(int(a) ^ int(b)) for a, b in zip(message_bits, random_binary))
        
        context = tokenizer.encode(stego_prompt[i], return_tensors='pt').to(settings.device)

        start1 = time.time()
        generated_ids = encode_spar(model, context, xor_result, settings=settings, random_seed=seed, block_size=block_size)
        end1 = time.time()
        
        start2 = time.time()
        xor_result_decoded = decode_spar(model, context, generated_ids, settings=settings, random_seed=seed, block_size=block_size)
        message_binary_decoded = ''.join(str(int(a) ^ int(b)) for a, b in zip(xor_result_decoded, random_binary))
        end2 = time.time()
        
        if message_binary_decoded == "":
            continue
        
        sample['stego'] = {
            'stego_text': tokenizer.decode(generated_ids),
            'encode_time': end1 - start1,
            'decode_time': end2 - start2,
            'message_binary_decoded': message_binary_decoded
        }
        
        result_data.append(sample)
        
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    return result_data