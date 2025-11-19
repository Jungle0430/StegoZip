import time
import random
import torch
from scipy.stats import entropy
from math import log2
from typing import List, Dict, Optional, Union

from config import Settings
from huffman import create_huffman_tree
from utils import get_probs_indices_past, SingleExampleOutput, SingleEncodeStepOutput, SingleDecodeStepOutput, set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
table = {}  # token_idx (int) -> [message_decoded (str), n_ptr_consumed (int)]

def encode_decode_step(algo: str,
                       probs: torch.Tensor,
                       indices: List,
                       message_bits: Optional[str] = None,
                       decode: Optional[str] = None,
                       stego_t: Optional[int] = None) -> Optional[Union[SingleEncodeStepOutput, SingleDecodeStepOutput, Dict]]:
    probs_cumsum = probs.cumsum(dim=0)
    interval_begin = torch.cat((torch.tensor([0], device=probs_cumsum.device), probs_cumsum[:-1]), dim=0)
    
    def ptr_to_index(ptr: float) -> int:
        index_idx = (ptr >= interval_begin).nonzero()[-1].item()
        index = indices[index_idx]
        return index

    if algo == 'sample':
        return SingleEncodeStepOutput(ptr_to_index(random.random()), 0, entropy(probs.tolist(), base=2), 0, 1,
                                      -log2(probs[0].item()))
        
    elif algo == 'Discop_baseline':
        if decode is None:
            # Determine capacity
            capacity = int(log2(1 / probs[0]))
            capacity_upper_bound = capacity + 1

            dc_tbl = {}  # message_bits to idx
            ptr = random.random()

            while capacity <= capacity_upper_bound:
                shift_distance = 2**-capacity
                is_available = True
                dc_tbl_new = {}
                for i in range(2**capacity):
                    ptr_i = ptr + i * shift_distance
                    if ptr_i > 1.0:
                        ptr_i -= 1
                    idx_order = (ptr_i >= interval_begin).nonzero()[-1].item()
                    idx = indices[idx_order]
                    if idx in dc_tbl_new.values(
                    ):  # If `idx` already exists, it means that the pairwise difference is not satisfied
                        is_available = False
                        break
                    dc_tbl_new[i] = idx
                if not is_available:
                    break
                dc_tbl = dc_tbl_new
                capacity += 1
            capacity -= 1

            if capacity < 1:  # Cannot embed message, but still needs to return a token
                return SingleEncodeStepOutput(ptr_to_index(ptr), 0, entropy(probs.tolist(), base=2), 0, 1, -log2(probs[0].item()))
            return SingleEncodeStepOutput(dc_tbl[int(message_bits[:capacity], 2)], capacity, entropy(probs.tolist(), base=2), 0,
                                          1, -log2(probs[0].item()))
        elif decode == 'directly':
            pass
        elif decode == 'table':
            pass

    elif algo == 'Discop':
        if type(probs) == torch.Tensor:
            probs = probs.tolist()
        node = create_huffman_tree(indices=indices, probs=probs, search_for=stego_t)
        code_len = 0
        d = 0  # n_ptr_consumed
        if decode is None:
            while not node.is_leaf():
                probs_sum = node.prob
                ptr = random.random()
                ptr_0 = ptr * probs_sum
                ptr_1 = (ptr + 0.5) * probs_sum
                if ptr_1 > probs_sum:
                    ptr_1 -= probs_sum
                path_table = {}  # message_i (str) -> selected subtree (node)

                path_table['0'] = node.left if ptr_0 < node.left.prob else node.right
                path_table['1'] = node.left if ptr_1 < node.left.prob else node.right

                node = path_table[message_bits[code_len]]
                if path_table['0'] != path_table['1']:  # can embed
                    code_len += 1
                d += 1
            return SingleEncodeStepOutput(node.index, code_len, entropy(probs, base=2), 0, d, -log2(probs[0]))
        elif decode == 'directly':
            message_decoded = ''
            while not node.is_leaf():
                probs_sum = node.prob
                ptr = random.random()
                ptr_0 = ptr * probs_sum
                ptr_1 = (ptr + 0.5) * probs_sum
                if ptr_1 > probs_sum:
                    ptr_1 -= probs_sum
                path_table = {}  # message_i (str) -> selected subtree (int)

                # -1 corresponds to left
                # +1 corresponds to right
                path_table['0'] = -1 if ptr_0 < node.left.prob else 1
                path_table['1'] = -1 if ptr_1 < node.left.prob else 1

                if path_table['0'] != path_table['1']:  # can embed 1 bit
                    path_table = dict(zip(path_table.values(), path_table.keys()))  # selected subtree (int) -> message_i (str)
                    if node.search_path is None:  # fail to decode
                        return
                    message_decoded += path_table[node.search_path]
                    if node.search_path == -1:
                        node = node.left
                    elif node.search_path == 1:
                        node = node.right
                else:
                    if path_table['0'] == -1:
                        node = node.left
                    else:
                        node = node.right
                d += 1
            if node.search_path != 0:  # fail to decode
                return
            return SingleDecodeStepOutput(message_decoded, d)

    elif algo == 'ADG':
        from adg import adg_encode_decode_step
        if decode is None:
            return adg_encode_decode_step(probs, 
                                          torch.tensor(indices, dtype=int, device=probs.device), 
                                          message_bits, 
                                          decode=decode, 
                                          stego_t=None)
        else:
            return adg_encode_decode_step(probs,
                                          torch.tensor(indices, dtype=int, device=probs.device),
                                          message_bits=None,
                                          decode=decode,
                                          stego_t=stego_t)


@torch.no_grad()
def encode_text(model,
                tokenizer,
                message_bits: Optional[str] = None,
                prompt: str = None,
                settings: Settings = Settings(),
                verbose: bool = False,
                segment: Optional[int] = None) -> SingleExampleOutput:
    # General architecture of Steganography Encoding (message_bits -> English text)
    algo, temp, top_p, length, seed = settings()
        
    if algo == 'Arithmetic':
        from arithmetic import encode_arithmetic
        return encode_arithmetic(prompt, message_bits, settings, verbose)
    elif algo == 'Meteor':
        from meteor import encode_meteor
        print("Meteor Encode.")
        return encode_meteor(model, tokenizer, prompt, message_bits, settings, verbose)
    
    message_length = len(message_bits)
    message_bits += '0' * 500

    if algo != 'sample' and (message_bits is None or len(message_bits) == 0):
        raise ValueError
    if segment is not None and algo != 'Discop':
        raise NotImplementedError
    if verbose:
        print('=' * 40 + 'Encoding' + '=' * 40)

    start = time.time()

    set_seed(seed)
    prompt = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(device)

    past = None  # pass into the `past_keys_values` for speed-up
    prev = prompt  # indices that were never passed to the model before
    generated_ids = None

    total_capacity = 0
    total_entropy = 0
    total_minimum_entropy = 0
    total_log_probs = 0  # for perplexity
    total_kld = 0
    max_kld = 0
    t = 0
    finish_flag = False
    
    eos_tokens = [".", "!", "?", '."', "...", ";", ")", "]", "}"]
    eos_tokens_id = [tokenizer.convert_tokens_to_ids(token) for token in eos_tokens]
    
    while t < length:
        if segment is None:
            # probs, indices, past = get_probs_indices_past(model, prev, past, top_p, temp)
            probs, indices, past = get_probs_indices_past(model, prev, past, settings)
            indices = indices.tolist()
        else:
            probs = torch.tensor([], dtype=int, device=device)
            indices = []  # paired
            probs_1, indices_1, past = get_probs_indices_past(model, prev, past, top_p, temp)
            for i in range(len(indices_1)):
                probs_2, indices_2, past_2 = get_probs_indices_past(model,
                                                                    torch.tensor([indices_1[i]], device=device).unsqueeze(0),
                                                                    past, top_p, temp)
                probs = torch.cat((probs, probs_1[i] * probs_2))
                indices.extend(list([indices_1[i].item(), x] for x in indices_2.tolist()))

        sampled_index, capacity_t, entropy_t, kld_step, n_ptr_consumed, min_entropy_t = encode_decode_step(algo, probs, indices, message_bits)()

        indices_idx = indices.index(sampled_index)
        total_entropy += entropy_t
        total_minimum_entropy += min_entropy_t
        total_log_probs += log2(probs[indices_idx].item())
        total_kld += kld_step
        if kld_step > max_kld:
            max_kld = kld_step

        # when `capacity_t == 0`, cannot embed message, but still needs to return a token_index
        if capacity_t > 0:
            total_capacity += capacity_t
            message_bits = message_bits[capacity_t:]  # remove the encoded part of `message_bits`

        # print(sampled_index)
        if generated_ids is None:
            generated_ids = [sampled_index] if type(sampled_index) == int else sampled_index
        else:
            if type(sampled_index) == int:
                generated_ids.append(sampled_index)
            else:
                generated_ids.extend(sampled_index)
        if segment is None:
            t += 1
            prev = torch.tensor([sampled_index], device=device).unsqueeze(0)
        else:
            t += 2
            prev = torch.tensor(sampled_index, device=device).unsqueeze(0)
            
        if total_capacity >= message_length:
            finish_flag = True

        if finish_flag and sampled_index in eos_tokens_id:
            break

    end = time.time()
    embedding_efficiency = total_capacity / total_entropy if total_entropy != 0 else 0
    perplexity = 2 ** (-1 / length * total_log_probs)
    ave_kld = total_kld / length
    stego_object = tokenizer.decode(generated_ids)

    if verbose:
        print(generated_ids)
    
    return SingleExampleOutput(generated_ids, stego_object,
                               total_capacity, total_entropy,
                               ave_kld, max_kld, 
                               perplexity, end - start, 
                               settings, total_minimum_entropy)
    

def decode_text(model,
                tokenizer,
                stego: Union[str, List[int]],
                prompt: str,
                settings: Settings = Settings(),
                verbose: bool = False) -> str:
    # General architecture of Steganography Decoding (English text -> message_bits)
    # Returns `message_decoded`
    algo, temp, top_p, length, seed = settings()
        
    if algo == 'Meteor':
        from meteor import decode_meteor
        print("Meteor Decode.")
        return decode_meteor(model, tokenizer, prompt, stego, settings, verbose)

    if verbose:
        print('=' * 40 + 'Decoding' + '=' * 40)
        
    set_seed(seed)

    prompt = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(device)

    past = None  # pass into the `past_keys_values` for speed-up
    prev = prompt  # indices that were never passed to the model before
    message_decoded = ''
    
    if type(stego) == list:
        t = 0
        while t < len(stego):
            probs, indices, past = get_probs_indices_past(model, prev, past, settings)
            indices = indices.tolist()
            message_decoded_t = encode_decode_step(algo, probs, indices, decode='directly', stego_t=stego[t])

            if message_decoded_t is None:
                message_decoded = ''
                print("Fail to decode!")
                break

            prev = torch.tensor([stego[t]], device=device).unsqueeze(0)
            t += 1
            if message_decoded_t != '-1':
                message_decoded += message_decoded_t
                
            if len(message_decoded) > 4 and message_decoded[:4] != "0000":
                message_decoded = ''
                print("Fail to decode!")
                break

        return message_decoded