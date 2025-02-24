import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Any, Optional

def get_prefix(instruction, model_name):
    if "Qwen" in model_name:
        return f"""### Instruction: {instruction} \n### Input: """
    elif "vicuna" in model_name:
        return f"""SYSTEM: {instruction} \nUSER: """
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
class LLMzip_encode:
    def __init__(self, model, tokenizer):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.lack_token = np.array(tokenizer.encode(" []"))
        
    def _gen_rank(self, probs, next_token):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True, stable=True)
        rank = torch.where(probs_idx == next_token)[-1]
        prob = probs_sort[rank]
        return rank, prob
    
    def _get_str_array(self, array):
        array_used = array.reshape(-1)
        str_out = str()
        for i in range(array_used.size):
            str_out += (str(array_used[i]) if array_used[i] != -1 else ' ') + ' '
        return str_out
    
    def _encoder_str2bin(self, string_data):
        encoding_map = {
            '0': '101', '1': '0001', '2': '0010', '3': '0011',
            '4': '0100', '5': '0101', '6': '0110', '7': '0111',
            '8': '1000', '9': '1001', ' ': '11'
        }
        binary_string = ''.join(encoding_map[char] for char in string_data)
        # binary_string += '0000'

        return binary_string
        
    def _encode_batch(self,
                     prompt_tokens: np.ndarray,
                     start_pos: int = 0,
                     past_key_values: Optional[Tuple[Any, ...]] = None,
                     is_expand: bool = True,
                     expand_factor: int = 100,
    ) -> np.ndarray:
        batch_size, prompt_size = prompt_tokens.shape
        
        tokens = torch.full((batch_size, prompt_size), self.pad_id).cuda().long()
        tokens[:batch_size, :prompt_size] = torch.tensor(prompt_tokens).long()
        rank_list = []
        prob_list = []
        
        outputs = self.model.forward(tokens)
        logits = outputs['logits'][0].to(tokens.device)

        if is_expand:
            logits = logits * expand_factor
        
        probs = torch.softmax(logits, dim=-1)
        for cur_pos in range(start_pos, tokens.shape[1]):
            rank, prob = self._gen_rank(probs[cur_pos - 1], next_token=tokens[:, cur_pos])
            rank_list.append(rank.cpu().numpy())
            prob_list.append(prob.detach().cpu().numpy())
           
        return rank_list, prob_list
    
    def encode_from_tokens(self,
                          win_size: int,
                          text_tokens: Optional[np.ndarray] = None,
                          with_context_start: bool = False,
                          context_start: Optional[np.ndarray] = None,
                          is_expand: bool = True,
                          expand_factor: int = 100,
                          use_cache: bool = False,
    ) -> np.ndarray:
        ranks_list = []
        probs_list = []
        start_pos = 1
        
        if with_context_start:
            tokens_full = np.concatenate([context_start, text_tokens])
            start_pos = len(context_start)
        
        tokens_in = np.array([tokens_full[ :win_size].tolist()])
        tokens_in = np.insert(tokens_in, 0, 0, axis=1)
        ranks, probs = self._encode_batch(tokens_in, 1, is_expand=is_expand, expand_factor=expand_factor)
        ranks_list += ranks
        probs_list += probs
        
        half_win_size = win_size // 2
        n_batches = np.ceil(tokens_full.size / half_win_size).astype(int) - 1
        for b_ind in range(1, n_batches):
            token_start = b_ind * half_win_size
            token_stop = np.minimum(tokens_full.size, (b_ind + 2) * half_win_size)
            tokens_batch = np.array([tokens_full[token_start : token_stop].tolist()])
            
            ranks, probs = self._encode_batch(tokens_batch, half_win_size, is_expand=is_expand, expand_factor=expand_factor)
            ranks_list += ranks
            probs_list += probs
          
        if with_context_start:
            ranks_list = ranks_list[start_pos:]
            # probs_list = probs_list[start_pos:]
        
        lack_position = np.where(text_tokens == self.lack_token[0])[0]
        for pos in lack_position:
            ranks_list[pos] = -1
            
        ranks_full = np.concatenate(ranks_list, 0).squeeze()
        # probs_full = np.concatenate(probs_list, 0).squeeze()

        str_ranks = self._get_str_array(ranks_full)
        ranks_code = self._encoder_str2bin(str_ranks)
        
        return str_ranks, ranks_code

class LLMzip_decode:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.start_token = self.pad_id
        self.vocab = tokenizer.get_vocab()
        self.lack_token = torch.tensor(tokenizer.encode(" []")).long().cuda()
        
    def _decoder_bin2str(self, binary_string):
        decoding_map = {
            '0001': '1', '0010': '2', '0011': '3', '0100': '4',
            '0101': '5', '0110': '6', '0111': '7', '1000': '8',
            '1001': '9', '101': '0', '11': ' '
        }
        
        temp = ''
        binary_data = ''
        for i in range(len(binary_string)):
            if binary_string[i] == 'x':
                break
            temp += binary_string[i]
            if temp == '11':
                binary_data += ' '
                temp = ''
            elif temp == '101':
                binary_data += '0'
                temp = ''
            elif len(temp) == 4:
                if temp == '0000':
                    break
                binary_data += decoding_map[temp]
                temp = ''

        return binary_data
        
    def _gen_next_token(self, probs, rank):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True, stable=True)
        next_token = torch.gather(probs_idx, -1, rank)
        return next_token
        
    def decode_ranks(self,
                     win_size,
                     ranks_code,
                     with_context_start: bool = False,
                     context_start: Optional[np.ndarray] = None,
                     is_expand: bool=True,
                     expand_factor: int=100):
        str_ranks = self._decoder_bin2str(ranks_code).replace('   ', ' -1 ')
        ranks_in = np.fromstring(str_ranks, sep=' ', dtype=np.int64)
        
        batch_size = 1  
        total_length = ranks_in.shape[0]
        
        if with_context_start:
            context_start = np.insert(context_start, 0, 0)
            total_length += len(context_start)
            ranks_in = np.append(np.zeros(len(context_start), dtype=np.int64), ranks_in)
            ranks = torch.tensor(ranks_in).reshape(batch_size,-1).cuda()
        else:
            ranks = torch.tensor(ranks_in).reshape(batch_size, -1).cuda()

        tokens = torch.full((batch_size, total_length), self.pad_id).long()

        if with_context_start:
            tokens[:, :len(context_start)] = torch.tensor(context_start).long()
            start_pos = len(context_start)
        else:
            tokens[:, 0] = self.pad_id
            start_pos = 1
        tokens = tokens.cuda()
        
        for cur_pos in range(start_pos, total_length):
            if ranks[:, cur_pos] == -1:
                tokens[:, cur_pos] = self.lack_token[0]
                continue
            
            logits = self.model.forward(tokens[:, :cur_pos])['logits'][0][-1]
            if is_expand:
                logits = logits * expand_factor

            probs = torch.softmax(logits, dim=-1)
            next_token = self._gen_next_token(probs, ranks[:, cur_pos])
            tokens[:, cur_pos] = next_token.clone().detach().long()
            
        decoded_text = self.tokenizer.decode(tokens.tolist()[0][1:][len(context_start)-1:])
        
        return decoded_text
    
def token2binary(model, tokenizer, test_data, test_setting, output_file):
    compress_encoder = LLMzip_encode(model, tokenizer)
    if test_setting["prefix"]:
        prefix = get_prefix(test_setting["instruction"], test_setting["model_name"])
    else:
        prefix = ""
    prefix_tokens = np.array(tokenizer.encode(prefix))
    
    sample_size = min(test_setting["test_size"], len(test_data))
    print(f"Encoding {sample_size} samples")
    sampled_data = random.sample(test_data, sample_size)
    
    # Process samples
    for data in tqdm(sampled_data, desc="Processing samples"):
        # Generate restored text
        start_time = time.time()
        text_tokens = np.array(tokenizer.encode(data["compressed_text"].replace("[LACK]", "[]")))
        ranks_str, ranks_code = compress_encoder.encode_from_tokens(
            win_size=test_setting["win_size"],
            tokens_full=text_tokens,
            with_context_start=test_setting["prefix"],
            context_start=prefix_tokens,
            is_expand = True
        )
        end_time = time.time()
        
        data["ranks_str"] = ranks_str
        data["ranks_code"] = ranks_code
        data["encode_time_cost"] = round(end_time - start_time, 6)
    
    print(f"=================== Index Encoding Complete ===================")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

def binary2token(model, tokenizer, test_data, test_setting, output_file):
    compress_decoder = LLMzip_decode(model, tokenizer)
    if test_setting["prefix"]:
        prefix = get_prefix(test_setting["instruction"], test_setting["model_name"])
    else:
        prefix = ""
    prefix_tokens = np.array(tokenizer.encode(prefix))
    
    sample_size = min(test_setting["test_size"], len(test_data))
    print(f"Encoding {sample_size} samples")
    sampled_data = random.sample(test_data, sample_size)
    
    # Process samples
    for data in tqdm(sampled_data, desc="Processing samples"):
        # Generate restored text
        start_time = time.time()
        decoded_text = compress_decoder.decode_ranks(
            win_size=test_setting["win_size"],
            ranks_code=data["ranks_code"],
            with_context_start=test_setting["prefix"],
            context_start=prefix_tokens,
            is_expand=True
        ).replace("[]", "[LACK]")
        end_time = time.time()
        
        data["decoded_text"] = decoded_text
        data["decode_time_cost"] = round(end_time - start_time, 6)
    
    print(f"=================== Index Decoding Complete ===================")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=4)

    