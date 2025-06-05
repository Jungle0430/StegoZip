import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Any, Optional

def get_prefix(instruction, model_name):
    if "Qwen" in model_name:
        return f"""<|im_start|>system\n{instruction}<|im_end|>
<|im_start|>user\n"""
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
            str_out += (str(array_used[i]) if array_used[i] != -1 else '') + ' '
        return str_out[:-1]
    
    def _encoder_str2bin(self, string_data):
        encoding_map = {
            '0': '101', '1': '0001', '2': '0010', '3': '0011',
            '4': '0100', '5': '0101', '6': '0110', '7': '0111',
            '8': '1000', '9': '1001', ' ': '11'
        }
        binary_string = ''.join(encoding_map[char] for char in string_data)

        return binary_string
        
    def _encode_batch(self,
                     prompt_tokens: np.ndarray,
                     start_pos: int = 0,
                     past_key_values: Optional[Tuple[Any, ...]] = None,
                     is_expand: bool = False,
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
                          is_expand: bool = False,
                          expand_factor: int = 100,
                          use_cache: bool = False,
    ) -> np.ndarray:
        ranks_list = []
        probs_list = []
        
        if with_context_start:
            tokens_full = np.concatenate([context_start, text_tokens])
            start_pos = len(context_start)
        else:
            tokens_full = text_tokens
            start_pos = 1
            
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
            ranks_list[pos] = np.array([-1])
        
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
        flag = False
        for i in range(len(binary_string)):
            temp += binary_string[i]
            if temp == '11':
                if flag:
                    binary_data += '-1 '
                else:
                    flag = True
                    binary_data += ' '
                temp = ''
            elif temp == '101':
                binary_data += '0'
                flag = False
                temp = ''
            elif len(temp) == 4:
                if temp == '0000':
                    break
                binary_data += decoding_map[temp]
                flag = False
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
                     is_expand: bool = False,
                     expand_factor: int = 100):
        str_ranks = self._decoder_bin2str(ranks_code)
        ranks_in = np.fromstring(str_ranks, sep=' ', dtype=np.int64)
        
        batch_size = 1  
        total_length = ranks_in.shape[0]
        
        if with_context_start:
            context_start = np.insert(context_start, 0, 0)
            total_length += len(context_start)
            ranks_in = np.append(np.zeros(len(context_start), dtype=np.int64), ranks_in)
            ranks = torch.tensor(ranks_in).reshape(batch_size,-1).cuda()
        else:
            total_length += 1
            ranks_in = np.append(np.zeros(1, dtype=np.int64), ranks_in)
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
        
        decoded_text = self.tokenizer.decode(tokens.tolist()[0][1:][start_pos-1:])
        
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
    sample_data = random.sample(test_data, sample_size)
    
    # Process samples
    for data in tqdm(sample_data, desc="Processing samples"):
        # Generate restored text
        start_time = time.time()
        text_tokens = np.array(tokenizer.encode(data["compressed_text"]))
        ranks_str, ranks_code = compress_encoder.encode_from_tokens(
            win_size=test_setting["max_new_tokens"],
            text_tokens=text_tokens,
            with_context_start=test_setting["prefix"],
            context_start=prefix_tokens,
            is_expand=False
        )
        end_time = time.time()
        
        data[f"ranks_str_{test_setting['use_lora']}"] = ranks_str
        data[f"ranks_code_{test_setting['use_lora']}"] = ranks_code
        data[f"encode_time_cost"] = round(end_time - start_time, 6)
    
    print(f"=================== Index Encoding Complete ===================")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)
        
    return sample_data

def binary2token(model, tokenizer, test_data, test_setting, output_file):
    compress_decoder = LLMzip_decode(model, tokenizer)
    if test_setting["prefix"]:
        prefix = get_prefix(test_setting["instruction"], test_setting["model_name"])
    else:
        prefix = ""
    prefix_tokens = np.array(tokenizer.encode(prefix))
    
    sample_size = min(test_setting["test_size"], len(test_data))
    print(f"Decoding {sample_size} samples")
    sample_data = random.sample(test_data, sample_size)
    
    # Process samples
    for data in tqdm(sample_data, desc="Processing samples"):
        # Generate restored text
        start_time = time.time()
        decoded_text = compress_decoder.decode_ranks(
            win_size=test_setting["max_new_tokens"],
            ranks_code=data["ranks_code_True"],
            with_context_start=test_setting["prefix"],
            context_start=prefix_tokens,
            is_expand=False
        )
        end_time = time.time()
        
        data[f"decoded_text_{test_setting['use_lora']}"] = decoded_text
        data[f"decode_time_cost"] = round(end_time - start_time, 6)
    
    print(f"=================== Index Decoding Complete ===================")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)
    
    return sample_data
        
def huffman_zip(test_data, output_file):
    from argparse import Namespace
    import heapq
    import random

    class Node:
        def __init__(self, char=None, freq=0, left=None, right=None):
            self.char = char
            self.freq = freq
            self.left = left
            self.right = right

        def __lt__(self, other):
            return self.freq < other.freq

    def build_huffman_tree_from_freq(freq):
        heap = [Node(char=ch, freq=f) for ch, f in freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = Node(char=None, freq=node1.freq + node2.freq, left=node1, right=node2)
            heapq.heappush(heap, merged)

        return heap[0] if heap else None

    def build_huffman_tree(text):
        freq = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        return build_huffman_tree_from_freq(freq)

    def build_huffman_code_table(root, code='', table=None):
        if table is None:
            table = {}
        if root is not None:
            if root.char is not None:
                table[root.char] = code if code != '' else '0'
            else:
                build_huffman_code_table(root.left, code + '0', table)
                build_huffman_code_table(root.right, code + '1', table)
        return table

    def build_global_huffman(texts):
        global_freq = {}
        for text in texts:
            for ch in text:
                global_freq[ch] = global_freq.get(ch, 0) + 1

        tree = build_huffman_tree_from_freq(global_freq)
        code_table = build_huffman_code_table(tree)
        return tree, code_table

    def huffman_encode(text, code_table):
        return ''.join(code_table[ch] for ch in text)

    def huffman_decode(encoded_text, tree):
        decoded_text = ''
        node = tree
        for bit in encoded_text:
            node = node.left if bit == '0' else node.right
            if node and node.char is not None:
                decoded_text += node.char
                node = tree
        return decoded_text
    
    compressed_text_data = [sample['compressed_text'] for sample in test_data]
    original_text_data = [sample['original_text'] for sample in test_data]
    
    tree1, code_table1 = build_global_huffman(compressed_text_data)
    tree2, code_table2 = build_global_huffman(original_text_data)
    
    for sample in test_data:
        sample['base_compressed_code'] = huffman_encode(sample['compressed_text'], code_table1)
        sample['base_original_code'] = huffman_encode(sample['original_text'], code_table2)
        
    print(f"=================== Base Eecoding Complete ===================")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    
    compress_encoder = LLMzip_encode(model, tokenizer)
    compress_decoder = LLMzip_decode(model, tokenizer)
    
    test_text = "Nikkei [] regains 11,000 level TOKYO - Japan #39;s benchmark Nikkei stock index [] recovered [] [] 11,000 [] Monday morning [] widespread [] prompted by advances in US shares last Friday."
    test_tokens = np.array(tokenizer.encode(test_text))
    print(f"test tokens: {test_tokens}")
    ranks_str, ranks_code = compress_encoder.encode_from_tokens(
        win_size=512,
        text_tokens=test_tokens,
        with_context_start=False,
        context_start=None,
        is_expand=False
    )
    print(f"Ranks string: {ranks_str}")
    print(f"Ranks code: {ranks_code}")
    
    decoded_text = compress_decoder.decode_ranks(
        win_size=512,
        ranks_code=ranks_code,
        with_context_start=False,
        context_start=None,
        is_expand=False
    )
    print(f'dencoded text: {decoded_text}')