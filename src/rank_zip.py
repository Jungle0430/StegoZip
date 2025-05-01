import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Any, Optional
from transformers import LogitsProcessor, LogitsProcessorList

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_prefix(instruction, model_name):
    if "Qwen" in model_name:
        return f"""<|im_start|>system\n{instruction}<|im_end|>
<|im_start|>user\n"""
    elif "deepseek" in model_name:
        return f"""System:{instruction}\n\nUser:"""
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
class Rank_Encoder(LogitsProcessor):
    def __init__(self,
                 eos_token_id: int = None,
                 lack_token: int = None):
        self.eos_token_id = eos_token_id
        self.lack_token = lack_token
        
    def initialize(self,
                   text_tokens: List[int]):
        self.text_tokens = text_tokens
        self.current_pos = 0
        self.ranks_list = []
        self.lack_position = np.where(self.text_tokens == self.lack_token[0])[0]
    
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
    
    def get_ranks(self):
        for pos in self.lack_position:
            self.ranks_list[pos] = np.array([-1])
            
        ranks_full = np.concatenate(self.ranks_list, 0).squeeze()

        str_ranks = self._get_str_array(ranks_full)
        ranks_code = self._encoder_str2bin(str_ranks)

        return str_ranks, ranks_code
    
    def __call__(self, input_ids, scores):
        if self.current_pos >= len(self.text_tokens):
            scores[:] = float('-inf')
            scores[:, self.eos_token_id] = 1e8
            return scores
        
        target_id = self.text_tokens[self.current_pos]
        
        sorted_indices = torch.argsort(scores, dim=-1, descending=True)
        rank = (sorted_indices == target_id).nonzero(as_tuple=True)[1].item()
        modified_scores = torch.full_like(scores, float('-inf'))
        modified_scores[0, target_id] = 1e8
        
        self.ranks_list.append(np.array([rank]))
        self.current_pos += 1
        
        return modified_scores

class Rank_Decoder(LogitsProcessor):
    def __init__(self,
                 eos_token_id: int = None,
                 lack_token: int = None):
        self.eos_token_id = eos_token_id
        self.lack_token = torch.tensor(lack_token).long().cuda()
        
    def initialize(self,
                   binary_ranks: str):
        str_ranks = self._decoder_bin2str(binary_ranks)
        ranks_in = np.fromstring(str_ranks, sep=' ', dtype=np.int64)
        self.ranks = torch.tensor(ranks_in).reshape(-1).cuda()
        self.current_pos = 0
        
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
        
    def __call__(self, input_ids, scores):
        if self.current_pos >= len(self.ranks):
            scores[:] = float('-inf')
            scores[:, self.eos_token_id] = 1e8
            return scores
        
        if self.ranks[self.current_pos] == -1:
            scores[:] = float('-inf')
            scores[:, self.lack_token] = 1e8
            self.current_pos += 1
            return scores
        
        probs = torch.softmax(scores[-1], dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True, stable=True)
        
        next_token = torch.gather(probs_idx, -1, self.ranks[self.current_pos])
        modified_scores = torch.full_like(scores, float('-inf'))
        modified_scores[0, next_token] = 1e8
        self.current_pos += 1
        
        return modified_scores
    
def token2binary(model, tokenizer, test_data, test_settings, output_file):
    rank_encoder = Rank_Encoder(tokenizer.eos_token_id, np.array(tokenizer.encode(" []")))
    if test_settings["prefix"]:
        prefix = get_prefix(test_settings["instruction"], test_settings["model_name"])
    else:
        prefix = " "
    prefix_tokens = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
    
    sample_size = min(test_settings["test_size"], len(test_data))
    print(f"Encoding {sample_size} samples")
    sample_data = random.sample(test_data, sample_size)
    
    # Process samples
    for data in tqdm(sample_data, desc="Processing samples"):
        # Generate restored text
        try:
            start_time = time.time()
            text_tokens = tokenizer.encode(data["compressed_text"])
            if 'deepseek' in test_settings["model_name"]:
                text_tokens = text_tokens[1:]
            rank_encoder.initialize(text_tokens)
            
            with torch.no_grad():
                outputs = model.generate(input_ids=prefix_tokens,
                                        max_new_tokens=test_settings["max_new_tokens"],
                                        do_sample=False,
                                        logits_processor=LogitsProcessorList([rank_encoder]),
                                        early_stopping=True,
                                        pad_token_id=tokenizer.pad_token_id)
            ranks_str, ranks_code = rank_encoder.get_ranks()
            end_time = time.time()
            
            data[f"ranks_str_{test_settings['use_lora']}"] = ranks_str
            data[f"ranks_code_{test_settings['use_lora']}"] = ranks_code
            data[f"encode_time_cost"] = round(end_time - start_time, 6)
            
        except Exception as e:
            print(f"Error: {e}")
            print(f"data ID: {data['id']}")
            data[f"ranks_str_{test_settings['use_lora']}"] = ''
            data[f"ranks_code_{test_settings['use_lora']}"] = ''
            data[f"encode_time_cost"] = 0
            continue
    
    print(f"=================== Index Encoding Complete ===================")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)
        
    return sample_data

def binary2token(model, tokenizer, test_data, test_settings, output_file):
    rank_decoder = Rank_Decoder(tokenizer.eos_token_id, np.array(tokenizer.encode(" []")))
    if test_settings["prefix"]:
        prefix = get_prefix(test_settings["instruction"], test_settings["model_name"])
    else:
        prefix = " "
    prefix_tokens = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
    prefix_pos = prefix_tokens.shape[1]
    
    sample_size = min(test_settings["test_size"], len(test_data))
    print(f"Decoding {sample_size} samples")
    sample_data = random.sample(test_data, sample_size)
    
    # Process samples
    for data in tqdm(sample_data, desc="Processing samples"):
        try:
            # Generate restored text
            start_time = time.time()
            rank_decoder.initialize(data["ranks_code_True"])
            with torch.no_grad():
                outputs = model.generate(input_ids=prefix_tokens,
                                        max_new_tokens=test_settings["max_new_tokens"],
                                        do_sample=False,
                                        logits_processor=LogitsProcessorList([rank_decoder]),
                                        early_stopping=True,
                                        pad_token_id=tokenizer.pad_token_id)
            message_tokens = outputs[0].cpu().numpy()[prefix_pos:]
            decoded_text = tokenizer.decode(message_tokens, skip_special_tokens=True)
            end_time = time.time()
            
            data[f"decoded_text_{test_settings['use_lora']}"] = decoded_text
            data[f"decode_time_cost"] = round(end_time - start_time, 6)
    
        except Exception as e:
            print(f"Error: {e}")
            print(f"data ID: {data['id']}")
            data[f"decoded_text_{test_settings['use_lora']}"] = ''
            data[f"decode_time_cost"] = 0
            continue
    
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
    setup_seed(42)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # model_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    rank_encoder = Rank_Encoder(tokenizer.eos_token_id, np.array(tokenizer.encode(" []")))
    rank_decoder = Rank_Decoder(tokenizer.eos_token_id, np.array(tokenizer.encode(" []")))
    
    test_data = "This is a test data for encoding and decoding."
    prefix = get_prefix("Here is a test instruction: ", model_name)
    prefix_tokens = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
    prefix_pos = prefix_tokens.shape[1]
    
    text_tokens = tokenizer.encode(test_data)[1:]
    print(f"Text Tokens: {text_tokens}")
    rank_encoder.initialize(text_tokens)
            
    with torch.no_grad():
        outputs = model.generate(input_ids=prefix_tokens,
                                 max_new_tokens=1024,
                                 do_sample=False,
                                 logits_processor=LogitsProcessorList([rank_encoder]),
                                 early_stopping=True,
                                 pad_token_id=tokenizer.pad_token_id)
    ranks_str, ranks_code = rank_encoder.get_ranks()
    
    rank_decoder.initialize(ranks_code)
    with torch.no_grad():
        outputs = model.generate(input_ids=prefix_tokens,
                                 max_new_tokens=1024,
                                 do_sample=False,
                                 logits_processor=LogitsProcessorList([rank_decoder]),
                                 early_stopping=True,
                                 pad_token_id=tokenizer.pad_token_id)
    message_tokens = outputs[0].cpu().numpy()[prefix_pos:]
    decoded_text = tokenizer.decode(message_tokens, skip_special_tokens=True)
    
    print(f"Decoded Text: {decoded_text}")