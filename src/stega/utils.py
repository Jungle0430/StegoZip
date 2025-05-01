import os
import random
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

from config import Settings

class SingleEncodeStepOutput:
    def __init__(self,
                 sampled_index: Union[int, List[int]],
                 total_capacity: int,
                 entropy: float,
                 kld: float,
                 n_ptr_consumed: int,
                 minimum_entropy: float = 0) -> None:
        self.sampled_index = sampled_index
        self.total_capacity = total_capacity
        self.entropy = entropy
        self.kld = kld
        self.n_ptr_consumed = n_ptr_consumed
        self.minimum_entropy = minimum_entropy

    def __call__(self) -> Tuple:
        return self.sampled_index, self.total_capacity, self.entropy, self.kld, self.n_ptr_consumed, self.minimum_entropy


class SingleDecodeStepOutput:
    def __init__(self, message_decoded, n_ptr_consumed) -> None:
        self.message_decoded = message_decoded
        self.n_ptr_consumed = n_ptr_consumed

    def __call__(self) -> Tuple:
        return self.message_decoded, self.n_ptr_consumed
        
# Sampling (Encoding) results and statistics for single example
class SingleExampleOutput:
    def __init__(self, 
                 generated_ids, 
                 stego_object, 
                 total_capacity, 
                 total_entropy, 
                 ave_kld, 
                 max_kld, 
                 perplexity, 
                 time_cost, 
                 settings,
                 total_minimum_entropy):
        self.generated_ids = generated_ids
        self.stego_object = stego_object
        self.algo = settings.algo
        self.temp = settings.temp
        self.top_p = settings.top_p
        self.total_capacity = total_capacity
        if generated_ids is not None:
            self.n_tokens = len(generated_ids)
        else:
            self.n_tokens = len(stego_object)
        self.total_entropy = total_entropy
        self.ave_kld = ave_kld
        self.max_kld = max_kld
        self.embedding_rate = total_capacity / self.n_tokens
        self.utilization_rate = total_capacity / total_entropy if total_entropy != 0 else 0
        self.perplexity = perplexity
        self.time_cost = time_cost
        self.total_minimum_entropy = total_minimum_entropy

    def __str__(self) -> str:
        d = self.__dict__
        excluded_attr = ['generated_ids']
        selected_attr = list(d.keys())
        for x in excluded_attr:
            selected_attr.remove(x)
        return '\n'.join('{} = {}'.format(key, d[key]) for key in selected_attr)


def set_seed(sd):
    random.seed(sd)


# The token indices should be filtered out and their corresponding reasons
# https://huggingface.co/gpt2/raw/main/vocab.json
# filter_out_indices_gpt = {
#     -1: "endoftext can't happen",
#     198: "1 newline can't happen",
#     628: "2 newlines can't happen",
#     220: "just one space can't happen",
#     302: "`\u0120re` can't happen",
#     797: "`\u0120Re` can't happen",
#     15860: "`\u0120Enh` can't happen",
#     2943: "`EC` can't happen",
#     764: "`\u0120.` (764) may cause failed decoding to `.` (13)",
#     837: "`\u0120,` (837) may cause failer decoding to `,` (11)"
# }
filter_out_indices_gpt = {
    -1: "endoftext can't happen",
    198: "1 newline can't happen",
    628: "2 newlines can't happen",
    764: "`\u0120.` (764) may cause failed decoding to `.` (13)",
    837: "`\u0120,` (837) may cause failer decoding to `,` (11)"
}
contain_dollar_lst = [
    3, 720, 7198, 13702, 16763, 17971, 22799, 25597, 29568, 29953, 32047, 32382, 32624, 34206, 35307, 36737, 38892, 39280, 40111,
    43641, 45491, 47113, 48082
]
contain_bad_ellipsis_lst = [19424, 20004, 39864, 44713, 44912, 47082]


def gen_random_message(seed=None, length: int = 1000, save_path: str = os.path.join('temp', 'message.txt')) -> None:
    # Generating binary message (str) randomly via build-in `random` lib
    import random
    random.seed(seed)

    message = ''
    for _ in range(length):
        message += str(random.randint(0, 1))
    print(message)

    if save_path is None:
        return message
    with open(save_path, 'w', encoding='utf-8') as fout:
        fout.write(message)


def limit_past(past):
    if past is None:
        return None
    past = list(past)
    for i in range(len(past)):
        past[i] = list(past[i])
        for j in range(len(past[i])):
            past[i][j] = past[i][j][:, :, -1022:]
    return past


@torch.no_grad()
def get_probs_indices_past(model,
                           prev=None,
                           past=None,
                           settings: Settings = Settings(),
                           gpt_filter: bool = True) -> Tuple:
    # first, get logits from the model
    if settings.task == 'text':
        if 'gpt2' in settings.model_name:
            past = limit_past(past)
            model_output = model(prev, past_key_values=past)
            past = model_output.past_key_values
            logits = model_output.logits[0, -1, :].to(settings.device)
            if gpt_filter:
                for ele in filter_out_indices_gpt.keys():
                    logits[ele] = -1e10
        elif settings.model_name == 'transfo-xl-wt103':
            model_output = model(prev, mems=past)
            past = model_output.mems
            logits = model_output.logits[0, -1, :].to(settings.device)
            logits[0] = -1e10  # <eos>
            logits[24] = -1e10  # <unk>
        elif settings.model_name in ['LLaMA-7B']:
            past = limit_past(past)
            model_output = model(prev, past_key_values=past)
            past = model_output.past_key_values
            logits = model_output.logits[0, -1, :].to(settings.device)            

    logits, indices = logits.sort(descending=True)
    logits = logits.double()
    indices = indices.int()

    if settings.temp is None:
        settings.temp = 1.0
    logits_temp = logits / settings.temp
    
    # Applying repetition penalty
    # if settings.repetition_penalty is not None and settings.repetition_penalty != 1.0:
    #     assert settings.repetition_penalty > 0, '`repetition_penalty` must be >0!'
    #     for i in range(logits.size(0)):
    #         for token in indices:
    #             if logits[i, token] > 0:
    #                 logits[i, token] /= settings.repetition_penalty
    #             else:
    #                 logits[i, token] *= settings.repetition_penalty

    probs = F.softmax(logits_temp, dim=-1)
        
    # Getting the top-p `probs` and `indices` from the last layer of `logits`
    if not (settings.top_p is None or settings.top_p == 1.0):
        assert settings.top_p > 0 and settings.top_p < 1.0, '`top_p` must be >0 and <=1!'
        cum_probs = probs.cumsum(0)
        k = (cum_probs > settings.top_p).nonzero()[0].item() + 1
        probs = probs[:k]
        indices = indices[:k]
        probs = 1 / cum_probs[k - 1] * probs  # Normalizing
    return probs, indices, past

def is_alpha(s: str) -> bool:
    # A-Za-z
    for i in range(len(s)):
        c = s[i].lower()
        if ord(c) < ord('a') or ord(c) > ord('z'):
            return False
    return True


def check_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
        print('A folder called "{}" is created.'.format(dir))

def generate_binary_message(length: int, filename: str):
    binary_bits = ''.join(random.choice('01') for _ in range(length))
    with open(filename, 'w') as f:
        f.write(binary_bits)
    print(f"message_bits saved to {filename}.")

def read_binary_message(filename: str) -> str:
    with open(filename, 'r') as f:
        binary_bits = f.read()
    return binary_bits

def get_lower_upper_bound(cumulative_probs, v):
    lower_bound = cumulative_probs[v-1] if v > 0 else torch.tensor(0)
    upper_bound = cumulative_probs[v] if v < len(cumulative_probs)-1 else torch.tensor(1)
    SE = [lower_bound.item(), upper_bound.item()]
    return SE

def func_mrn(k_m, n_m, r):
    result = ((k_m / n_m) + r)
    if result >= 1:
        result = result - 1
    return result

def dec2bin(km, lm):
    bin_str = bin(km)[2:]
    return bin_str.zfill(lm)

def get_probs_past(model,
                   prev=None,
                   past=None,
                   device='cuda',
                   top_p=1.0):
    if past is not None:
        past = limit_past(past)
    model_output = model(prev, past_key_values=past)
    past = model_output.past_key_values

    logits = model_output.logits[0,-1,:].to(device)
    logits,indices = logits.sort(descending=True)
    logits = logits.double()
    indices = indices.int()
    probs = F.softmax(logits, dim=-1)

    if 0 < top_p < 1.0:
        cum_probs = probs.cumsum(0)
        k = (cum_probs > top_p).nonzero()[0].item() + 1
        probs = probs[:k]
        indices = indices[:k]
        probs = 1 / cum_probs[k - 1] * probs  # Normalizing
    return probs, indices, past

