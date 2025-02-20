import os
import time
import hmac
import torch
import torch.functional as F
from scipy.stats import entropy
import hashlib
import numpy as np

from config import Settings, text_default_settings_meteor
from utils import get_probs_indices_past, SingleExampleOutput, set_seed

def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2**i)
    return res

def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eps = 2**-32

class DRBG(object):
    def __init__(self, key, seed):
        self.key = key
        self.val = b'\x01' * 64
        self.reseed(seed)

        self.byte_index = 0
        self.bit_index = 0

    def hmac(self, key, val):
        return hmac.new(key, val, hashlib.sha512).digest()

    def reseed(self, data=b''):
        self.key = self.hmac(self.key, self.val + b'\x00' + data)
        self.val = self.hmac(self.key, self.val)

        if data:
            self.key = self.hmac(self.key, self.val + b'\x01' + data)
            self.val = self.hmac(self.key, self.val)

    def generate_bits(self, n):
        xs = np.zeros(n, dtype=bool)
        for i in range(0, n):
            xs[i] = (self.val[self.byte_index] >> (7 - self.bit_index)) & 1

            self.bit_index += 1
            if self.bit_index >= 8:
                self.bit_index = 0
                self.byte_index += 1

            if self.byte_index >= 8:
                self.byte_index = 0
                self.val = self.hmac(self.key, self.val)

        self.reseed()
        return xs


def bin_sort(l, token_indices, total, entropy, device):
    #compute entropy for upper bound on the number of bins we need

    bucket_size = total
    num_bins = 2**int(entropy + 1)
    bucket_size = total / num_bins

    bins = [torch.empty(0, dtype=torch.long, device=device)] * num_bins
    value_in_bins = [0] * num_bins
    space_left_after = [total - i * bucket_size for i in range(0, num_bins)]

    token_bins = [torch.empty(0, dtype=torch.long, device=device)] * num_bins

    # Figuring out what the search order should be
    step_size = num_bins / 4
    search_order = []
    priorities = [0] * num_bins
    priority = 0
    search_order.append(int(num_bins / 2))
    search_order.append(0)
    priorities[int(num_bins / 2)] = 0
    priorities[0] = 0
    while (step_size >= 1):
        priority += 1
        for x in range(num_bins - int(step_size), -1, -int(step_size * 2)):
            search_order.append(x)
            priorities[x] = priority
        step_size = step_size / 2

    # Adding the actual elements
    for (item, token_index) in zip(l.tolist(), token_indices.tolist()):
        found_single_bucket_fit = False
        single_bucket_index = -1
        single_bucket_value = bucket_size

        found_multi_bucket_bumpless_fit = False
        multi_bucket_bumpless_index = -1
        multi_bucket_bumpless_value = total

        found_multi_bucket_bumping_fit = False
        multi_bucket_bumping_index = -1
        multi_bucket_bumping_value = total

        for i in search_order:  # for index in search_order
            if (item > space_left_after[i]):
                continue
            if (value_in_bins[i] >= bucket_size):
                continue

            # Priority of choices
            #  1. Can i place this thing in an empty bucket all on its own?
            #  2. Can i plan this somewhere where is doesnt have to bump anything else around?
            #    2a. Minimize the wasted space.  Aka use the smallest space (of equal priority) that accomplishes this goal
            #  3. If not (1) and (2), then put it in the space the bumps stuff the least.

            if (value_in_bins[i] + item > bucket_size):  #Would overflow.

                space_before_next_block = bucket_size - value_in_bins[i]
                for j in range(i + 1, len(bins)):
                    if (value_in_bins[j] >
                            0):  # We have found a bucket with something in it.  This is how much space we have here.
                        space_before_next_block = space_before_next_block + (bucket_size - value_in_bins[i])
                        break
                    else:  # This was a empty bucket
                        space_before_next_block = space_before_next_block + bucket_size

                if ((not found_multi_bucket_bumpless_fit)
                        or (found_multi_bucket_bumpless_fit
                            and priorities[i] <= priorities[multi_bucket_bumpless_index])):  #This could potentially be a match

                    # If this is a valid space to put this without bumping and it is a better fit than previous spaces
                    if (space_before_next_block > item and space_before_next_block < multi_bucket_bumpless_value):
                        # set this to be the pointer!  we can fit stuff here
                        found_multi_bucket_bumpless_fit = True
                        multi_bucket_bumpless_index = i
                        multi_bucket_bumpless_value = space_before_next_block

                    # Find the overflow that will bump the least
                    if (item - space_before_next_block < multi_bucket_bumping_value):
                        found_multi_bucket_bumping_fit = True
                        multi_bucket_bumping_index = i
                        multi_bucket_bumping_value = item - space_before_next_block

            if (value_in_bins[i] + item <= bucket_size):  #Would fit
                if (single_bucket_value > value_in_bins[i]):
                    found_single_bucket_fit = True
                    single_bucket_value = value_in_bins[i]
                    single_bucket_index = i

        if (single_bucket_index == multi_bucket_bumpless_index == multi_bucket_bumping_index == -1):
            bins[0] = torch.cat((torch.tensor([item], device=device), bins[0]), 0)
            token_bins[0] = torch.cat((torch.tensor([token_index], device=device), token_bins[0]), 0)
            continue

        if found_single_bucket_fit:
            # We found somewhere we can actually fit!
            bins[single_bucket_index] = torch.cat((bins[single_bucket_index], torch.tensor([item], device=device)), 0)
            token_bins[single_bucket_index] = torch.cat(
                (token_bins[single_bucket_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[single_bucket_index] += item
            for i in range(0, single_bucket_index + 1):
                space_left_after[i] -= item

        elif found_multi_bucket_bumpless_fit:
            # Found somewhere we can put this without upsetting the force
            part_in_bucket = bucket_size - value_in_bins[multi_bucket_bumpless_index]
            part_overflow = item - part_in_bucket
            bins[multi_bucket_bumpless_index] = torch.cat(
                (bins[multi_bucket_bumpless_index], torch.tensor([item], device=device)), 0)
            token_bins[multi_bucket_bumpless_index] = torch.cat(
                (token_bins[multi_bucket_bumpless_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[multi_bucket_bumpless_index] = bucket_size

            # Fill this bucket and continue overflowing
            j = multi_bucket_bumpless_index + 1
            for i in range(0, j):
                space_left_after[i] -= item

            while (part_overflow > 0):
                new_part_overflow = (value_in_bins[j] + part_overflow) - bucket_size
                value_in_bins[j] = min(bucket_size, part_overflow + value_in_bins[j])  # mark the bucket as filled
                space_left_after[j] -= part_overflow
                part_overflow = new_part_overflow
                j += 1

        else:
            part_in_bucket = bucket_size - value_in_bins[multi_bucket_bumping_index]
            part_overflow = item - part_in_bucket
            bins[multi_bucket_bumping_index] = torch.cat((bins[multi_bucket_bumping_index], torch.tensor([item], device=device)),
                                                         0)
            token_bins[multi_bucket_bumping_index] = torch.cat(
                (token_bins[multi_bucket_bumping_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[multi_bucket_bumping_index] = bucket_size

            # Fill this bucket and continue overflowing
            j = multi_bucket_bumping_index + 1
            for i in range(0, j):
                space_left_after[i] -= item
            while (part_overflow > 0):
                new_part_overflow = (value_in_bins[j] + part_overflow) - bucket_size
                value_in_bins[j] = min(bucket_size, part_overflow + value_in_bins[j])  # mark the bucket as filled
                space_left_after[j] -= part_overflow
                part_overflow = new_part_overflow
                j += 1

    sorted_tensor = torch.cat(bins, 0)
    sorted_tokens = torch.cat(token_bins, 0)

    return sorted_tensor, sorted_tokens


# Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
sample_key = b'0x01' * 64
sample_seed_prefix = b'sample'
sample_nonce_counter = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

mask_generator = DRBG(sample_key, sample_seed_prefix + sample_nonce_counter)
mask_bits = mask_generator.generate_bits(10000) 

@torch.no_grad()
def encode_meteor(model,
                  tokenizer,
                  context: str,
                  message: str,
                  settings: Settings = text_default_settings_meteor,
                  verbose: bool = False,
                  precision: int = 32,
                  is_sort: bool = True,
                  randomize_key: bool = False,
                  input_key: bytes = sample_key,
                  input_nonce: bytes = sample_nonce_counter):
    message_length = len(message)
    message += '0' * 500
    
    if randomize_key:
        input_key = os.urandom(64)

    algo, temp, top_p, length, seed = settings()
    message = list(int(x) for x in message)

    start = time.time()
    set_seed(seed)
    context = tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(device)

    past = None
    prev = context
    generated_ids = None

    max_val = 2**precision
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    total_num = 0
    total_log_probs = 0
    total_kld = 0  # in bits
    total_entropy = 0
    max_kld = 0
    total_minimum_entropy = 0
    
    eos_tokens = [".", "!", "?", '."', "...", ";", ")", "]", "}", "<s>", "</s>"]
    eos_tokens_id = [tokenizer.convert_tokens_to_ids(token) for token in eos_tokens]

    i = 0  # index of message_bits
    t = 0  # index of token
    finish_flag = False
    while t < length:
        probs_temp, indices, past = get_probs_indices_past(model, prev, past, settings, temp)

        # Cutoff low probabilities that would be rounded to 0
        cur_int_range = cur_interval[1] - cur_interval[0]
        cur_threshold = 1 / cur_int_range
        original_indices = indices.clone()

        if (probs_temp < cur_threshold).nonzero().numel() > 0:
            k = max(2, (probs_temp < cur_threshold).nonzero()[0].item())  # not less than 2
            probs_temp_int = probs_temp[:k]  # Cutoff all but top k
            indices = indices[:k]
        else:
            probs_temp_int = probs_temp.clone()
        
        # Rescale to correct range
        probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
        entropy_in_this_distribution = entropy(probs_temp.tolist(), base=2)
        probs_temp_int = probs_temp_int.round().long()

        if is_sort:
            probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range, entropy_in_this_distribution, device)
        cum_probs = probs_temp_int.cumsum(0)

        # Remove any elements from the bottom if rounding caused the total prob to be too large
        overfill_index = (cum_probs > cur_int_range).nonzero()
        if len(overfill_index) > 0:
            cum_probs = cum_probs[:overfill_index[0]]

        # Add any mass to the top if removing/rounding causes the total prob to be too small
        cum_probs += cur_int_range - cum_probs[-1]  # add

        # Get out resulting probabilities
        probs_final = cum_probs.clone()
        probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

        # Convert to position in range
        cum_probs += cur_interval[0]

        # Apply the mask to the message
        message_bits = message[i:i + precision]

        if i + precision > len(message):
            print('Finish Embedding, padding 0')
            message_bits = message_bits + [0] * (i + precision - len(message))
            break

        for b in range(0, len(message_bits)):
            message_bits[b] = message_bits[b] ^ mask_bits[i + b]

        # Get selected index based on binary fraction from message bits
        message_idx = bits2int(reversed(message_bits))
        selection = (cum_probs > message_idx).nonzero()[0].item()
        sampled_index = indices[selection].item()

        # Calculate new range as ints
        new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
        new_int_top = cum_probs[selection]

        # Convert range to bits
        new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
        new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

        # Consume most significant bits which are now fixed and update interval
        num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
        i += num_bits_encoded

        # Gather statistics
        index_in_original_distribution = (original_indices == sampled_index).nonzero()[0].item()
        total_log_probs += torch.log2(probs_temp[index_in_original_distribution]).item()

        actual_distribution = probs_final.double() / probs_final.sum()

        actual_distribution_sorted, _ = actual_distribution.sort(descending=True)

        kld_step = entropy(probs_temp.tolist(),
                           actual_distribution_sorted.tolist() + [eps] *
                           (len(probs_temp.tolist()) - len(actual_distribution.tolist())),
                           base=2)
        total_kld += kld_step

        if kld_step > max_kld:
            max_kld = kld_step
            
        total_entropy += entropy(actual_distribution.tolist(), base=2)

        if generated_ids is None:
            generated_ids = [sampled_index]
        else:
            generated_ids.append(sampled_index)

        # Update history with new token
        prev = torch.tensor([sampled_index], device=device).unsqueeze(0)
        t += 1
        
        if i > message_length:
            finish_flag = True
        
        if finish_flag and sampled_index in eos_tokens_id:
            break

    end = time.time()

    ave_kld = total_kld / t
    perplexity = 2**(-1 / length * total_log_probs)
    embedding_efficiency = i / total_entropy
    stego_object = tokenizer.decode(generated_ids)
    # settings.length = t

    if verbose:
        print(generated_ids)
        print('embeding_rate = {:.2f}bpw'.format(i / t))
        print('total_entropy = {:.2f}'.format(total_entropy))
        print('embedding_efficiency = {:.3f}'.format(embedding_efficiency))
        print('perplexity = {:.3f}'.format(perplexity))
    
    
    return SingleExampleOutput(generated_ids, stego_object, 
                               i, total_entropy, 
                               ave_kld, max_kld, 
                               perplexity, end - start, 
                               settings, total_minimum_entropy)

def decode_meteor(model,
                  tokenizer,
                  context,
                  stego,
                  settings: Settings = text_default_settings_meteor,
                  verbose: bool = False,
                  precision: int = 32,
                  is_sort: bool = True,
                  randomize_key: bool = False,
                  input_key: bytes = sample_key,
                  input_nonce: bytes = sample_nonce_counter):
    stego_length = len(stego)

    algo, temp, top_p, length, seed = settings()
    set_seed(seed)
    context = tokenizer(context, return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(device)
    
    if randomize_key:
        input_key = os.urandom(64)

    max_val = 2**precision
    cur_interval = [0, max_val]

    prev = context
    past = None
    message_decoded = '' 

    with torch.no_grad():
        t = 0 
        i = 0  
        while t < stego_length:
            probs_temp, indices, past = get_probs_indices_past(model, prev, past, settings, temp)

            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range

            if (probs_temp < cur_threshold).nonzero().numel() > 0:
                k = max(2, (probs_temp < cur_threshold).nonzero()[0].item()) 
                probs_temp_int = probs_temp[:k]
                indices = indices[:k]
            else:
                probs_temp_int = probs_temp.clone()

            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            entropy_in_this_distribution = entropy(probs_temp.tolist(), base=2)
            probs_temp_int = probs_temp_int.round().long()

            if is_sort:
                probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range, entropy_in_this_distribution, device)
            cum_probs = probs_temp_int.cumsum(0)
            
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]

            cum_probs += cur_int_range - cum_probs[-1]
            cum_probs += cur_interval[0]

            sampled_index = stego[t]
            selection_true = (indices == sampled_index).nonzero()
            if selection_true.numel() == 0:
                prev = torch.tensor([sampled_index], device=device).unsqueeze(0)
                t += 1
                continue
            
            selection = selection_true[0].item()
            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            
            if selection >= len(cum_probs):
                print("Fail to decode!")
                break
            
            new_int_top = cum_probs[selection]

            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, precision)))  

            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            extracted_bits = new_int_bottom_bits_inc[:num_bits_encoded]

            for b in range(num_bits_encoded):
                extracted_bits[b] = extracted_bits[b] ^ mask_bits[i + b]

            message_decoded += ''.join(str(x) for x in extracted_bits)
            i += num_bits_encoded
            
            prev = torch.tensor([sampled_index], device=device).unsqueeze(0)
            t += 1
            
            if len(message_decoded) > 4 and message_decoded[:4] != "0000":
                message_decoded = ''
                print("Fail to decode!")
                break

    return message_decoded
