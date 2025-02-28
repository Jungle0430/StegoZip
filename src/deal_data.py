import os
import time
import json
import jsonlines
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

def download_dataset(dataset_name, split, output_file):
    dataset_name = dataset_name.lower()
    
    if os.path.exists(output_file):
        print("Data already exists in " + output_file)
        return
    
    if dataset_name == 'ag_news':
        raw_dataset = load_dataset(dataset_name)
        combined_dataset = concatenate_datasets([raw_dataset[split]])
        # 2 is Business class id.
        filter_dataset = combined_dataset.filter(lambda example: example['label'] == 2)
        dataset = [data['text'] for data in filter_dataset]
    elif dataset_name == 'imdb':
        raw_dataset = load_dataset(dataset_name, split=split)
        dataset = [data['text'].replace('<br /><br />', ' ').replace('<br />', ' ') for data in raw_dataset]
    elif dataset_name == 'wikitext':
        raw_dataset = load_dataset("Salesforce/wikitext",
                               "wikitext-2-v1",
                               split=split,
                               ignore_verifications=True)
        filter_dataset = raw_dataset.filter(lambda example: len(example['text']) > 100)
        dataset = [data['text'] for data in filter_dataset]
    else:
        raise ValueError("Invalid dataset name!")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for idx, example in enumerate(dataset):
            record = {
                "id": idx,
                "content": example,
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    print("Successfully saved data to " + output_file)
    
def create_compressed_data(model, tokenizer, device, input_file, output_file, mode='train', ave_info=0, reduce_ratio=0.3, eta=1.0, use_unit_info=True):
    if mode not in ['train', 'test']:
        raise ValueError("Invalid mode! Must be either 'train' or 'test'.")
    
    print(f"Average self-information of training data: {ave_info}")
        
    if os.path.exists(input_file) is False:
        raise FileNotFoundError(f"Input file {input_file} does not exist!")
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists, will not regenerate again.")
        return
    
    from src.semantic_compressor import SemanticCompressor
    compressor = SemanticCompressor(model, tokenizer, device, ave_info=ave_info, eta=eta)
    results = []
    
    with jsonlines.open(input_file) as reader:
        for item in tqdm(reader):
            # Perform text compression
            start_time = time.time()
            compressed_text, masked_words, self_info = compressor.compress_text(
                item['content'], 
                reduce_ratio=reduce_ratio,
                use_unit_info=use_unit_info
            )
            end_time = time.time()
            
            # compressed_tokens, masked_tokens = compressor.compress_text(
            #     item['content'],
            #     reduce_ratio=reduce_ratio
            # )
            
            # if isinstance(masked_tokens, np.ndarray):
            #     masked_tokens = masked_tokens.tolist()
            # if isinstance(compressed_tokens, np.ndarray):
            #     compressed_tokens = compressed_tokens.tolist()
                
            # Write the enhanced data
            sample = {
                "id": item['id'],
                "original_text": item['content'],
                "compressed_text": compressed_text,
                "compress_time_cost": round(end_time - start_time, 6),
                # "compressed_tokens": compressed_tokens,
                "masked_words": masked_words,
                "self_info": self_info,
                # "masked_tokens": masked_tokens
            }
            results.append(sample)
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)  
    
    print(f"Processing complete, compressed data has been saved to {output_file}")