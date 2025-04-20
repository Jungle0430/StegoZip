import os
import json
import torch
import random
import argparse
import jsonlines
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.deal_data import download_dataset, create_compressed_data
from src.finetune import finetune_model
from src.rank_zip import token2binary, binary2token, huffman_zip
from src.main_discop import discop_stego
from src.restore import restore_message
from src.evaluation import evaluation
from peft import PeftModel

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test", "stego", "eval"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dataset", type=str, default="ag_news")
    parser.add_argument("--domain", type=str, default="business", choices=["world", "sports", "business", "tech", "comment"])
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--reduce_ratio", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_unit_info", type=str2bool, default=True)
    parser.add_argument("--use_lora", type=str2bool, default=True)
    parser.add_argument("--whether_restore", type=str2bool, default=True)
    parser.add_argument("--base_zip", type=str2bool, default=True)

    parser.add_argument("--instruction", type=str,
                        default="You are a text restoration specialist. Your task is to ONLY fill in the missing content within square brackets [] in the input text. Requirements:\n1. Strictly preserve all existing text and punctuation outside brackets.\n2. Maintain original text structure and formatting.")
    parser.add_argument("--end_marker", type=str, default="[END]")
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--stable_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=350)
    parser.add_argument("--save_steps", type=int, default=350)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--prefix", type=str2bool, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    
    parser.add_argument("--stego_algo", type=str, default="Discop", choices=["ADG", "Meteor", "Discop", "SparSamp"])
    parser.add_argument("--stego_dataset", type=str, default="wikitext")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lora_path = f"checkpoint/{args.model_name.split('/')[1]}_{args.dataset}_{args.domain}_{int(args.reduce_ratio*100)}"
    print(f"LoRA model path: {lora_path}")
    
    if args.mode == 'train':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        # <unk> token tokenizer.convert_ids_to_tokens(128244)
        tokenizer.pad_token = "<unk>"
        
        train_settings = {
            "model_name": args.model_name,
            "instruction": args.instruction,
            "end_marker": args.end_marker,
            "val_ratio": args.val_ratio,
            "cutoff_len": args.cutoff_len,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.batch_size // args.micro_batch_size,
            "warmup_ratio": args.warmup_ratio,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules
        }
        ave_info = 0.0
        args.eta = 0.0
        
    elif args.mode == 'test':
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token_id = 128244
        tokenizer.pad_token = "<unk>"
        
        if os.path.exists(lora_path):
            model = PeftModel.from_pretrained(base_model, lora_path)
            model = model.merge_and_unload().eval()
            print(f"LoRA model loaded from {lora_path}")
        else:
            raise FileNotFoundError(f"LoRA model {lora_path} does not exist!")
        
        train_compressed_dataset_path = f"data/compress/{args.model_name.split('/')[1]}_{args.dataset}_{args.domain}_train_{int(args.reduce_ratio*100)}.json"
        if os.path.exists(train_compressed_dataset_path):
            with open(train_compressed_dataset_path, "r") as f:
                train_data = json.load(f)
                ave_info = np.mean([item['self_info'] for item in train_data])
        else:
            raise FileNotFoundError(f"Compressed training data {train_compressed_dataset_path} does not exist!")
        
        test_settings = {
            "model_name": args.model_name,
            "instruction": args.instruction,
            "end_marker": args.end_marker,
            "max_new_token": args.cutoff_len,
            "test_size": args.test_size,
            "temperature": args.temperature,
            "prefix": args.prefix,
            "max_new_tokens": args.max_new_tokens,
            "use_lora": args.use_lora,
            "seed": args.seed
        }
    
    # DSRP
    if args.mode == 'train':
        dataset_path = f"data/origin/{args.dataset}_{args.domain}_{args.mode}.jsonl"
        download_dataset(args.dataset, args.mode, output_file=dataset_path)
    
        compressed_dataset_path = f"data/compress/{args.model_name.split('/')[1]}_{args.dataset}_{args.domain}_{args.mode}_{int(args.reduce_ratio*100)}.json"
        create_compressed_data(model,
                               tokenizer,
                               device,
                               dataset_path,
                               compressed_dataset_path,
                               args.mode,
                               ave_info=ave_info,
                               reduce_ratio=args.reduce_ratio,
                               eta=args.eta,
                               use_unit_info=args.use_unit_info)

    elif args.mode == 'test':
        dataset_path = f"data/origin/{args.dataset}_{args.domain}_{args.mode}.jsonl"
        download_dataset(args.dataset, args.mode, output_file=dataset_path)
        
        compressed_dataset_path = f"data/compress/{args.model_name.split('/')[1]}_{args.dataset}_{args.domain}_{args.mode}_{int(args.reduce_ratio*100)}_{int(args.eta*100)}.json"
        create_compressed_data(model,
                               tokenizer,
                               device,
                               dataset_path,
                               compressed_dataset_path,
                               mode=args.mode,
                               setting=test_settings,
                               ave_info=ave_info,
                               reduce_ratio=args.reduce_ratio,
                               eta=args.eta,
                               use_unit_info=args.use_unit_info)
    
    if args.mode == 'train':
        train_data = load_dataset("json", data_files=compressed_dataset_path)['train']
        
        if args.train_size > 0:
            train_data = train_data.select(range(args.train_size))
        print(f"Load {len(train_data)} training examples from {compressed_dataset_path}")
        
        # Fine-tuning is more stable and avoids over-modification
        indices = np.random.choice(len(train_data), size=int(len(train_data)*args.stable_ratio), replace=False)
        def modify_func(example, idx):
            if idx in indices:
                return {"compressed_text": example["original_text"]}
            return {"compressed_text": example["compressed_text"]}

        train_data = train_data.map(
            lambda example, idx: modify_func(example, idx),
            with_indices=True,
            load_from_cache_file=False
        )
        
        model = model.train()
        finetune_model(model, tokenizer, train_data, lora_path, train_settings)
        
    elif args.mode == 'test':
        results_file = f"result/{args.model_name.split('/')[1]}_{args.dataset}_{args.domain}_{int(args.reduce_ratio*100)}_{int(args.eta*100)}.json"
        if os.path.exists(results_file):
            print(f"Load test data from {results_file}")
            with open(results_file, "r") as f:
                test_dataset = json.load(f)
        else:
            print(f"Load test data from {compressed_dataset_path}")
            with open(compressed_dataset_path, "r") as f:
                test_dataset = json.load(f)
        
        # Restore
        if args.whether_restore:
            test_dataset = restore_message(model, tokenizer, test_dataset, test_settings, results_file)
        
        if not args.use_lora:
            print("token-rank mapping does not use Restorer")
            del model, base_model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, device_map="auto")
            test_dataset = token2binary(model, tokenizer, test_dataset, test_settings, results_file)
            test_dataset = binary2token(model, tokenizer, test_dataset, test_settings, results_file)
        else:
            test_dataset = token2binary(model, tokenizer, test_dataset, test_settings, results_file)
            test_dataset = binary2token(model, tokenizer, test_dataset, test_settings, results_file)
            
        if args.base_zip:
            huffman_zip(test_dataset, results_file)

    elif args.mode == 'stego':
        stego_dataset_path = f"data/origin/{args.stego_dataset}_train.jsonl"
        download_dataset(args.stego_dataset, 'train', output_file=stego_dataset_path)
        with jsonlines.open(stego_dataset_path) as reader:
            stego_dataset = [item for item in reader]
        
        results_file = f"result/{args.model_name.split('/')[1]}_{args.dataset}_{args.domain}_{int(args.reduce_ratio*100)}_{int(args.eta*100)}.json"
        with open(results_file, "r") as f:
            test_dataset = json.load(f)
            
        random_indices = random.sample(range(len(stego_dataset)), len(test_dataset))
        selected_texts = [stego_dataset[i]['content'] for i in random_indices]
        
        stego_prompt = []
        for text in selected_texts:
            sentences = text.split('. ')
            length = len(sentences)
            stego_prompt.append('. '.join(sentences[:2]) if length > 2 else text)
        
        stego_file = f"result/{args.model_name.split('/')[1]}_{args.dataset}_{args.domain}_{int(args.reduce_ratio*100)}_{int(args.eta*100)}_stego.json"
        if args.stego_algo == 'Discop':
            discop_stego(test_dataset, stego_prompt, args.seed, stego_file)
        else:
            print("As a proven master of steganography, I'm sure you can make it on your own! (๑•̀ㅂ•́)و✧")
            raise ValueError(f"Invalid stego algorithm: {args.stego_algo}")     
        
    elif args.mode == 'eval':
        stego_file = f"result/{args.model_name.split('/')[1]}_{args.dataset}_{args.domain}_{int(args.reduce_ratio*100)}_{int(args.eta*100)}_stego.json"
        eval_file = f"result/{args.model_name.split('/')[1]}_{args.dataset}_{args.domain}_{int(args.reduce_ratio*100)}_{int(args.eta*100)}_eval.json"
        evaluation(stego_file, eval_file)
        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
        
if __name__ == "__main__":
    main()