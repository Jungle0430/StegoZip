import os
import json
import torch
import random
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.deal_data import download_dataset, create_compressed_data
from src.finetune import finetune_model
from src.rank_zip import token2binary, binary2token
from src.main_discop import discop_stego
from src.restore import restore_message
from src.evaluation import evaluation
from peft import prepare_model_for_int8_training, PeftModel

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B", choices=["Qwen/Qwen2.5-7B", "lmsys/vicuna-7b-v1.5"])
    parser.add_argument("--dataset", type=str, default="ag_news", choices=["ag_news", "imdb"])
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test", "stego", "eval"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--reduce_ratio", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_unit_info", type=bool, default=True)

    parser.add_argument("--instruction", type=str,
                        default="You are a text restoration expert. Insert missing words indicated by [LACK] into the input text to reconstruct the original without deleting or modifying existing content.")
    parser.add_argument("--end_marker", type=str, default="[END]")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=str, default=2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=list, default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--prefix", type=bool, default=True)
    parser.add_argument("--win_size", type=int, default=1024)
    
    parser.add_argument("--stego_algo", type=str, default="Discop", choices=["Discop", "Meteor"])
    parser.add_argument("--stego_dataset", type=str, default="wikitext")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lora_path = f"checkpoint/{args.model_name.split('/')[0]}_{args.dataset}_{int(args.reduce_ratio*10)}_{int(args.eta*10)}"
    print(f"LoRA model path: {lora_path}")
    
    if args.mode == 'train':
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = prepare_model_for_int8_training(base_model)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
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
    elif args.mode == 'test':
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        if os.path.exists(lora_path):
            model = PeftModel.from_pretrained(base_model, lora_path)
            model = model.merge_and_unload().eval()
            print(f"LoRA model loaded from {lora_path}")
        else:
            raise FileNotFoundError(f"LoRA model {lora_path} does not exist!")
        
        train_compressed_dataset_path = f"data/compress/{args.model_name.split('/')[0]}_{args.dataset}_train_{int(args.reduce_ratio*10)}_{int(args.eta*10)}.json"
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
            "win_size": args.win_size
        }
    
    # DSRP
    if args.mode == 'train' or args.mode == 'test':
        dataset_path = f"data/origin/{args.dataset}_{args.mode}.jsonl"
        download_dataset(args.dataset, args.mode, output_file=dataset_path)
    
        compressed_dataset_path = f"data/compress/{args.model_name.split('/')[0]}_{args.dataset}_{args.mode}_{int(args.reduce_ratio*10)}_{int(args.eta*10)}.json"
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
    
    if args.mode == 'train':
        train_data = load_dataset("json", data_files=compressed_dataset_path)['train']
        print(f"Load {len(train_data)} training examples from {compressed_dataset_path}")
        finetune_model(base_model, tokenizer, train_data, lora_path, train_settings)
        
    elif args.mode == 'test':
        with open(compressed_dataset_path, "r") as f:
            test_dataset = json.load(f)
        
        results_file = f"result/{args.model_name.split('/')[0]}_{args.dataset}_{int(args.reduce_ratio*10)}_{int(args.eta*10)}.json"
        
        # ICC
        token2binary(model, tokenizer, test_dataset, test_settings, results_file)
        # De-ICC
        binary2token(model, tokenizer, test_dataset, test_settings, results_file)
        # Restore
        restore_message(model, tokenizer, test_dataset, test_settings, results_file)
        
    elif args.mode == 'stego':
        stego_dataset_path = f"data/origin/{args.stego_dataset}_train.jsonl"
        download_dataset(args.dataset, 'train', output_file=dataset_path)
        with open(stego_dataset_path, "r") as f:
            stego_dataset = json.load(f) 
        
        results_file = f"result/{args.model_name.split('/')[0]}_{args.dataset}_{int(args.reduce_ratio*10)}_{int(args.eta*10)}.json"
        with open(results_file, "r") as f:
            test_dataset = json.load(f)
            
        random_indices = random.sample(range(len(stego_dataset)), len(test_dataset))
        selected_texts = [stego_dataset[i]['content'] for i in random_indices]
        
        stego_prompt = []
        for text in selected_texts:
            sentences = text.split('. ')
            length = len(sentences)
            stego_prompt.append('. '.join(sentences[:2]) if length > 2 else text)
        
        stego_file = f"result/{args.model_name.split('/')[0]}_{args.dataset}_{int(args.reduce_ratio*10)}_{int(args.eta*10)}_stego.json"
        if args.stego_algo == 'Discop':
            discop_stego(test_dataset, stego_prompt, args.seed, stego_file)
        else:
            print("As a proven master of steganography, I'm sure you can make it on your own! (๑•̀ㅂ•́)و✧")
            raise ValueError(f"Invalid stego algorithm: {args.stego_algo}")     
        
    elif args.mode == 'eval':
        stego_file = f"result/{args.model_name.split('/')[0]}_{args.dataset}_{int(args.reduce_ratio*10)}_{int(args.eta*10)}_stego.json"
        eval_file = f"result/{args.model_name.split('/')[0]}_{args.dataset}_{int(args.reduce_ratio*10)}_{int(args.eta*10)}_eval.json"
        evaluation(stego_file, eval_file)
        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
        
if __name__ == "__main__":
    main()