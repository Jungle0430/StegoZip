#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

echo "python main.py --mode train --model_name Qwen/Qwen2.5-3B --dataset ag_news --domain business --reduce_ratio 0.4"
python main.py --mode train --model_name Qwen/Qwen2.5-3B --dataset ag_news --domain business --reduce_ratio 0.4

echo "python main.py --mode test --model_name Qwen/Qwen2.5-3B --dataset ag_news --domain business --reduce_ratio 0.4"
python main.py --mode test --model_name Qwen/Qwen2.5-3B --dataset ag_news --domain business --reduce_ratio 0.4

echo "python main.py --mode eval --model_name Qwen/Qwen2.5-3B --dataset ag_news --domain business --reduce_ratio 0.4"
python main.py --mode eval --model_name Qwen/Qwen2.5-3B --dataset ag_news --domain business --reduce_ratio 0.4
