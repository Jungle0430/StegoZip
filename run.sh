#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

echo "python main.py --mode 'train'"
python main.py --mode 'train' --reduce_ratio 0.3

echo "python main.py --mode 'test'"
python main.py --mode 'test' --reduce_ratio 0.3

echo "python main.py --mode 'stego'"
python main.py --mode 'stego' --reduce_ratio 0.3

echo "python main.py --mode 'eval'"
python main.py --mode 'eval' --reduce_ratio 0.3
