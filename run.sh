#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

echo "python main.py --mode='train'"
python main.py --mode='train'

echo "python main.py --mode='test'"
python main.py --mode='test'

echo "python main.py --mode='stego'"
python main.py --mode='stego'

echo "python main.py --mode='eval'"
python main.py --mode='eval'
