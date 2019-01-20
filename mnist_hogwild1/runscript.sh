#!/bin/bash
~/anaconda3/bin/python3.7 main.py --lr 0.1 --epochs 100 --usemysgd 1 --test-batch-size 2000 --num-processes 8 2>&1 | tee log1
~/anaconda3/bin/python3.7 main.py --lr 0.1 --epochs 100 --usemysgd 1 --test-batch-size 2000 --num-processes 18 2>&1 | tee log2
~/anaconda3/bin/python3.7 main.py --lr 0.1 --epochs 100 --usemysgd 1 --test-batch-size 2000 --num-processes 28 2>&1 | tee log3
~/anaconda3/bin/python3.7 main.py --lr 0.1 --epochs 100 --usemysgd 1 --test-batch-size 2000 --num-processes 38 2>&1 | tee log4
~/anaconda3/bin/python3.7 main.py --lr 0.1 --epochs 100 --usemysgd 1 --test-batch-size 2000 --num-processes 48 2>&1 | tee log5
~/anaconda3/bin/python3.7 main.py --lr 0.1 --epochs 100 --usemysgd 1 --test-batch-size 2000 --num-processes 58 2>&1 | tee log6
~/anaconda3/bin/python3.7 main.py --lr 0.1 --epochs 100 --usemysgd 1 --test-batch-size 2000 --num-processes 68 2>&1 | tee log7
~/anaconda3/bin/python3.7 main.py --lr 0.1 --epochs 100 --usemysgd 1 --test-batch-size 2000 --num-processes 78 2>&1 | tee log8