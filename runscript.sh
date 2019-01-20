#!/bin/bash
# cd mnist1/
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 2>&1 | tee log1
cd ../mnist_hogwild1/
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 1 2>&1 | tee log1
python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 2 2>&1 | tee log2
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 5 2>&1 | tee log3
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 10 2>&1 | tee log4
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 20 2>&1 | tee log5
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 40 2>&1 | tee log6
cd ../mnist_hogwild2/
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 1 2>&1 | tee log1
python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 2 2>&1 | tee log2
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 5 2>&1 | tee log3
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 10 2>&1 | tee log4
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 20 2>&1 | tee log5
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 40 2>&1 | tee log6
# cd ../mnist_hogwild3/
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 1 2>&1 | tee log1
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 2 2>&1 | tee log2
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 5 2>&1 | tee log3
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 10 2>&1 | tee log4
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 20 2>&1 | tee log5
# python main.py --lr 0.1 --epochs 80 --usemysgd 1 --test-batch-size 2000 --num-processes 40 2>&1 | tee log6
cd ..
exit 1