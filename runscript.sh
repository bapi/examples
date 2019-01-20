#!/bin/bash
cd mnist1/
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 2>log1
cd ../mnist_hogwild1/
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 1 2>log1
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 2 2>log2
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 5 2>log3
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 10 2>log4
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 20 2>log5
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 40 2>log6
cd ../mnist_hogwild2/
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 1 2>log1
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 2 2>log2
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 5 2>log3
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 10 2>log4
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 20 2>log5
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 40 2>log6
cd ../mnist_hogwild3/
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 1 2>log1
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 2 2>log2
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 5 2>log3
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 10 2>log4
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 20 2>log5
python main.py --lr 0.1 --epochs 50 --usemysgd 1 --test-batch-size 2000 --num-processes 40 2>log6
cd ..
exit 1