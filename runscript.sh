#!/bin/bash
cd mnist_hogwild6/
mkdir -p mnist_hogwild6_time_measure
cd mnist_hogwild6_time_measure
mkdir -p saved_models
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 1 2>log1 #| tee res0.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 10 2>log4 #| tee res3.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 20 2>log5 #| tee res4.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 40 2>log6 #| tee res5.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 79 2>log6 #| tee res5.txt
cd ..
mkdir -p mnist_hogwild6_converge_measure
cd mnist_hogwild6_converge_measure
mkdir -p saved_models
python ../main.py --timemeasure 0 --usemysgd 0 --tp 0 --lr 0.015 --epochs 50 --test-batch-size 1000 --num-processes 79 2>log6 #| tee res5.txt
cd ../.. 
cd mnist_hogwild4/
mkdir -p mnist_hogwild4_time_measure
cd mnist_hogwild4_time_measure
mkdir -p saved_models
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 1 2>log1 #| tee res0.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 10 2>log4 #| tee res3.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 20 2>log5 #| tee res4.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 40 2>log6 #| tee res5.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 79 2>log6 #| tee res5.txt
cd ..
mkdir -p mnist_hogwild4_converge_measure
cd mnist_hogwild4_converge_measure
mkdir -p saved_models
python ../main.py --timemeasure 0 --usemysgd 0 --tp 0 --lr 0.015 --epochs 50 --test-batch-size 1000 --num-processes 79 2>log6 #| tee res5.txt
cd ../.. 
cd mnist_hogwild5/
mkdir -p mnist_hogwild5_time_measure
cd mnist_hogwild5_time_measure
mkdir -p saved_models
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 1 2>log1 #| tee res0.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 10 2>log4 #| tee res3.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 20 2>log5 #| tee res4.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 40 2>log6 #| tee res5.txt
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 --num-processes 79 2>log6 #| tee res5.txt
cd ..
mkdir -p mnist_hogwild5_converge_measure
cd mnist_hogwild5_converge_measure
mkdir -p saved_models
python ../main.py --timemeasure 0 --usemysgd 0 --tp 0 --lr 0.015 --epochs 50 --test-batch-size 1000 --num-processes 79 2>log6 #| tee res5.txt
cd ../.. 
cd mnist1/
mkdir -p mnist1_time_measure
cd mnist1_time_measure
mkdir -p saved_models
python ../main.py --timemeasure 1 --usemysgd 1 --tp 1 --lr 0.015 --epochs 1 --test-batch-size 1000 2>log1
cd ..
mkdir -p mnist1_converge_measure
cd mnist1_converge_measure
mkdir -p saved_models
python ../main.py --timemeasure 0 --usemysgd 0 --tp 0 --lr 0.015 --epochs 50 --test-batch-size 1000 2>log6 #| tee res5.txt
cd ../.. 
exit 1
