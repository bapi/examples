from __future__ import print_function
import argparse
import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import torch.multiprocessing as mpt
from torchvision import datasets, transforms
from resnet_class import ResNet, ResidualBlock, transform, train_dataset, test_dataset, criterion

from train import train, test, modelsave

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=2000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--timemeasure', type=int, default=1, metavar='U',
                        help='Whether Time measure')
parser.add_argument('--usemysgd', type=int, default=1, metavar='U',
                        help='Whether to use custom SGD')
parser.add_argument('--tp', type=int, default=1, metavar='U',
                        help='Whether to use custom SGD')

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    torch.manual_seed(args.seed)

    model = ResNet(ResidualBlock, [2, 2, 2])
    model.share_memory() # gradients are allocated lazily, so they are not shared here
    numproc = args.num_processes
    barrier = torch.zeros([numproc], dtype=torch.int32)
    barrier.share_memory_()
    lock = mp.Lock()
    f = open('LR='+str(args.lr)+'_numproc='+str(args.num_processes)+'_epoch='+str(args.epochs)+'.txt',"w")
        
    start = time.time()
    processes = []
    for rank in range(args.num_processes):
        if args.tp:
            p = mp.Process(target=train, args=(rank, args, model, barrier, lock))
        else:
            p = mp.Process(target=train, args=(rank, args, model, barrier, lock))
        p.start()
        processes.append(p)
    
    # if args.timemeasure == 0:
    if args.tp:
        p = mp.Process(target=modelsave, args=(args, model, barrier))
    else:
        p = mp.Process(target=modelsave, args=(args, model, barrier))
    p.start()
    processes.append(p)
    
    for p in processes:
        p.join()
    train_end = time.time()
    train_time = (train_end - start)
    
    if args.timemeasure == 0:
        # Once training is complete, we can test the model
        torch.manual_seed(args.seed)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=args.batch_size, 
                                              shuffle=True, num_workers=1)
    

        results = torch.zeros(args.epochs,4)
        test(args, model, results, barrier, train_loader)
        f.write("\n\nEpoch\tTestLoss\tTestAccuracy\tTrainLoss\tTrainAccuracy\n\n")
        for i in range(args.epochs):
            f.write('{}\t'.format(i))
            f.write(str('%.6f'%results[i][0].item())+"\t")
            f.write(str('%.2f'%results[i][1].item())+"\t")
            f.write(str('%.6f'%results[i][2].item())+"\t")
            f.write(str('%.2f'%results[i][3].item())+"\n")
    
    print("Training time = " + str(train_time)) 
    f.write("\n\nTraining time = " + str(train_time)) 
    f.close()


