from __future__ import print_function
import argparse
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process, Value, Lock, Queue

from train import train, test

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=7, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--usemysgd', type=int, default=0, metavar='U',
                        help='Whether to use custom SGD')
parser.add_argument('--lra', type=bool, default=True, metavar='LR',
                        help='Whether to use adaptable learning rate')

class Counter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value
def func(counter):
    for i in range(50):
        time.sleep(0.01)
        counter.increment()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = Net()
    model.share_memory() # gradients are allocated lazily, so they are not shared here
    plength = 0
    for p in model.parameters():
      plength+=p.numel()
    chunk_size = int(plength/args.num_processes)
      
    val = Value('i', 0)
    lock = Lock()
    results = torch.zeros(args.epochs,3)
    results.share_memory_()
    
    if args.usemysgd:
      f = open('hogwild_SCD'+'_batch_size='+str(args.batch_size)+'_numproc='+str(args.num_processes)+'_usebackprop=True.txt',"w")
    else:
      f = open('hoigwild_SCD'+'_batch_size='+str(args.batch_size)+'_numproc='+str(args.num_processes)+'_usebackprop=False.txt',"w")

    print('Stochastic Gradient descent: Batch-size = {}'.format(args.batch_size))
    f.write('Stochastic Gradient descent: Batch-size = {}'.format(args.batch_size))
    f.write("\n\nEpoch\tLR\tLoss\tAccuracy\n\n")
    start = time.time()
    processes = []
    for rank in range(args.num_processes):
        p = Process(target=train, args=(rank, args, model, plength, chunk_size, results, val, lock))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    
    p = Process(target=test, args=(args, model, results, val, lock))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()
    # train(0, args, model, plength, chunk_size, results, val, lock)
    train_end = time.time()
    train_time = (train_end - start)
    
    for i in range(args.epochs):
      f.write('{}\t'.format(i))
      f.write(str('%.6f'%results[i][0].item())+"\t")
      f.write(str('%.6f'%results[i][1].item())+"\t")
      f.write(str('%.6f'%results[i][2].item())+"\n")
      

    print("Training time = " + str(train_time)) 
    f.write("\n\nTraining time = " + str(train_time)) 
    f.close()
