from __future__ import print_function
import argparse
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.multiprocessing as mp
from train import train, test
from torch.multiprocessing import Process, Value, Lock, Queue

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')

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
    f = open('stochastic_minibatch_coordinate_descent'+'_batch_size='+str(args.batch_size)+'_num_proc='+str(args.num_processes)+'.txt',"w")
    print('Stochastic Mini-batch co-ordinate descent: Batch-size = {}, Num-processes = {}'.format(args.batch_size, args.num_processes))
    f.write('Stochastic Mini-batch co-ordinate descent: Batch-size = {}, Num-processes = {}\n\n'.format(args.batch_size, args.num_processes))
    
    torch.manual_seed(args.seed)

    model = Net()
    model.share_memory() # gradients are allocated lazily, so they are not shared here
    result = torch.zeros(args.epochs, args.num_processes)
    result.share_memory_()

    learning_rates = torch.zeros(args.epochs)
    learning_rates.share_memory_()
    
    # m = torch.mean(result, 1, True)
    # print(m)
    # p = pool.ThreadPool(args.num_processes)
    # p.starmap_async(fill, [q, result])
    # # p.starmap_async(train, [q, args, model, result])
    # p.close()
    # p.join()
    start = time.time()
    processes = []
    for rank in range(args.num_processes):
        p = Process(target=train, args=(rank, args, model, result, learning_rates))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    train_end = time.time()
    # # train(7, args, model)
    # train(args,model)
    # Once training is complete, we can test the model
    f.write("(Ep,Prc):\t")
    for j in range(args.num_processes):
      f.write('{}\t'.format(j))
    f.write("LR\t")
    
    f.write('\n')  
    for i in range(args.epochs):
      f.write('{}\t'.format(i))
      for j in range(args.num_processes):
        f.write(str('%.6f'%result[i][j].item())+"\t")
      # f.write('{:.6f}'.format(learning_rates[i].item()))
      f.write(str('%.6f'%learning_rates[i].item()))
      f.write("\n")

    test(args, model)
    test_end = time.time()
    train_time = (train_end - start)
    test_time = (test_end - train_end)
    print("Training time = " + str(train_time) + " and Testing time = " + str(test_time)) 
    f.write("Training time = " + str(train_time) + " and Testing time = " + str(test_time)) 
    f.close()
