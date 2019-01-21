from __future__ import print_function
import argparse
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets, transforms

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
    numproc = args.num_processes
    barrier = torch.zeros([numproc], dtype=torch.int32)
    barrier.share_memory_()
    lock = mp.Lock()
    f = open('LR='+str(args.lr)+'_numproc='+str(args.num_processes)+'.txt',"w")
    
    start = time.time()
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, barrier, lock))
        p.start()
        processes.append(p)
    p = mp.Process(target=modelsave, args=(args, model, barrier))
    p.start()
    processes.append(p)
    
    for p in processes:
        p.join()
    train_end = time.time()
    train_time = (train_end - start)
    
    # Once training is complete, we can test the model
    torch.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.test_batch_size, shuffle=True, num_workers=1)

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


