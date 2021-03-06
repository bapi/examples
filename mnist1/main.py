from __future__ import print_function
import argparse
import multiprocessing as mp
import torch.multiprocessing as mpt
import os
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torchvision import datasets, transforms
# from torch.multiprocessing import Process, Value, Lock, Queue
from mysgd import StochasticGD
from myscheduler import MyLR



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
    
def train_epoch(args, model, train_loader, optimizer, epoch, scheduler):
    model.train()
    # lerning_rate = 0
    # for param_group in optimizer.param_groups:
    #     lerning_rate = param_group['lr']
        
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if not args.usemysgd:
          loss.backward()
          optimizer.step()
        else:
          optimizer.step(loss)
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLR: {:.6f}\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), lerning_rate, loss.item()))
    return 10000*loss

def test_epoch(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}\n'.format(
            # test_loss, correct, len(test_loader.dataset)))
    

    test_loss = (test_loss*10000) / len(test_loader.dataset)
    accuracy = correct#100. * correct / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset), accuracy))
    return (test_loss,accuracy)

def train(args, model, train_loader, optimizer, val, scheduler):
    if args.tp:
        os.system("taskset -apc %d %d" % (0 % mp.cpu_count(), os.getpid()))
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        print("Training: Epoch = " + str(epoch))
        loss = train_epoch(args, model, train_loader, optimizer, epoch, scheduler)
        val.value += 1
        print("TrainError = " + str('%.6f'%loss.item()) + "\n")


def modelsave(args, model, val):
    if args.tp:
        os.system("taskset -apc %d %d" % (1 % mp.cpu_count(), os.getpid()))
    counter = 0
    while counter < args.epochs:
        if val.value > counter:
            torch.save(model.state_dict(),"./saved_models/mnist_cnn"+str(counter)+".pt")
            counter += 1
def testerror(args, model, test_loader, results):
    for i in range(args.epochs):
        print("TestError Computing: Epoch = " + str(i) + "\n")
        model.load_state_dict(torch.load("./saved_models/mnist_cnn"+str(i)+".pt"))
        l,a = test_epoch(model, test_loader)
        results[i][0] = l
        results[i][1] = a    
def trainerror(args, model, test_loader, results):
    for i in range(args.epochs):
        print("TrainError Computing: Epoch = " + str(i) + "\n")
        model.load_state_dict(torch.load("./saved_models/mnist_cnn"+str(i)+".pt"))
        l,a = test_epoch(model, test_loader)
        results[i][2] = l
        results[i][3] = a    

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=2000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--usemysgd', type=int, default=1, metavar='U',
                        help='Whether to use custom SGD')
    parser.add_argument('--tp', type=int, default=0, metavar='U',
                        help='Whether to use Thread pinning')
    parser.add_argument('--timemeasure', type=int, default=1, metavar='U',
                        help='Whether Time measure')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    use_cuda = False #not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': mp.cpu_count(), 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, num_workers=mp.cpu_count())

    model = Net()#.to(device)
    model.share_memory() # gradients are allocated lazily, so they are not shared here
    
    
    if args.usemysgd:
      optimizer = StochasticGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
      optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    gamma = 0.9# + torch.rand(1).item()/10
    scheduler = MyLR(optimizer, gamma)#lrs.ReduceLROnPlateau(optimizer, 'min', gamma) #
    # gamma = 0.9 + torch.rand(1).item()/10
    # scheduler = lrs.ExponentialLR(optimizer, gamma)
    val = mp.Value('i', 0)
    
    f = open('LR='+str(args.lr)+'_epoch='+str(args.epochs)+'.txt',"w")
    
    # print('Stochastic Gradient descent: Batch-size = {}'.format(args.batch_size))
    # f.write('Stochastic Gradient descent: Batch-size = {}'.format(args.batch_size))
    start = time.time()
    processes = []
    if args.tp:
        p = mp.Process(target=train, args=(args, model, train_loader, optimizer, val, scheduler))
    else:
        p = mpt.Process(target=train, args=(args, model, train_loader, optimizer, val, scheduler))
    p.start()
    processes.append(p)
    
    # if args.timemeasure == 0:
    if args.tp:
        p = mp.Process(target=modelsave, args=(args, model, val))
    else:
        p = mpt.Process(target=modelsave, args=(args, model, val))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()
    train_end = time.time()
    train_time = (train_end - start)
    
    if args.timemeasure == 0:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, num_workers=mp.cpu_count())

        results = torch.zeros(args.epochs,4)
        testerror(args, model, test_loader, results)
        trainerror(args, model, train_loader, results)
        f.write("\n\nEpoch\tLR\tTestLoss\tTestAccuracy\tTrainLoss\tTrainAccuracy\n\n")
        
        for i in range(args.epochs):
            f.write('{}\t'.format(i))
            f.write(str('%.6f'%results[i][0].item())+"\t")
            f.write(str('%.2f'%results[i][1].item())+"\t")
            f.write(str('%.6f'%results[i][2].item())+"\t")
            f.write(str('%.2f'%results[i][3].item())+"\n")
        

    print("Training time = " + str(train_time)) 
    f.write("\n\nTraining time = " + str(train_time)) 
    f.close()

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
