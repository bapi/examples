from __future__ import print_function
import argparse
import torch.multiprocessing as mp
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
from resnet_class import ResNet, ResidualBlock, transform, train_dataset, test_dataset, criterion

def train_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # lerning_rate = 0
    # for param_group in optimizer.param_groups:
    #     lerning_rate = param_group['lr']
        
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if not args.usemysgd:
          loss.backward()
          optimizer.step()
        else:
          optimizer.step(loss)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return 10000*loss

def test_epoch(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item() # sum up batch loss
            _, predicted = torch.max(output.data, 1)
            # total += target.size(0)
            correct += (predicted == target).sum().item()
            # accuracy = correct/total

            

    test_loss = (test_loss*10000) / len(test_loader.dataset)
    # accuracy = correct#100. * correct / len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset), accuracy))
    return (test_loss,correct)

def train(args, model, device, train_loader, optimizer, val):
    if args.tp:
        os.system("taskset -apc %d %d" % (0 % mp.cpu_count(), os.getpid()))
    for epoch in range(1, args.epochs + 1):
        print("Training: Epoch = " + str(epoch))
        loss = train_epoch(args, model, device, train_loader, optimizer, epoch)
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
    parser.add_argument('--usemysgd', type=int, default=0, metavar='U',
                        help='Whether to use custom SGD')
    parser.add_argument('--tp', type=int, default=1, metavar='U',
                        help='Whether to use Thread pinning')
    parser.add_argument('--timemeasure', type=int, default=1, metavar='U',
                        help='Whether Time measure')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = False #not args.no_cuda and torch.cuda.is_available()
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=args.batch_size, 
                                              shuffle=True, num_workers=mp.cpu_count())
    

    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

    # model = Net()#.to(device)
    model.share_memory() # gradients are allocated lazily, so they are not shared here
    
    if args.usemysgd:
      optimizer = StochasticGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
      optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # gamma = 0.9 + torch.rand(1).item()/10
    # scheduler = lrs.ExponentialLR(optimizer, gamma)
    val = mp.Value('i', 0)
    
    if args.usemysgd:
      f = open('LR='+str(args.lr)+'_usebackprop=True.txt',"w")
    else:
      f = open('LR='+str(args.lr)+'_usebackprop=False.txt',"w")

    # print('Stochastic Gradient descent: Batch-size = {}'.format(args.batch_size))
    # f.write('Stochastic Gradient descent: Batch-size = {}'.format(args.batch_size))
    start = time.time()
    processes = []
    p = mp.Process(target=train, args=(args, model, device, train_loader, optimizer, val))
    p.start()
    processes.append(p)
    
    if args.timemeasure == 0:
        p = mp.Process(target=modelsave, args=(args, model, val))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    train_end = time.time()
    train_time = (train_end - start)
    
    f.write("\n\nEpoch\tLR\tTestLoss\tTestAccuracy\tTrainLoss\tTrainAccuracy\n\n")
    if args.timemeasure == 0:
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
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
