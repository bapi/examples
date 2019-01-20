import multiprocessing
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
from torchvision import datasets, transforms

from mysgd import StochasticGD

def train(rank, args, model, result, train_loader, barrier, lock):
    # os.system("taskset -apc %d %d" % (rank % multiprocessing.cpu_count(), os.getpid()))
    torch.manual_seed(args.seed + rank)
    
    if args.usemysgd:
      optimizer = StochasticGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
      optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, train_loader, optimizer, lock)
        barrier[rank] += 1
        result[epoch-1][rank] = get_lr(optimizer)

def test(args, model, results, test_loader, barrier, istrain):
    # if istrain:
    #     os.system("taskset -apc %d %d" % (args.num_processes % multiprocessing.cpu_count(), os.getpid()))
    # else:
    #     os.system("taskset -apc %d %d" % ((args.num_processes+1) % multiprocessing.cpu_count(), os.getpid()))
    torch.manual_seed(args.seed)

    counter = torch.zeros([len(barrier)], dtype=torch.int32)
    count = 0
    # np = args.num_processes
    while count < args.epochs:
        allincremented = True
        for i in range(len(barrier)):
            if barrier[i] <= counter[i]:
                allincremented = False
                break

        if allincremented:
            if istrain:
                l,a = test_epoch(model, test_loader)
                print("Epoch: "+ str(count) + " Train_loss= " + str('%.6f'%l) + 
                " Train_accuracy= " + str('%.2f'%a) + "\n")
                results[count][args.num_processes+2] = l
                results[count][args.num_processes+3] = a
            else:
                l,a = test_epoch(model, test_loader)
                print("Epoch: "+ str(count) + " Test_loss= " + str('%.6f'%l) + 
                " Test_accuracy= " + str('%.2f'%a) + "\n")
                results[count][args.num_processes] = l
                results[count][args.num_processes+1] = a
            count +=1
            for i in range(len(barrier)):
                counter[i] +=count
        


def train_epoch(epoch, args, model, data_loader, optimizer, lock):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if not args.usemysgd:
          loss.backward()
          optimizer.step()
        else:
          optimizer.step(loss, lock)
        # if batch_idx % args.log_interval == 0:
        #     print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         pid, epoch, batch_idx * len(data), len(data_loader.dataset),
        #         100. * batch_idx / len(data_loader), loss.item()))
    


def test_epoch(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target).sum().item()

    test_loss = (test_loss*10000) / len(data_loader.dataset)
    accuracy = correct#100. * correct / len(data_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(data_loader.dataset),accuracy))
    return (test_loss,accuracy)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
