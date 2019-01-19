import multiprocessing
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
from torchvision import datasets, transforms

from mysgd import BATCH_PARTITIONED_SGD
from myscheduler import MyLR

def train(rank, args, model, result, test_loader, barrier, lock, rankstart, rankstop):
    os.system("taskset -apc %d %d" % (rank % multiprocessing.cpu_count(), os.getpid()))
    torch.manual_seed(args.seed + rank)
    gamma = 0.9 + torch.rand(1).item()/10
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    # if args.usemysgd:
    optimizer = BATCH_PARTITIONED_SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # else:
    #   optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = MyLR(optimizer, gamma)#lrs.ReduceLROnPlateau(optimizer, 'min', gamma) #
    for epoch in range(1, args.epochs + 1):
        # if args.lra:
        scheduler.step()
        train_epoch(epoch, args, model, train_loader, optimizer, rankstart, rankstop)
        with lock:
          barrier[rank] += 1
        # if rank == 0:
        result[epoch-1][rank] = get_lr(optimizer)
        tl, a = test_epoch(model, test_loader, False)

def test(args, model, results, test_loader, barrier, lock):
    os.system("taskset -apc %d %d" % (args.num_processes % multiprocessing.cpu_count(), os.getpid()))
    torch.manual_seed(args.seed)

     # l,a = test_epoch(model, test_loader)
    counter = 0
    l_counter = 0
    already_checked = torch.zeros(len(barrier))
    # np = args.num_processes
    while counter < args.epochs:
        for i in range(len(barrier)):
            if barrier[i] > 0 and already_checked[i] == 0:
                with lock:
                    barrier[i] -= 1
                l_counter += 1
                already_checked[i] = 1

        # print("l_counter and counter = " + str(l_counter) + " " + str(counter))      

        if l_counter == args.num_processes:
            l,a = test_epoch(model, test_loader, True)
            print("Epoch: "+ str(counter) + " Test_loss= " + str('%.6f'%l) + "\n")
            results[counter][args.num_processes] = l
            results[counter][args.num_processes+1] = a
            counter += 1
            l_counter = 0
            already_checked = torch.zeros(len(barrier))


def train_epoch(epoch, args, model, data_loader, optimizer, rankstart, rankstop):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if not args.usemysgd:
          loss.backward()
        #   optimizer.step()
        # else:#rank, l, plength, numproc, chunk_size, usemysgd, 
        optimizer.step(loss, rankstart, rankstop, args.usemysgd)
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    


def test_epoch(model, data_loader, istesting):
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
    accuracy = 100. * correct / len(data_loader.dataset)
    if istesting:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),accuracy))
    return (test_loss,accuracy)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
