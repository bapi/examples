import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
from torchvision import datasets, transforms

from mysgd import BATCH_PARTITIONED_SGD

def train(rank, args, model, plength, chunk_size, result, val, lock):
    torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    if args.usemysgd:
      optimizer = BATCH_PARTITIONED_SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
      optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = lrs.ExponentialLR(optimizer, gamma = 0.95)
    for epoch in range(1, args.epochs + 1):
        if args.lra:
          scheduler.step()
        train_epoch(epoch, args, model, plength, chunk_size, train_loader, optimizer, rank)
        with lock:
          val.value += 1
        if rank == 0:
          result[epoch-1][0] = get_lr(optimizer)

def test(args, model, results, val, lock):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    # l,a = test_epoch(model, test_loader)
    counter = 0
    np = args.num_processes
    while counter < args.epochs:
      if val.value == np:
        with lock:
          val.value -= np
        l,a = test_epoch(model, test_loader)
        print("Epoch: "+ str(counter) + " Test_loss= " + str('%.6f'%l) + "\n")
        results[counter][1] = l
        results[counter][2] = a
        counter += 1
      # print("value and counter = " + str(val.value) + " " + str(counter))
  


def train_epoch(epoch, args, model, plength, chunk_size, data_loader, optimizer, rank):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if not args.usemysgd:
          loss.backward()
          optimizer.step()
        else:#rank, l, plength, numproc, chunk_size, usemysgd, 
          optimizer.step(rank, loss, plength, args.num_processes, chunk_size, args.usemysgd)
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
    


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

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),accuracy))
    return (test_loss,accuracy)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
