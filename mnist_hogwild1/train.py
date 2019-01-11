import os
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from torchvision import datasets, transforms
from mysgd import BATCH_PARTITIONED_SGD

def train(rank, args, model, result, learning_rates):
    # rank = q.get()
    torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    optimizer = BATCH_PARTITIONED_SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = lrs.ExponentialLR(optimizer, gamma = 0.95)
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train_epoch(rank, epoch, args, model, train_loader, optimizer)
        result[epoch-1][rank] = test(args, model)
        if rank == 0:
          learning_rates[epoch-1] = get_lr(optimizer)


def test(args, model):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    return test_epoch(model, test_loader)


def train_epoch(rank, epoch, args, model, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step(rank, args.num_processes)
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLearning-rate: {:.6f}\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), get_lr(optimizer), loss.item()))
        


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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    return test_loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
