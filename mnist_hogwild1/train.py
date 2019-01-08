import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from mysgd import BATCH_PARTITIONED_SGD

def train(rank, args, model, result):
    # rank = q.get()
    torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    optimizer = BATCH_PARTITIONED_SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(rank, epoch, args, model, train_loader, optimizer, result)

def test(args, model):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    test_epoch(model, test_loader)


def train_epoch(rank, epoch, args, model, data_loader, optimizer, result):
    model.train()
    pid = os.getpid()
    l = 10
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step(rank, args.num_processes)
        l += loss.item()
        # if batch_idx % args.log_interval == 0:
        #     print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         pid, epoch, batch_idx * len(data), len(data_loader.dataset),
        #         100. * batch_idx / len(data_loader), loss.item()))
    result[epoch - 1][rank] = l
    # print("result[",epoch - 1,"][",rank,"]= ", result[epoch - 1][rank])
        


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
