import os
import torch
# import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from mysgd import StochasticGD
def train(rank, args, model, barrier):
    torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    optimizer = StochasticGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, train_loader, optimizer)
        barrier[rank] +=1

def test(args, model, results, barrier):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, num_workers=1)
    counter = torch.zeros([len(barrier)], dtype=torch.int32)
    count = 0
            
    while count < args.epochs:
        allincremented = True
        for i in range(len(barrier)):
            if barrier[i] <= counter[i]:
                allincremented = False
                break
        if allincremented:
            l, a = test_epoch(model, test_loader)
            print("Epoch: "+ str(count) + " Test_loss= " + str('%.6f'%l) + 
                " Test_accuracy= " + str('%.2f'%a) + "\n")
            results[count][0] = l
            results[count][1] = a
            count +=1
            for i in range(len(barrier)):
                counter[i] +=count
            



def test_train(args, model, results, barrier):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, num_workers=1)

    counter = torch.zeros([len(barrier)], dtype=torch.int32)
    count = 0
            
    while count < args.epochs:
        allincremented = True
        for i in range(len(barrier)):
            if barrier[i] <= counter[i]:
                allincremented = False
                break
        if allincremented:
            l, a = test_epoch(model, test_loader)
            print("Epoch: "+ str(count) + " Train_loss= " + str('%.6f'%l) + 
                " Train_accuracy= " + str('%.2f'%a) + "\n")
            results[count][2] = l
            results[count][3] = a
            count +=1
            for i in range(len(barrier)):
                counter[i] +=count
    

def train_epoch(epoch, args, model, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step(loss)
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

    test_loss /= len(data_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset)))
    return (test_loss*10000, correct)