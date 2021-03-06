import os
import torch
import torch.multiprocessing as mp
# import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from mysgd import StochasticGD
from myscheduler import MyLR
def train(rank, args, model, barrier):
    if args.tp:
        os.system("taskset -apc %d %d" % (rank % mp.cpu_count(), os.getpid()))
    torch.manual_seed(args.seed + rank)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1)

    optimizer = StochasticGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    gamma = 0.9 + torch.rand(1).item()/10
    scheduler = MyLR(optimizer, gamma)#lrs.ReduceLROnPlateau(optimizer, 'min', gamma) #
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        print("Training: Epoch = " + str(epoch))
        loss = train_epoch(epoch, args, model, train_loader, optimizer)
        barrier[rank] +=1
        print("TrainError = " + str('%.6f'%loss.item()) + "\n")

def modelsave(args, model, barrier):
    if args.tp:
        os.system("taskset -apc %d %d" % (args.num_processes % mp.cpu_count(), os.getpid()))
    counter = torch.zeros([len(barrier)], dtype=torch.int32)
    count = 0
            
    while count < args.epochs:
        allincremented = True
        for i in range(len(barrier)):
            if barrier[i] <= counter[i]:
                allincremented = False
                break
        if allincremented:
            torch.save(model.state_dict(),"./saved_models/mnist_cnn"+str(args.num_processes)+str(count)+".pt")
            count += 1
            for i in range(len(barrier)):
                counter[i] =count
            

def testerror(args, model, test_loader, results):
    for i in range(args.epochs):
        print("TestError Computing: Epoch = " + str(i) + "\n")
        model.load_state_dict(torch.load("./saved_models/mnist_cnn"+str(args.num_processes)+str(i)+".pt"))
        l,a = test_epoch(model, test_loader)
        results[i][0] = l
        results[i][1] = a    
def trainerror(args, model, test_loader, results):
    for i in range(args.epochs):
        print("TrainError Computing: Epoch = " + str(i) + "\n")
        model.load_state_dict(torch.load("./saved_models/mnist_cnn"+str(args.num_processes)+str(i)+".pt"))
        l,a = test_epoch(model, test_loader)
        results[i][2] = l
        results[i][3] = a    

def train_epoch(epoch, args, model, data_loader, optimizer):
    model.train()
    # pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if not args.usemysgd:
            loss.backward()
        optimizer.step(loss, args.usemysgd)
        # if batch_idx % args.log_interval == 0:
        #     print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         pid, epoch, batch_idx * len(data), len(data_loader.dataset),
        #         100. * batch_idx / len(data_loader), loss.item()))
    return 10000*loss


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

def test(args, model, results, barrier, train_loader):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, num_workers=mp.cpu_count())
    testerror(args, model, test_loader, results)
    trainerror(args, model, train_loader, results)
