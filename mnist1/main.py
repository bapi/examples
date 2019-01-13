from __future__ import print_function
import argparse
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torchvision import datasets, transforms
from torch.multiprocessing import Process, Value, Lock, Queue


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    lerning_rate = 0
    for param_group in optimizer.param_groups:
        lerning_rate = param_group['lr']
        
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLR: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lerning_rate, loss.item()))
    return lerning_rate

def test_epoch(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

def train(args, model, device, train_loader, optimizer, scheduler, results, val, lock):
  for epoch in range(1, args.epochs + 1):
        scheduler.step()
        lerning_rate = train_epoch(args, model, device, train_loader, optimizer, epoch)
        with lock:
          val.value += 1
        results[epoch - 1][0] = lerning_rate


def test(args, model, device, test_loader, results, val, lock):
  counter = 0
  while counter < args.epochs:
    if val.value > 0:
      with lock:
        val.value -= 1
      l = test_epoch(args, model, device, test_loader)
      print("Epoch: "+ str(counter) + " Test_loss= " + str('%.6f'%l) + "\n")
      results[counter][1] = l
      # f.write(str('%.6f'%l)+"\n")
      counter += 1
    # print("still waiting for update!")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
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
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net()#.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = lrs.ExponentialLR(optimizer, gamma = 0.95)
    val = Value('i', 0)
    lock = Lock()
    results = torch.zeros(args.epochs,2)
    results.share_memory_()
    
    f = open('stochastic_gradient_descent'+'_batch_size='+str(args.batch_size)+'.txt',"w")
    print('Stochastic Gradient descent: Batch-size = {}'.format(args.batch_size))
    f.write('Stochastic Gradient descent: Batch-size = {}'.format(args.batch_size))
    f.write("\n\nEpoch\tLR\tLoss\n\n")
    start = time.time()
    processes = []
    for rank in range(1):
        p = Process(target=train, args=(args, model, device, train_loader, optimizer, scheduler, results, val, lock))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    p = Process(target=test, args=(args, model, device, test_loader, results, val, lock))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()
    train_end = time.time()
    train_time = (train_end - start)
    
    for i in range(args.epochs):
      f.write('{}\t'.format(i))
      f.write(str('%.6f'%results[i][0].item())+"\t")
      f.write(str('%.6f'%results[i][1].item())+"\n")
      

    print("Training time = " + str(train_time)) 
    f.write("\n\nTraining time = " + str(train_time)) 
    f.close()


    # for epoch in range(1, args.epochs + 1):
    #     scheduler.step()
    #     train_epoch(args, model, device, train_loader, optimizer, epoch)
    #     test_epoch(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
