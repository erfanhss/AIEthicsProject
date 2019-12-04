import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### GPU Settings
use_cuda = True
device = torch.device('cuda:0' if use_cuda else 'cpu')
### Defining Required Functions and global Parameters
T = 40
state = {}
state['lr'] = 0.01


def eval_test(model, data_loader):
  model.eval()
  num_correct = 0
  loss_test = 0
  for _, (data, label) in enumerate(data_loader):
    data = data.to(device)
    label = label.to(device)
    scores = model(data)
    pred = torch.argmax(scores, 1)
    num_correct += torch.sum(torch.eq(pred, label)).item()
  return  num_correct/(len(data_loader.dataset))


def smooth_cross_enropy(labels, inputs):
  log_input = F.log_softmax(inputs) 
  tmp = torch.mul(labels, log_input)
  losses = torch.sum(tmp, axis=1)
  return torch.mean(losses)


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch+1 %10 == 0:
        state['lr'] *= 0.95 
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

### Loading Dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
### Definging Model Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.dp1 = nn.Dropout()
        self.dp2 = nn.Dropout()
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.dp1(self.fc1(x)))
        x = F.relu(self.dp2(self.fc2(x)))
        x = self.fc3(x)
        return x
### Instantiating and Training the First Model
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
best_acc = 0
for epoch in range(100): 
    net.train()
    adjust_learning_rate(optimizer, epoch)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        outputs = torch.mul(outputs, 1/T)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    net.eval()
    train_acc = eval_test(net, trainloader)
    test_acc = eval_test(net, testloader)
    print(epoch)
    print('Train Accuracy: ' + str(train_acc))
    print('Test Accuracy: ' + str(test_acc))
    if test_acc > best_acc:
      best_acc = test_acc
      PATH = './defensive_first_ep' + str(epoch) + '_T' + str(T) + '.pth'
      torch.save(net.state_dict(), PATH)
      print('Model Saved!')
    print('-------------------')
%time
print('Finished Training')
##### Training the Second Network Using the preditions of the first network as labels
net_first = Net().to(device)
path_first = './defensive_first_ep95_T40.pth'
net_first.load_state_dict(torch.load(path_first))
net_first.eval()
print(eval_test(net_first, testloader))
net = Net().to(device)
state['lr'] = 0.01
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
best_acc = 0
for epoch in range(100):  # loop over the dataset multiple times
    net.train()
    adjust_learning_rate(optimizer, epoch)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, _ = data
        inputs = inputs.to(device)
        labels = net_first(inputs)
        labels = torch.mul(labels, 1/T)
        labels = F.softmax(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        outputs = torch.mul(outputs, 1/T)
        loss = -smooth_cross_enropy(labels, outputs)
        loss.backward()
        optimizer.step()
    net.eval()
    train_acc = eval_test(net, trainloader)
    test_acc = eval_test(net, testloader)
    print(epoch)
    print('Train Accuracy: ' + str(train_acc))
    print('Test Accuracy: ' + str(test_acc))
    if test_acc > best_acc:
      best_acc = test_acc
      PATH = './defensive_second_ep' + str(epoch) + '_T' + str(T) + '.pth'
      torch.save(net.state_dict(), PATH)
      print('Model Saved')
    print('-------------------')
%time
print('Finished Training Second Model')
