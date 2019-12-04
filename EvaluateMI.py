import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
### Setting GPU Parameters
use_cuda = True
device = torch.device('cuda:0' if use_cuda else 'cpu')
### Defining Global Parameters and Functions
T=40


def make_one_hot(labels, C=2):
    if use_cuda:
      one_hot = torch.cuda.FloatTensor(labels.size(0), C).zero_()
    else:
      one_hot = torch.FloatTensor(labels.size(0), C).zero_()
    target = one_hot.scatter_(1, labels.data, 1)  
    target = Variable(target)
    return target


def eval_test(model, data_loader):
  num_correct = 0
  loss_test = 0
  for _, (data, label) in enumerate(data_loader):
    data = data.to(device)
    label = label.to(device)
    scores = model(data)
    pred = torch.argmax(scores, 1)
    num_correct += torch.sum(torch.eq(pred, label)).item()
  return  num_correct/(len(data_loader.dataset))


def eval_attacker(att, clsfr, intrain_loader, intest_loader):
  num_correct = 0
  for _, ((intrain_data, intrain_label), (intest_data, intest_label)) in enumerate(
      zip(intrain_loader, intest_loader)
  ):
    intrain_data, intrain_label = intrain_data.to(device), intrain_label.to(device)
    intest_data, intest_label = intest_data.to(device), intest_label.to(device)
    data = torch.cat((intrain_data, intest_data))
    labels = make_one_hot(torch.cat((intrain_label, intest_label)).view(-1, 1), 10)
    scores = att(torch.nn.functional.softmax(clsfr(data)), labels)
    pred = (scores > 0).float()
    u = torch.cat((torch.ones(len(intrain_data)), torch.zeros(len(intest_data)))).to(device)
    num_correct += torch.sum(torch.eq(pred.view(-1), u)).item()
  return  num_correct/(len(intrain_loader.dataset) + len(intest_loader.dataset))


### Loading Dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
torch.manual_seed(0)
trainset_nonprivate_train, trainset_nonprivate_test, _ = torch.utils.data.dataset.random_split(trainset, [8000, 2000, 40000])
testset_nonprivate_train, testset_nonprivate_test = torch.utils.data.dataset.random_split(testset, [8000, 2000])
torch.manual_seed(torch.initial_seed())

trainset_train_loader = torch.utils.data.DataLoader(trainset_nonprivate_train, batch_size=64, shuffle=True)
testset_train_loader = torch.utils.data.DataLoader(testset_nonprivate_train, batch_size=64, shuffle=True)


trainset_test_loader = torch.utils.data.DataLoader(trainset_nonprivate_test, batch_size=256, shuffle=False)
testset_test_loader = torch.utils.data.DataLoader(testset_nonprivate_test, batch_size=256, shuffle=False)
### Defining Model Architecture
class Attacker (nn.Module):
    def __init__(self):
        super(Attacker, self).__init__()
        self.linear_prob = nn.Sequential(nn.Linear(10, 1024),
                               nn.ReLU(),
                               nn.Linear(1024, 512),
                               nn.ReLU(), 
                               nn.Linear(512, 64),
                               nn.ReLU())
        self.linear_label = nn.Sequential(nn.Linear(10, 512), 
                                          nn.ReLU(),
                                          nn.Linear(512, 64),
                                          nn.ReLU())
        self.linear_cat = nn.Sequential(nn.Linear(128, 256), 
                                        nn.ReLU(),
                                        nn.Linear(256, 64), 
                                        nn.ReLU(),
                                        nn.Linear(64, 1))
    def forward(self, prob_vec, labels):
        first = self.linear_prob(prob_vec)
        sec = self.linear_label(labels)
        x = torch.cat((first, sec), axis=1)
        return self.linear_cat(x)



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


### Instantiating Adversary and Loading the Targer Classifier
adversary = Attacker().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(adversary.parameters(), lr=0.00001)
classifier = Net().to(device)
classifier.eval()
path = './defensive_second_ep80_T41.pth'
classifier.load_state_dict(torch.load(path))
print(eval_test(classifier, torch.utils.data.DataLoader(testset, batch_size=128)))
num_epochs = 200
best_acc = 0
for epoch in range(num_epochs):
  for _, ((intrain_data, intrain_label), (intest_data, intest_label)) in enumerate(
      zip(trainset_train_loader, testset_train_loader)
  ):
    adversary.train()
    intrain_data, intrain_label = intrain_data.to(device), intrain_label.to(device)
    intest_data, intest_label = intest_data.to(device), intest_label.to(device)
    data = torch.cat((intrain_data, intest_data))
    labels = make_one_hot(torch.cat((intrain_label.view(-1, 1), intest_label.view(-1, 1))), 10)
    u = torch.cat((torch.ones(len(intrain_data)), torch.zeros(len(intest_data)))).to(device)
    prob_vecs = classifier(data)
    att_logits = adversary(torch.nn.functional.softmax(prob_vecs), labels)
    
    loss = criterion(att_logits, u.view(-1, 1))
    loss.backward()
    optimizer.step()
  adversary.eval()
  train_acc = eval_attacker(adversary, classifier, trainset_train_loader, testset_train_loader)
  test_acc = eval_attacker(adversary, classifier, trainset_test_loader, testset_test_loader)
  print(epoch)
  print('Train Acc: '+str(train_acc))
  print('Test Acc:' + str(test_acc))
  if test_acc > best_acc :
    best_acc = test_acc
    path = './att_model_' + str(T) + '_epoch' + str(epoch) + '.att'
    torch.save(adversary.state_dict(), path)
    print('model saved!')
  
  print('----------------------------------------------------')
  
