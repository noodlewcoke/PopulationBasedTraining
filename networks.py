import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import random

# MNIST classifier for test of concept
class classifierMNISTCNN(torch.nn.Module):

    def __init__ (self, initus, exitus, bias=False, lr=0.001):
        super(classifierMNISTCNN, self).__init__()
        
        # self.cnnLayer01 = torch.nn.Conv2d(1, 16, 3)
        # self.cnnLayer02 = torch.nn.Conv2d(16, 32, 3)
        
        # self.fcLayer01 = torch.nn.Linear(18432, 32, bias=bias)
        # self.fcLayer02 = torch.nn.Linear(32, exitus, bias=bias)

        self.cnnLayer01 = torch.nn.Conv2d(1, 20, 5, 1)
        self.cnnLayer02 = torch.nn.Conv2d(20, 50, 1)
        
        self.fcLayer01 = torch.nn.Linear(1800, 500, bias=bias)
        self.fcLayer02 = torch.nn.Linear(500, exitus, bias=bias)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward (self, input):
        q = input 
        q = F.relu(self.cnnLayer01(q))
        q = F.max_pool2d(q, 2, 2)
        q = F.relu(self.cnnLayer02(q))
        q = F.max_pool2d(q, 2, 2)
        q = torch.flatten(q, 1)
        q = F.relu(self.fcLayer01(q))
        q = self.fcLayer02(q)

        return q 
    
    def predict (self, input):
        return F.softmax(self(input), dim=1)

    def update (self, input, target):
        output = self(input)

        self.optimizer.zero_grad()

        loss = F.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()
        return loss

    def alterLR (self, learning_rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        print("Learning rate changed to {}".format(learning_rate))

    def eval(self, input, target):
        output = self.predict(input)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return correct
    
    def save(self, filepath):
        torch.save({
            'model_state_dict' : self.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict()
        } , filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def perturb(self, worker, seed):
        np.random.seed(seed)
        for _,v in self.state_dict().items():
            v += Variable(torch.FloatTensor(np.random.normal(0.0, 0.001, size = v.shape))).cuda()

        #* Hyperparameter update
        lrChange = max(0.001, np.random.normal(0.0, 0.01))
        for g in self.optimizer.state_dict()['param_groups']:
            g['lr'] += lrChange


               
class classifierMNISTFC(torch.nn.Module):

    def __init__ (self, initus, exitus, bias=False, lr=0.001):
        super(classifierMNISTFC, self).__init__()
        
        self.fcLayer01 = torch.nn.Linear(initus, 256, bias=bias)
        self.fcLayer02 = torch.nn.Linear(256, exitus, bias=bias)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


    def forward (self, input):
        q = input 
        q = F.relu(self.fcLayer01(q))
        q = self.fcLayer02(q)

        return q 
    
    def predict (self, input):
        return F.softmax(self(input))

    def update (self, input, target):
        output = self(input)

        self.optimizer.zero_grad()

        loss = F.cross_entropy(output, target)
        loss.backward()
        self.optimizer.step()
        return loss

    def eval(self, input, target):
        output = self.predict(input)
        _, prediction = torch.max(output.data, 1)
        correct = (prediction == target).sum().item()
        return correct

    def alterLR (self, learning_rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        print("Learning rate changed to {}".format(learning_rate))
    
    def save(self, filepath):
        torch.save({
            'model_state_dict' : self.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict()
        } , filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def perturb(self, worker, seed):
        np.random.seed(seed)
        for _,v in self.state_dict().items():
            v += Variable(torch.FloatTensor(np.random.normal(0.0, 0.001, size = v.shape))).cuda()

        #* Hyperparameter update
        lrChange = max(0.001, np.random.normal(0.0, 0.01))
        for g in self.optimizer.state_dict()['param_groups']:
            g['lr'] += lrChange

        
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, next_action, done, q1n, q2n):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, next_action, done, q1n, q2n)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, next_action, done, q1n, q2n = map(np.stack, zip(*batch))
        return state, action, reward, next_state, next_action, done, q1n, q2n
    
    def __len__(self):
        return len(self.buffer)


class nstepReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, return1, return2):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, return1, return2)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, return1, return2 = map(np.stack, zip(*batch))
        return state, action, return1, return2
    
    def __len__(self):
        return len(self.buffer)

class DeepDoubleSarsa(torch.nn.Module):

    def __init__(self, initus, exitus, bias=False, lr=0.00025):
        super(DeepDoubleSarsa, self).__init__()

        # self.cn1 = torch.nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=1)
        # self.cn1b = torch.nn.BatchNorm2d(32)
        # self.cn2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        # self.cn2b = torch.nn.BatchNorm2d(64)
        # self.cn3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.cn3b = torch.nn.BatchNorm2d(64)
        
        self.fc1 = torch.nn.Linear(initus, 64, bias=bias)
        # self.fc1b = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(64, 64, bias=bias)
        self.fc3 = torch.nn.Linear(64, exitus, bias=bias)

        
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, input):
        q = input
        # q = torch.nn.functional.relu(self.cn1b(self.cn1(q)))
        # q = torch.nn.functional.relu(self.cn2b(self.cn2(q)))
        # q = torch.nn.functional.relu(self.cn3b(self.cn3(q)))

        # q = q.view(-1, self.num_flat_features(q))
        q = torch.nn.functional.relu(self.fc1(q))
        q = torch.nn.functional.relu(self.fc2(q))
        q = self.fc3(q)

        return q

    def update(self, sarsa, q2, gamma):   
        s, a, r, sn, an, d = sarsa
        s = Variable(torch.FloatTensor(s)).cuda()
        r = Variable(torch.FloatTensor(r))
        d = Variable(torch.FloatTensor(d))
        qb = Variable(torch.FloatTensor(q2))

        # q = self(s.view(-1, 1, 84, 80))
        q = self(s)

        in_q = [np.arange(len(a)), a]
        in_qb = [np.arange(len(an)), an]

        self.optimizer.zero_grad()
        loss = torch.mean(torch.pow(r + (1.0 - d)*gamma*qb[in_qb] - q.cpu()[in_q],2)/2.0)
        loss.backward()
        self.optimizer.step()
        return loss

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def save(self, filepath):
        torch.save({
            'model_state_dict' : self.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict()
        } , filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def perturb(self, worker, seed):
        np.random.seed(seed)
        for _,v in self.state_dict().items():
            v += Variable(torch.FloatTensor(np.random.normal(0.0, 0.001, size = v.shape))).cuda()

        #* Hyperparameter update
        lrChange = max(0.001, np.random.normal(0.0, 0.01))
        for g in self.optimizer.state_dict()['param_groups']:
            g['lr'] += lrChange


class nstepDeepSarsa(torch.nn.Module):

    def __init__(self, initus, exitus, bias=False, lr = 0.001):
        super(nstepDeepSarsa, self).__init__()

        # self.cn1 = torch.nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=1)
        # self.cn2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(initus, 64, bias=bias)
        self.fc2 = torch.nn.Linear(64, 64, bias=bias)
        self.fc3 = torch.nn.Linear(64, exitus, bias=bias)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, input):
        x = input
        # x = F.relu(self.cn1(x))
        # x = F.relu(self.cn2(x))
        # x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x 

    def update(self, sarsa): # n-step gamma to be calculated in while loop
        s, a, G_tn = sarsa
        s = Variable(torch.FloatTensor(s)).cuda()
        G_tn = Variable(torch.FloatTensor(G_tn))

        q = self(s)
        q = torch.squeeze(q)
        in_q = [np.arange(len(a)), a]
        self.optimizer.zero_grad()
        loss = torch.mean(torch.pow(G_tn - q.cpu()[in_q],2)/2.0)
        loss.backward()
        self.optimizer.step()
        return loss

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def save(self, filepath):
        torch.save({
            'model_state_dict' : self.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict()
        } , filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def perturb(self, worker, seed):
        np.random.seed(seed)
        for _,v in self.state_dict().items():
            v += Variable(torch.FloatTensor(np.random.normal(0.0, 0.001, size = v.shape))).cuda()

        #* Hyperparameter update
        lrChange = max(0.001, np.random.normal(0.0, 0.01))
        for g in self.optimizer.state_dict()['param_groups']:
            g['lr'] += lrChange
