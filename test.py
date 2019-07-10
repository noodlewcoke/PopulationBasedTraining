import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
import torch
from torch.autograd import Variable
from networks import classifierMNISTFC, classifierMNISTCNN
import torch.multiprocessing as mp 
import copy

from torchvision import datasets, transforms




train_loader = torch.utils.data.DataLoader(
datasets.MNIST('../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
batch_size=64, shuffle=False)

test_loader = torch.utils.data.DataLoader(
datasets.MNIST('../data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
batch_size=64, shuffle=True)


def fcMNIST():
    classifier = classifierMNISTFC(784, 10, bias=True, lr=0.001)
    classifier.cuda()

    for e in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data.cuda())
            data = torch.flatten(data, 1)
            target = Variable(target.cuda())
            loss = classifier.update(data, target)
            print("{:.2f}%\tEpoch :{:.2f}\tLoss :{:.2f}".format(100*batch_idx/len(train_loader), e, loss))

def evaluate(model):
    correct = 0
    total = len(test_loader.dataset)
    print(total)
    for data, target in test_loader:
        data = Variable(data.cuda())
        # data = torch.flatten(data, 1)

        # target = oneHot(target)
        target = Variable(target.cuda())
        numCorrect, _ = model.eval(data, target)
        correct += numCorrect
    return 100*correct/total    

def cnnMNIST():
    lr = 0.001
    classifier = classifierMNISTCNN(784, 10, bias=False, lr=lr)
    classifier.cuda()
    for e in range(5):
        epLoss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data.cuda())
            target = Variable(target.cuda())
            loss = classifier.update(data, target)
            epLoss += loss.item()
        print("{} Loss: {}".format(e, epLoss/len(train_loader)))
    ev  = evaluate(classifier)
    print("Model {} evaluation accuracy: %{}".format(1, ev))

# cnnMNIST()
# model1 = classifierMNISTCNN(None, 10, bias=False, lr=0.1).to('cuda')
# model2 = classifierMNISTCNN(None, 10, bias=True, lr=0.001).to('cuda')
def sarsaTest():

    act_len = 9 #! Breakout: 4 | MsPackman: 9
    env = gym.make(ENV0)
    if ENV == ENV0:
        act_len = 9
    elif ENV == ENV1:
        act_len = 4
    obs_len = env.observation_space.shape[0]
    ddsarsa = DeepDoubleSarsa(obs_len, act_len, bias=True)
    ddsarsa.to(device)
    target = DeepDoubleSarsa(obs_len, act_len, bias=True)
    target.to(device)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    alpha, gamma, epsilon = 0.1, 0.99, 1.0
    max_score = 0
    rewards, lossa, lossb = [], [], []
    print(env.observation_space.shape)
    print(env.action_space)
    
    for e in range(1, EPISODES+1):
        done = False
        total_reward = 0

        obs1 = env.reset()
        obs = Variable(torch.from_numpy(obs1))
        obs = obs.view(-1, obs_len)
        obs = obs.float()
        obs = obs.to(device)
        qa = ddsarsa(obs)
        qb = target(obs)
        qa = np.squeeze(qb.cpu().data.numpy())
        qb = np.squeeze(qa.cpu().data.numpy())
        a = gym_act(env, qa, qb, epsilon)

        loss = Variable(torch.FloatTensor([0.0]))
        t = 0
        while not done:
            n_obs1, r, done, _ = env.step(a)
            total_reward +=r
            n_obs = Variable(torch.from_numpy(n_obs1))
            n_obs = n_obs.view(-1, obs_len)
            n_obs = n_obs.float()
            n_obs = n_obs.to(device)
            n_qa = ddsarsa(n_obs)
            n_qb = target(n_obs)
            n_qa = np.squeeze(n_qa.cpu().data.numpy())
            n_qb = np.squeeze(n_qb.cpu().data.numpy())
            an = gym_act(env, n_qa, n_qb, epsilon) 
            if done:
                replay_buffer.push(obs1, a, r, n_obs1, an, 1.0, np.squeeze(n_qa.cpu().data.numpy()), np.squeeze(n_qb.cpu().data.numpy()))
            else:
                replay_buffer.push(obs1, a, r, n_obs1, an, 0.0, np.squeeze(n_qa.cpu().data.numpy()), np.squeeze(n_qb.cpu().data.numpy()))
            a = an
            if len(replay_buffer)>=BATCH_SIZE:
                # if e%target_update:
                if np.random.rand(1)[0] < 0.5:
                    s, ac, r, sp, ap, d, q1nn, q2nn = replay_buffer.sample(BATCH_SIZE)
                    loss = ddsarsa.update([s, ac, r, sp, ap, d], q2nn, gamma)
                    # lossa.append(loss.item())
                else:
                    s, ac, r, sp, ap, d, q1nn, q2nn = replay_buffer.sample(BATCH_SIZE)
                    loss = target.update([s, ac, r, sp, ap, d], q1nn, gamma)
                    # ddsarsa.save('models/sarsa/target.pt')
                    # target.load('models/sarsa/target.pt')

            obs1 = n_obs1
            t += 1

        if e<epsilon_decay:
                epsilon -= (epsilon_start - epsilon_stop)/epsilon_decay
        rewards.append(total_reward)

        if max_score < total_reward:
            max_score = total_reward
        
        if not e%50:
            gymEvaluate(env, ddsarsa, target, numEpisodes=20)
        # if SAVE and e%saving_fq==0:
        #     ddsarsa.save("models/cartpole_dm1a.pt")
        #     target.save("models/cartpole_dm1b.pt")
        #     np.save('models/cartpole_fc1_rewards', rewards)
        #     np.save('models/cartpole_fc1_lossa', lossa)

        # if SAVE and total_reward > max_score:
        #     if e>epsilon_decay:
        #         ddsarsa.save("models/max_dma.pt")
        #         target.save("models/max_dmb.pt")
        #     max_score = total_reward
        
        print("Episode: {} | Reward: {} | Loss: {}".format(e, total_reward, loss.item()))
        print("Max reward: {}".format(max_score))
        # rewards.append(total_reward)
# pre = copy.deepcopy(model1.state_dict())
# print(pre["cnnLayer01.weight"][0])
# for k,v in model1.optimizer.state_dict().items():
    # print(v.shape)
    # print(k)
# for i in model1.optimizer.state_dict()["param_groups"]:
#     print(i["lr"])

# model1.perturb()
#     # print(np.random.normal(0.0, 1.0, size = v.shape))
#     v += Variable(torch.FloatTensor(np.random.normal(0.0, 0.1, size = v.shape)))

# post = model1.state_dict().copy()
# print(pre["cnnLayer01.weight"][0])

# print(post["cnnLayer01.weight"][0])
# print(pre["cnnLayer01.weight"][0] == post["cnnLayer01.weight"][0])


# print(dict(model1.state_dict())["cnnLayer01.weight"])
# for g in model1.optimizer.state_dict()["param_groups"]:
#     print(type(g["lr"]))

# print(model1.optimizer.state_dict()['state'])
# print(model1.optimizer.state_dict()['param_groups'])

# print(model2.optimizer.state_dict())

# print(model1.state_dict()['cnnLayer01.weight'][0] == model2.state_dict()['cnnLayer01.weight'][0])
# print(model1.optimizer.state_dict() == model2.optimizer.state_dict())
# model1.state_dict = model2.state_dict
# model1.optimizer.state_dict = model2.optimizer.state_dict
# print(model1.state_dict()['cnnLayer01.weight'][0] == model2.state_dict()['cnnLayer01.weight'][0])
# print(model1.optimizer.state_dict() == model2.optimizer.state_dict())


# for i in range(1, 10000):
#     if not i%5:
#         print("\r The line at: {}".format(i),end='', flush=True)


# def oneHot(vec):
#     mat = torch.zeros((vec.size(0), 10))
#     indices = [np.arange(vec.size(0)), vec]
#     mat[indices] = 1
#     return mat
# print(len(train_loader))
# for d,t in train_loader:
#     print(t)
#     # one = oneHot(t)
#     # print(one)
#     t = Variable(t)
#     p = torch.LongTensor([5, 0])
#     # print(t)
#     # print(p)
#     # print((t == p).sum().item())
#     break
# print(len(train_loader))
