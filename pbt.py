import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
import torch
from torch.autograd import Variable
from functools import partial
from multiprocessing import Pool
import torch.multiprocessing as mp 
import operator


def evaluate(model):
    correct = 0
    total = len(test_loader.dataset)
    for data, target in test_loader:
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        numCorrect = model.eval(data, target)
        correct += numCorrect
    return 100*correct/total

def train_asyncLoad(worker, model, train_acc):

    #* This method selects the best performer only (not one from TOP %20)
    epochLoss = 0.0; numEpisodes = len(train_loader)
    print("Training Model {}...".format(worker))
    for data, target in train_loader:
        data = Variable(data.cuda())
        target = Variable(target.cuda())

        loss = model.update(data, target)
        epochLoss += loss.item()
    print("Model {} epoch training loss: {}".format(worker, epochLoss/numEpisodes))
    
    #* If epoch for the individual is done then it is READY 
    
    #? EVALUATE PERFORMANCE
    ev  = evaluate(model)
    print("Model {} evaluation accuracy: %{}".format(worker, ev))
    train_acc[worker] = ev

    model.save('models/model{}'.format(worker))


def exploitPopulation(train_acc):
    #? EXPLOIT
    bestPerformer = max(train_acc.items(), key = operator.itemgetter(1))[0]
    return bestPerformer

def exploreIndividual(worker, model, bestPerformer):
    #? EXPLORE

    if not bestPerformer == worker:
        model.load('models/model{}'.format(bestPerformer)) #! Async behavior forces to load a file which is being saved.
        print("Model {} is changed to Model {}".format(worker, bestPerformer))

        #? Perturbation sequence
        seed = np.random.randint(0, 1000, 1)[0]
        model.perturb(worker, seed)
        model.save('models/model{}'.format(worker)) #* Save method saves both weights and optimizer hyperparameters
        train_acc[worker] = evaluate(model)


def genericPBT(EPOCH):
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except:
        pass
    num_processes = 3
    lrs = [0.1, 0.05, 0.01]
    models = []

    
    for i in lrs:
        models.append(classifierMNISTCNN(None, 10, True, i).to(device))
    processes = []
    train_acc = mp.Manager().dict()
    for e in range(1, EPOCH+1):
        for rank in range(num_processes):
            p = mp.Process(target=train_asyncLoad, args=(rank, models[rank], train_acc))
            p.start()
            processes.append(p)
        print("\n===========================================\n")
        print("Epoch {} - Training/Evaluation Sequences\n".format(e))
        for p in processes: p.join()

        print("\nEpoch {} - Exploration/Exploitation Sequences\n".format(e))
        bestPerformer = exploitPopulation(train_acc)
        for i, model in enumerate(models):
            exploreIndividual(i, model, bestPerformer)
    
    #* Save best model at the end
    print("\nSaving best performer: Model {}.".format(bestPerformer))
    models[bestPerformer].save("models/bestPerformer")