import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.multiprocessing as mp 
import operator
import random



#! NOTE : hyper_dist should be a dictionary of hyperparameter distributions
class PBT:

    def __init__(self, model_class, kwargs, n_best, n_worst, n_population, hyper_dist, train_fn, eval_fn, train_kwargs, eval_kwargs): #? SNR?
        self.model_class = model_class
        self.kwargs = kwargs
        self.n_best = n_best
        self.n_worst = n_worst
        self.n_population = n_population
        self.hyper_dist = hyper_dist
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.train_kwargs = train_kwargs
        self.eval_kwargs = eval_kwargs

        self.shared = [model(**kwargs) for i in range(n_best)] 
        for m in self.shared:
            m.share_memory_()

        self.manager = mp.Manager()
        
        self.best_performance = self.manager.list([-np.inf for i in range(n_best)])
        self.worst_performance = self.manager.list([np.inf for i in range(n_worst)])


        # self.performance = self.manager.dict()

    def worker(self):
        
        model = self.model_class(**self.kwargs)

        #! Hyperparameter sample DISCRETE
        



        #* train 
        #* eval
        #* exploit
        #* explore


if __name__ == "__main__":
    pass