import math
import time
import torch
import numpy as np
from .dataselectionstrategy_new import DataSelectionStrategy_onlinesubmod
from ..helpers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG, OrthogonalMP_REG_Parallel_V1
from torch.utils.data import Subset, DataLoader
from submodlib import FacilityLocationFunction, GraphCutFunction, \
    DisparityMinFunction, DisparitySumFunction, LogDeterminantFunction, SetCoverFunction, ProbabilisticSetCoverFunction



class OnlineSubmodStrategy(DataSelectionStrategy_onlinesubmod):
    """
    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    loss: class
        PyTorch loss function for training
    eta: float
        Learning rate. Step size for the one step gradient update
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    selection_type: str
        Type of selection -
        - 'PerClass': PerClass method is where OMP algorithm is applied on each class data points seperately.
        - 'PerBatch': PerBatch method is where OMP algorithm is applied on each minibatch data points.
        - 'PerClassPerGradient': PerClassPerGradient method is same as PerClass but we use the gradient corresponding to classification layer of that class only.
    logger : class
        - logger object for logging the information
    valid : bool
        If valid==True, we use validation dataset gradient sum in OMP otherwise we use training dataset (default: False)
    v1 : bool
        If v1==True, we use newer version of OMP solver that is more accurate
    lam : float
        Regularization constant of OMP solver
    eps : float
        Epsilon parameter to which the above optimization problem is solved using OMP algorithm
    """

    def __init__(self, trainloader, valloader, model, loss,
                 eta, device, num_classes, linear_layer,
                 selection_type, logger, num_val_points, valid=False, v1=True, lam=0, eps=1e-4):
        """
        Constructor method
        """
        super().__init__(trainloader, valloader, model, num_classes, linear_layer, loss, device, logger)
        self.eta = eta  # step size for the one step gradient update
        self.device = device
        self.init_out = list()
        self.init_l1 = list()
        self.selection_type = selection_type
        self.valid = valid
        self.lam = lam
        self.eps = eps
        self.v1 = v1
        self.device = "cuda"
        self.num_val_points = num_val_points


    # def eps_greedy(self, X, Y, subset, budget):
    #     lamb = self.lam
    #     pi = self.args["pi"]
    #     # thresh = step/((step+lamb)**pi)
    #     eps = torch.rand(1).item()
    #     submod_budget = budget
        
        
    #     greedyList = self.get_greedy_list(self.funcs, submod_budget)
    #     # if(eps > thresh):
    #     return self.best_submod_bandit(subset, len_opt, greedyList,  0.5, budget)
    #     # else:
    #     #     sample = torch.randint(len(greedyList), ()).item()
    #     #     return greedyList[sample]
  
    def select(self, budget, clone_params, orig_params, clone_model, orig_model):
        """
        Parameters
        ----------
        budget: int
            The number of data points to be selected
        clone_params: OrderedDict
            Python dictionary object containing models parameters

        Returns
        ----------
        idxs: list
            List containing indices of the best datapoints,
        gammas: weights tensors
            Tensor containing weights of each instance
        """
        omp_start_time = time.time()
        self.update_model(clone_params)

        if self.selection_type == 'PerBatch':
            # trn_gradients = self.grads_per_elem
            # sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
            # idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
            #                                          sum_val_grad, math.ceil(budget / self.trainloader.batch_size))
            # batch_wise_indices = list(self.trainloader.batch_sampler)
            # for i in range(len(idxs_temp)):
            #     tmp = batch_wise_indices[idxs_temp[i]]
            #     idxs.extend(tmp)
            #     gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))
            batch_scores = self.compute_gradients(self.valid, perBatch=True, perClass=False)
            idxs_temp, gammas_temp = self.get_greedy_lists(budget, clone_params, orig_params, clone_model, orig_model,
                                                 self.grads_per_elem, self.val_grads_per_elem, batch_scores.cpu().detach().numpy(), 128)
            
            idxs = []
            gammas = []
            trn_gradients = self.grads_per_elem
            sum_val_grad = torch.sum(trn_gradients, dim=0)
            # idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
            #                                          sum_val_grad, math.ceil(budget / self.trainloader.batch_size))
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(idxs_temp)):
                tmp = batch_wise_indices[idxs_temp[i]]
                idxs.extend(tmp)
                gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))
        else:
            raise NotImplementedError
        
        diff = budget - len(idxs)
        self.logger.debug("Random points added: %d ", diff)

        if diff > 0:
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            idxs.extend(new_idxs)
            gammas.extend([1 for _ in range(diff)])
            idxs = np.array(idxs)
            gammas = np.array(gammas)

        # if self.selection_type in ["PerClass", "PerClassPerGradient"]:
        #     rand_indices = np.random.permutation(len(idxs))
        #     idxs = list(np.array(idxs)[rand_indices])
        #     gammas = list(np.array(gammas)[rand_indices])
        
        idxs = [int(x) for x in idxs]
        omp_end_time = time.time()
        self.logger.debug("OnlineSubmod algorithm Subset Selection time is: %.4f", omp_end_time - omp_start_time)
        return idxs, torch.FloatTensor(gammas)