import math
import time
import torch
import numpy as np
from .dataselectionstrategy import DataSelectionStrategy
from ..helpers import OrthogonalMP_REG_Parallel, OrthogonalMP_REG, OrthogonalMP_REG_Parallel_V1
from torch.utils.data import Subset, DataLoader
import threading
from submodlib import FacilityLocationFunction, GraphCutFunction, \
    DisparityMinFunction, DisparitySumFunction, LogDeterminantFunction

class OnlineSubmodStrategy(DataSelectionStrategy):
    """
    GradMatch strategy tries to solve the optimization problem given below:

    .. math::
        \\min_{\\mathbf{w}, S: |S| \\leq k} \\Vert \\sum_{i \\in S} w_i \\nabla_{\\theta}L_T^i(\\theta) -  \\nabla_{\\theta}L(\\theta)\\Vert

    In the above equation, :math:`\\mathbf{w}` denotes the weight vector that contains the weights for each data instance, :math:`\mathcal{U}` training set where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively,
    :math:`L_T` denotes the training loss, :math:`L` denotes either training loss or validation loss depending on the parameter valid,
    :math:`S` denotes the data subset selected at each round, and :math:`k` is the budget for the subset.

    The above optimization problem is solved using the Orthogonal Matching Pursuit(OMP) algorithm.

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
                 selection_type, logger, args, testset, valid=False, v1=True, lam=0, eps=1e-4):
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
        self.opt_subset = None
        self.current_batch = None
        self.args = args
        self.moment_sum = 0
        self.device = "cuda"
        self.testset = testset
        
    def get_val_images(self):
        assert self.testset is not None
        images = self.testset[0]
        labels = self.testset[1]
        rand = torch.randint(images.shape[0], (32,))
        val_images, val_labels = images[rand], labels[rand]
        return val_images, val_labels
        
    def importance_sampling(self, prev_subset, current_batch):
        sampling_mode = self.args["sampling_mode"]
        images, prev_images = current_batch["images"], prev_subset["images"]
        labels, prev_labels = current_batch["labels"], prev_subset["labels"]
        features, prev_features = current_batch["features"], prev_subset["features"]
        s_prev = 0 if prev_images is None else prev_images.shape[0]
        s_curr = images.shape[0]
        len_opt = 0
        if(prev_images is None): sampling_mode = None
        if(sampling_mode == "Uniform"):
            lamb = self.lam
            prev_sample = torch.randint(s_prev, (int(s_prev*lamb),))  
            curr_sample = torch.randint(s_curr, (int(s_curr*(1-lamb)),))  
            ret_images = torch.cat((prev_images[prev_sample], images[curr_sample]), dim=0)
            ret_labels = torch.cat((prev_labels[prev_sample], labels[curr_sample]), dim=0)
            ret_features = torch.cat((prev_features[prev_sample], features[curr_sample]), dim=0)
            len_opt = prev_sample.shape[0]
            return {"images": ret_images, "labels": ret_labels, "features": ret_features}, len_opt
        
        if(sampling_mode == "Binomial"):
            raise NotImplementedError
        
        if(sampling_mode == None):
            len_opt = current_batch["images"].shape[0]
            return {"images": images, "labels":labels, "features":features}, len_opt
        
    
    def get_greedy_list(funcs, submod_budget, multi_thread=False):
        greedyList = {}
        def thread(x):
            # Maximize the function
            f = funcs[x]
            _greedy = f.maximize(budget = submod_budget, optimizer='NaiveGreedy', 
                                stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, 
                                verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)
            greedyList[x] = _greedy
        
        if multi_thread:
            '''Multi-threading'''
            threads = [threading.Thread(target=thread, args=(i,)) for i in range(len(funcs))]
            [t.start() for t in threads]
            [t.join() for t in threads]
            
            '''Multi-processing'''
            # pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
            # [pool.submit(thread, i) for i in range(len(funcs))]
            # pool.shutdown(wait=True)
        else:
            for i,f in enumerate(funcs):
                # Maximize the function
                _greedy = f.maximize(budget = submod_budget, optimizer='NaiveGreedy', 
                                    stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, 
                                    verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)
                greedyList[i] = _greedy
        return greedyList

    
    def best_submod_bandit(self, subset, len_opt, greedyList, eta_n, budget):
        best_index = 0
        best_metric = -10000000
        subset_size = subset["images"].shape[0]
        subset_grads = cat(subset_grads, subset_size) # B,P
        val_grads = cat(val_grads, val_images.shape[0]) # Bv,P
        alpha = 0.7
        # moment_sum = alpha*(val_grads*val_grads) + (1-alpha)*moment_sum
        
        
        for i in range(len(greedyList)):
            submod_indices = [greedyList[i][j][0] for j in range(len(greedyList[i]))]
            term1 = eta_n*subset_grads[submod_indices]@(val_grads.transpose(0,1)) # B',1
            # print("term1", term1.shape)
            grad_sum = torch.sum(subset_grads[0:len_opt], dim=0, keepdim=True)
            # eps = 1e-5
            hessian = torch.ones(grad_sum.transpose(0,1).shape).to(self.device)
            # assert torch.equal(grad_sum.transpose(0,1), hessian*grad_sum.transpose(0,1)) == True
            # hessian = (moment_sum).transpose(0,1)
            term2 = eta_n*eta_n*subset_grads[submod_indices]@((hessian*(grad_sum.transpose(0,1)))) # B',1
            metric =  torch.sum(term1 - term2, dim=0)/subset_size
            metric = torch.mean(metric, dim=0)
            if(metric.item() > best_metric):
                best_metric = metric.item()
                best_index = i
        # print("Best", best_index, best_metric)
        return greedyList[best_index]
    
    def eps_greedy(self, X, Y, subset, budget):
        lamb = self.lam
        pi = self.args["pi"]
        # thresh = step/((step+lamb)**pi)
        eps = torch.rand(1).item()
        submod_budget = budget
        
        
        greedyList = self.get_greedy_list(self.funcs, submod_budget)
        # if(eps > thresh):
        return self.best_submod_bandit(subset, len_opt, greedyList,  0.5, budget)
        # else:
        #     sample = torch.randint(len(greedyList), ()).item()
        #     return greedyList[sample]


    def eps_greedy__OLD__(self, X, Y, bud):
        if self.device == "cpu":
            reg = OrthogonalMP_REG(X.numpy(), Y.numpy(), nnz=bud, positive=True, lam=0)
            ind = np.nonzero(reg)[0]
        else:
            if self.v1:
                reg = OrthogonalMP_REG_Parallel_V1(X, Y, nnz=bud,
                                                 positive=True, lam=self.lam,
                                                 tol=self.eps, device=self.device)
            else:
                reg = OrthogonalMP_REG_Parallel(X, Y, nnz=bud,
                                                positive=True, lam=self.lam,
                                                tol=self.eps, device=self.device)
            ind = torch.nonzero(reg).view(-1)
        return ind.tolist(), reg[ind].tolist()

    def select(self, budget, model_params):
        """
        Apply OMP Algorithm for data selection

        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing models parameters

        Returns
        ----------
        idxs: list
            List containing indices of the best datapoints,
        gammas: weights tensors
            Tensor containing weights of each instance
        """
        omp_start_time = time.time()
        self.update_model(model_params)
        n = subset["features"].shape
        subset, len_opt = self.importance_sampling(self.opt_subset, {"images": images, "features": features, "labels": labels}, args)
        n = subset["images"].shape[0]
        funcs = [
            GraphCutFunction(n = n, data = subset["features"], mode="sparse", metric="euclidean",num_neighbors=5, lambdaVal = 0.1),
            DisparityMinFunction(n = n, data = subset["features"], mode="sparse", metric="euclidean",num_neighbors=5),
            DisparitySumFunction(n = n, data = subset["features"], mode="sparse", metric="euclidean",num_neighbors=5),
            LogDeterminantFunction(n, mode="sparse", lambdaVal = 0.1,  data = subset["features"], metric="euclidean", num_neighbors=5),
            FacilityLocationFunction(n, mode="sparse", data = subset["features"], metric="euclidean", num_neighbors=5),
            # SetCoverFunction(n, cover_set=features, num_concepts=5),
            # ProbabilisticSetCoverFunction(n, cover_set=features, num_concepts=5),
        ]

        if self.selection_type == 'PerClass_LATER':
            self.get_labels(valid=self.valid)
            idxs = []
            gammas = []
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                                shuffle=False, pin_memory=True, collate_fn=self.trainloader.collate_fn)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                  shuffle=False, pin_memory=True, collate_fn=self.trainloader.collate_fn)

                self.compute_gradients(self.valid, perBatch=False, perClass=True)
                trn_gradients = self.grads_per_elem
                if self.valid:
                    sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
                else:
                    sum_val_grad = torch.sum(trn_gradients, dim=0)
                idxs_temp, gammas_temp = self.eps_greedy(torch.transpose(trn_gradients, 0, 1),
                                                         sum_val_grad,
                                                         math.ceil(budget * len(trn_subset_idx) / self.N_trn))
                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
                gammas.extend(gammas_temp)

        elif self.selection_type == 'PerBatch':
            val_images, val_labels = self.get_val_images()
            self.compute_gradients_v2(subset, val_images, val_labels)
            idxs = []
            gammas = []
            trn_gradients = self.grads_per_elem
            if self.valid:
                sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
            else:
                sum_val_grad = torch.sum(trn_gradients, dim=0)
            idxs_temp, gammas_temp = self.eps_greedy(torch.transpose(trn_gradients, 0, 1),
                                                     sum_val_grad, math.ceil(budget / self.trainloader.batch_size))
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(idxs_temp)):
                tmp = batch_wise_indices[idxs_temp[i]]
                idxs.extend(tmp)
                gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))

        elif self.selection_type == 'PerClassPerGradient_LATER':
            self.get_labels(valid=self.valid)
            idxs = []
            gammas = []
            embDim = self.model.get_embedding_dim()
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                                shuffle=False, pin_memory=True, collate_fn=self.trainloader.collate_fn)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                  shuffle=False, pin_memory=True, collate_fn=self.trainloader.collate_fn)
                self.compute_gradients(self.valid, perBatch=False, perClass=True)
                trn_gradients = self.grads_per_elem
                tmp_gradients = trn_gradients[:, i].view(-1, 1)
                tmp1_gradients = trn_gradients[:,
                                 self.num_classes + (embDim * i): self.num_classes + (embDim * (i + 1))]
                trn_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)

                if self.valid:
                    val_gradients = self.val_grads_per_elem
                    tmp_gradients = val_gradients[:, i].view(-1, 1)
                    tmp1_gradients = val_gradients[:,
                                     self.num_classes + (embDim * i): self.num_classes + (embDim * (i + 1))]
                    val_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)
                    sum_val_grad = torch.sum(val_gradients, dim=0)
                else:
                    sum_val_grad = torch.sum(trn_gradients, dim=0)

                idxs_temp, gammas_temp = self.eps_greedy(torch.transpose(trn_gradients, 0, 1),
                                                         sum_val_grad,
                                                         math.ceil(budget * len(trn_subset_idx) / self.N_trn))
                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
                gammas.extend(gammas_temp)
        diff = budget - len(idxs)
        self.logger.debug("Random points added: %d ", diff)

        if diff > 0:
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            idxs.extend(new_idxs)
            gammas.extend([1 for _ in range(diff)])
            idxs = np.array(idxs)
            gammas = np.array(gammas)

        if self.selection_type in ["PerClass", "PerClassPerGradient"]:
            rand_indices = np.random.permutation(len(idxs))
            idxs = list(np.array(idxs)[rand_indices])
            gammas = list(np.array(gammas)[rand_indices])
        
        idxs = [int(x) for x in idxs]
        omp_end_time = time.time()
        self.logger.debug("OnlineSubmod algorithm Subset Selection time is: %.4f", omp_end_time - omp_start_time)
        return idxs, torch.FloatTensor(gammas)