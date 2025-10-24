import torch
from submodlib import FacilityLocationFunction, GraphCutFunction, \
    DisparityMinFunction, DisparitySumFunction, LogDeterminantFunction, SetCoverFunction, ProbabilisticSetCoverFunction
from tqdm import tqdm
import submod_new as submod
import math


class DataSelectionStrategy_onlinesubmod(object):
    """
    Implementation of Data Selection Strategy class which serves as base class for other
    dataselectionstrategies for general learning frameworks.
    Parameters
    ----------
        trainloader: class
            Loading the training data using pytorch dataloader
        valloader: class
            Loading the validation data using pytorch dataloader
        model: class
            Model architecture used for training
        num_classes: int
            Number of target classes in the dataset
        linear_layer: bool
            If True, we use the last fc layer weights and biases gradients
            If False, we use the last fc layer biases gradients
        loss: class
            PyTorch Loss function
        device: str
            The device being utilized - cpu | cuda
        logger: class
            logger object for logging the information
    """

    def __init__(self, trainloader, valloader, model, num_classes, linear_layer, loss, device, logger):
        """
        Constructor method
        """
        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader

        self.model = model
        self.N_trn = len(trainloader.sampler)
        self.N_val = len(valloader.sampler)
        self.grads_per_elem = None
        self.val_grads_per_elem = None
        self.numSelected = 0
        self.linear_layer = linear_layer
        self.num_classes = num_classes
        self.trn_lbls = None
        self.val_lbls = None
        self.loss = loss
        self.device = device
        self.logger = logger
        self.num_val_points = 32
        self.prev_opt_subset = {}
        self.opt_subset = {}

        def get_samples():
            val_images = None
            val_labels = None
            for (_, images), labels in valloader:
                if images.dtype == torch.uint8:
                    images = images.float()
                images = images.to(device)
                labels = labels.to(device)
                if (val_images is None):
                    val_images = images
                    val_labels = labels
                else:
                    val_images = torch.cat((val_images, images), dim=0)
                    val_labels = torch.cat((val_labels, labels), dim=0)
            return [val_images, val_labels]
        self.val_list = get_samples()
        self.batch_scores = self.batch_sijs()

    def select(self, budget, model_params):
        pass

    def get_labels(self, valid=False):
        if isinstance(self.trainloader.dataset[0], dict):
            for batch_idx, batch in enumerate(self.trainloader):
                if batch_idx == 0:
                    self.trn_lbls = batch['labels'].view(-1, 1)
                else:
                    self.trn_lbls = torch.cat(
                        (self.trn_lbls, batch['labels'].view(-1, 1)), dim=0)
        else:
            for batch_idx, (_, targets) in enumerate(self.trainloader):
                if batch_idx == 0:
                    self.trn_lbls = targets.view(-1, 1)
                else:
                    self.trn_lbls = torch.cat(
                        (self.trn_lbls, targets.view(-1, 1)), dim=0)
        self.trn_lbls = self.trn_lbls.view(-1)

        if valid:
            if isinstance(self.valloader.dataset[0], dict):
                for batch_idx, batch in enumerate(self.valloader):
                    if batch_idx == 0:
                        self.val_lbls = batch['labels'].view(-1, 1)
                    else:
                        self.val_lbls = torch.cat(
                            (self.val_lbls, batch['labels'].view(-1, 1)), dim=0)
            else:
                for batch_idx, (_, targets) in enumerate(self.valloader):
                    if batch_idx == 0:
                        self.val_lbls = targets.view(-1, 1)
                    else:
                        self.val_lbls = torch.cat(
                            (self.val_lbls, targets.view(-1, 1)), dim=0)
            self.val_lbls = self.val_lbls.view(-1)

    # def compute_loss(self, params, buffers, sample, target):
    #     batch = sample.unsqueeze(0)
    #     targets = target.unsqueeze(0)
    #     predictions = torch.func.functional_call(self.model, (params, buffers), (batch,))
    #     loss = self.loss(predictions, targets)
    #     return loss

    # TODO: optimise

    def compute_loss(self,  params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = torch.func.functional_call(
            self.model, (params, buffers), (batch,))
        loss = self.loss(predictions, targets)
        return torch.mean(loss)

    def cat(self, x, batch_size):
        return torch.cat([x[key].view(batch_size, -1) for key in x.keys()], dim=1)

    def compute_grads(self, model, params, buffers, images, labels):
        self.model = model
        self.model.eval()
        ft_compute_grad = torch.func.grad(self.compute_loss)
        ft_compute_sample_grad = torch.func.vmap(
            ft_compute_grad, in_dims=(None, None, 0, 0))
        grads = ft_compute_sample_grad(
            params, buffers, images, labels.to(self.device))
        grads = self.cat(grads, images.shape[0])  # B,P
        self.model.train()
        return grads

    def compute_gradients_modelparams(self, model, params, buffers, images, labels, valid=False):
        self.model = model
        self.model.eval()
        ft_compute_grad = torch.func.grad(self.compute_loss)
        ft_compute_sample_grad = torch.func.vmap(
            ft_compute_grad, in_dims=(None, None, 0, 0))
        grads = ft_compute_sample_grad(
            params, buffers, images, labels.to(self.device))
        # if valid:
        #     self.val_grads_per_elem = grads
        # else: self.grads_per_elem = grads
        grads = self.cat(grads, images.shape[0])  # B,P
        self.model.train()
        return grads

        # return grads, val_grads

    def get_val_images(self):
        assert self.val_list is not None
        images = self.val_list[0]
        labels = self.val_list[1]
        rand = torch.randint(images.shape[0], (32,))
        val_images, val_labels = images[rand], labels[rand]
        return val_images, val_labels

    def get_greedy_list(self, funcs, submod_budget):
        greedyList = {}
        for i, f in enumerate(funcs):
            # Maximize the function
            _greedy = f.maximize(budget=submod_budget, optimizer='NaiveGreedy',
                                 stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1,
                                 verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)
            greedyList[i] = _greedy
        return greedyList

    def eps_greedy_composition(self, step, greedyList, lamb, pi, eta_n, moment_sum, subset_grads, val_grads, batch_idx,  greedyOnly=False):
        thresh = step/((step+lamb)**pi)
        eps = torch.rand(1).item()
        # submod.best_submod_bandit(self.model, )
        if (eps > thresh or greedyOnly):
            return self.best_submod_bandit(greedyList, eta_n, moment_sum, subset_grads, val_grads, batch_idx)
        else:
            sample = torch.randint(len(greedyList), ()).item()
            return greedyList[sample], sample

    def best_submod_bandit(self, greedyList, eta_n, moment_sum, subset_grads, val_grads, batch_idx):
        best_index = 0
        best_metric = -10000000000
        device = "cuda"
        alpha = 0.7
        # moment_sum = alpha*(val_grads*val_grads) + (1-alpha)*moment_sum
        with torch.autocast(device_type=device):
            for i in range(len(greedyList)):
                submod_indices = [greedyList[i][j][0]
                                  for j in range(len(greedyList[i]))]
                t = torch.mean(val_grads, dim=0, keepdim=True)
                term1 = eta_n * \
                    subset_grads[submod_indices]@(t.transpose(0, 1))  # B',1
                # print("term1", term1.shape)
                term2 = 0
                # if(batch_idx not in self.prev_opt_subset.keys()): term2 = 0

                # else:
                #     opt_indices = self.prev_opt_subset[batch_idx]
                #     grad_sum_opt = torch.sum(subset_grads[opt_indices], dim=0, keepdim=True)
                #     term2 = eta_n*eta_n*subset_grads[submod_indices]@((grad_sum_opt.transpose(0,1))) # B',1
                #     # print("term2", term2.shape)
                #     # print("*********************")
                metric = torch.mean(term1 - term2, dim=0)
                # metric = torch.mean(metric, dim=0)
                if (metric.item() > best_metric):
                    best_metric = metric.item()
                    best_index = i
        # print("Best metric", best_metric ,best_index)
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&")

        return greedyList[best_index], best_index

    def importance_sampling(self, prev_subset, current_batch, sampling_mode, lamb):
        images, prev_images = current_batch["images"], prev_subset["images"]
        labels, prev_labels = current_batch["labels"], prev_subset["labels"]
        features, prev_features = current_batch["features"], prev_subset["features"]
        size_at = 0 if prev_images is None else prev_images.shape[0]
        batch_size_curr = images.shape[0]
        len_opt = 0
        # change is here
        if (prev_images is None or size_at == 0):
            sampling_mode = None
        if (sampling_mode == "Uniform"):
            lamb = lamb
            frac_at = int(batch_size_curr*lamb)
            frac_curr = batch_size_curr - frac_at
            prev_sample = torch.randint(size_at, (frac_at,))
            curr_sample = torch.randint(batch_size_curr, (frac_curr,))
            ret_images = torch.cat(
                (prev_images[prev_sample], images[curr_sample]), dim=0)
            ret_labels = torch.cat(
                (prev_labels[prev_sample], labels[curr_sample]), dim=0)
            ret_features = torch.cat(
                (prev_features[prev_sample], features[curr_sample]), dim=0)
            len_opt = prev_sample.shape[0]
            return {"images": ret_images, "labels": ret_labels, "features": ret_features}, len_opt

        if (sampling_mode == "Binomial"):
            raise NotImplementedError

        if (sampling_mode == None):
            len_opt = current_batch["images"].shape[0]
            return {"images": images, "labels": labels, "features": features}, 0
    '''
    opt with and without batch check
    '''

    def batch_sijs(self):
        trainloader = self.trainloader
        batch_size = 0
        feature_mat = None
        for batch_idx, ((features, inputs), targets) in enumerate(tqdm(trainloader, desc="Subset selection")):
            if batch_size == 0:
                batch_size = inputs.shape[0]
            if (features.shape[0] != batch_size):
                continue
            if (feature_mat is None):
                feature_mat = features.unsqueeze(0)
            else:
                feature_mat = torch.cat(
                    (feature_mat, features.unsqueeze(0)), dim=0)
        # numbatches, batchsize, 768
        print("******Features***", feature_mat.shape)
        # n = 768, q= num batches, i = batchsize
        # sij = torch.einsum("nd,nd->")
        token_scores = torch.einsum('qin,pjn->qipj', feature_mat, feature_mat)
        scores, _ = token_scores.max(-1)
        print("scores1", scores.shape)
        scores = scores.sum(1).fill_diagonal_(0)
        print("scores2", scores.shape)
        return scores

    def get_greedy_lists_OLD_(self, budget, clone_params, orig_params, clone_model, orig_model):
        trainloader = self.trainloader
        print("here*******")
        idxs = []
        gammas = []
        device = "cuda"
        # assume trainloader is never shuffled
        # self.update_model(orig_params)
        for batch_idx, ((features, inputs), targets) in enumerate(tqdm(trainloader, desc="Subset selection")):
            submod_budget_per_batch = budget//len(trainloader)
            submod_budget_num_batches = budget//inputs.shape[0]
            subset = {"features": features,
                      "images": inputs, "labels": targets}
            if inputs.dtype == torch.uint8:
                inputs = inputs.float()
            inputs = inputs.to(device)
            targets = targets.to(device)
            features = features.to(device)
            n = inputs.shape[0]
            batch_size = inputs.shape[0]
            rval_images, rval_labels = self.get_val_images()
            batch_grads, val_grads = submod.calc_grads_cachedv2(clone_model, torch.nn.CrossEntropyLoss(
                reduction="none"), inputs, targets, rval_images, rval_labels)
            self.model = orig_model
            funcs = [
                GraphCutFunction(
                    n=n, data=subset["features"], mode="sparse", metric="euclidean", num_neighbors=5, lambdaVal=0.1),
                DisparityMinFunction(
                    n=n, data=subset["features"], mode="sparse", metric="euclidean", num_neighbors=5),
                DisparitySumFunction(
                    n=n, data=subset["features"], mode="sparse", metric="euclidean", num_neighbors=5),
                LogDeterminantFunction(n, mode="sparse", lambdaVal=0.1,
                                       data=subset["features"], metric="euclidean", num_neighbors=5),
                FacilityLocationFunction(
                    n, mode="sparse", data=subset["features"], metric="euclidean", num_neighbors=5),
                # SetCoverFunction(n, cover_set=features, num_concepts=5),
                # ProbabilisticSetCoverFun`ction(n, cover_set=features, num_concepts=5),
            ]
            submod_budget = submod_budget_per_batch
            greedyList = self.get_greedy_list(funcs, submod_budget)
            # print("greedyList", greedyList)
            self.model = orig_model
            # grads =  torch.cat((grads, batch_grads), dim=0) if grads is not None else batch_grads
            # subset is batch for now, without any importance sampling
            subset_grads = batch_grads

            moment_sum = 0
            eta_n = 0.5
            lamb = 0.1
            pi = 1.1
            # greedyFinal, best_arm = self.best_submod_bandit(greedyList, eta_n, moment_sum,subset_grads,val_grads, batch_idx)
            greedyFinal, best_arm = self.eps_greedy_composition(
                batch_idx, greedyList, lamb, pi, eta_n, moment_sum, subset_grads, val_grads, batch_idx)
            # within batch
            submod_indices = [greedyFinal[i][0]
                              for i in range(len(greedyFinal))]
            # if(batch_idx in self.prev_opt_subset.keys()):
            #     print("&&&&&&&&&&&&&&")
            #     print("curr: ", submod_indices)
            #     print("prev: ", self.prev_opt_subset[batch_idx])
            #     print("&&&&&&&&&&&&&&")
            # self.opt_subset
            self.prev_opt_subset[batch_idx] = [i for i in submod_indices]
            global_indices = [batch_idx *
                              (batch_size) + i for i in submod_indices]
            scores = [greedyFinal[i][1] for i in range(len(greedyFinal))]
            # print("greedyfinal",greedyFinal )
            # print("sub", submod_indices)
            # print("global", global_indices)
            # print("&&&&&&&&&&&&&&")
            idxs.extend(global_indices)
            gammas.extend(scores)
            # torch.cuda.empty_cache()
            # greedyLists.append(greedyFinal)
        # self.grads_per_elem = grads
        torch.cuda.empty_cache()
        print("*******final", len(idxs), len(gammas), "*****")
        return idxs, gammas

    def get_greedy_lists(self, budget, clone_params, orig_params, clone_model, orig_model, trn_grads, val_grads, batch_scores, batch_size):
        trainloader = self.trainloader
        print("here*******")
        idxs = []
        gammas = []
        device = "cuda"
        submod_budget_per_batch = budget//len(trainloader)
        submod_budget_num_batches = budget//batch_size
        # assume trainloader is never shuffled
        n = batch_scores.shape[0]
        funcs = [
            DisparityMinFunction(n=n, sijs=batch_scores,
                                 mode="dense", metric="euclidean"),
            DisparitySumFunction(n=n, sijs=batch_scores,
                                 mode="dense", metric="euclidean"),
            LogDeterminantFunction(
                n, mode="dense", lambdaVal=0.1,  sijs=batch_scores, metric="euclidean"),
            FacilityLocationFunction(
                n, mode="dense", sijs=batch_scores, metric="euclidean", separate_rep=False),
        ]
        submod_budget = submod_budget_num_batches
        greedyList = self.get_greedy_list(funcs, submod_budget)
        self.model = orig_model
        subset_grads = trn_grads

        moment_sum = 0
        eta_n = 0.5
        lamb = 0.1
        pi = 1.1
        greedyFinal, best_arm = self.best_submod_bandit(
            greedyList, eta_n, moment_sum, subset_grads, val_grads, 0)
        # within batch
        submod_indices = [greedyFinal[i][0] for i in range(len(greedyFinal))]
        # self.prev_opt_subset[batch_idx] = [i for i in submod_indices]
        # global_indices = [batch_idx*(batch_size) + i for i in submod_indices]
        scores = [greedyFinal[i][1] for i in range(len(greedyFinal))]

        idxs.extend(submod_indices)
        gammas.extend(scores)
        torch.cuda.empty_cache()
        print("*******final", idxs, gammas, "*****")
        return idxs, gammas

    def get_greedy_lists_pb(self, budget, clone_params, orig_params, clone_model, orig_model):
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        grads = None
        trainloader = self.trainloader
        idxs = []
        gammas = []
        device = "cuda"
        # assume trainloader is never shuffled
        # self.update_model(orig_params)
        for batch_idx, ((features, inputs), targets) in enumerate(tqdm(trainloader, desc="Subset selection")):
            submod_budget_per_batch = budget//len(trainloader)
            submod_budget_num_batches = budget//inputs.shape[0]
            subset = {"features": features,
                      "images": inputs, "labels": targets}
            if inputs.dtype == torch.uint8:
                inputs = inputs.float()
            inputs = inputs.to(device)
            targets = targets.to(device)
            features = features.to(device)
            n = inputs.shape[0]
            batch_size = inputs.shape[0]
            rval_images, rval_labels = self.get_val_images()
            # val_grads = self.compute_gradients_modelparams(clone_model, params, buffers, rval_images, rval_labels)
            self.model = orig_model
            funcs = [
                GraphCutFunction(
                    n=n, data=subset["features"], mode="sparse", metric="euclidean", num_neighbors=5, lambdaVal=0.1),
                DisparityMinFunction(
                    n=n, data=subset["features"], mode="sparse", metric="euclidean", num_neighbors=5),
                DisparitySumFunction(
                    n=n, data=subset["features"], mode="sparse", metric="euclidean", num_neighbors=5),
                LogDeterminantFunction(n, mode="sparse", lambdaVal=0.1,
                                       data=subset["features"], metric="euclidean", num_neighbors=5),
                FacilityLocationFunction(
                    n, mode="sparse", data=subset["features"], metric="euclidean", num_neighbors=5),
                # SetCoverFunction(n, cover_set=features, num_concepts=5),
                # ProbabilisticSetCoverFun`ction(n, cover_set=features, num_concepts=5),
            ]
            print("**NEW***")
            batch_grads, val_grads = submod.calc_grads_cached(clone_model, torch.nn.CrossEntropyLoss(
                reduction="none"), inputs, targets, rval_images, rval_labels)

            submod_budget = submod_budget_per_batch
            greedyList = self.get_greedy_list(funcs, submod_budget)
            # print("greedyList", greedyList)
            # batch_grads = self.compute_grads(clone_model, params, buffers, inputs, targets)
            self.model = orig_model
            # grads =  torch.cat((grads, batch_grads), dim=0) if grads is not None else batch_grads
            # subset is batch for now, without any importance sampling
            subset_grads = batch_grads

            moment_sum = 0
            eta_n = 0.5
            lamb = 0.1
            pi = 1.1
            # greedyFinal, best_arm = self.best_submod_bandit(greedyList, eta_n, moment_sum,subset_grads,val_grads, batch_idx)
            greedyFinal, best_arm = self.eps_greedy_composition(
                batch_idx, greedyList, lamb, pi, eta_n, moment_sum, subset_grads, val_grads, batch_idx)
            # within batch
            submod_indices = [greedyFinal[i][0]
                              for i in range(len(greedyFinal))]
            # if(batch_idx in self.prev_opt_subset.keys()):
            #     print("&&&&&&&&&&&&&&")
            #     print("curr: ", submod_indices)
            #     print("prev: ", self.prev_opt_subset[batch_idx])
            #     print("&&&&&&&&&&&&&&")
            # self.opt_subset
            self.prev_opt_subset[batch_idx] = [i for i in submod_indices]
            global_indices = [batch_idx *
                              (batch_size) + i for i in submod_indices]
            scores = [greedyFinal[i][1] for i in range(len(greedyFinal))]
            # print("greedyfinal",greedyFinal )
            # print("sub", submod_indices)
            # print("global", global_indices)
            # print("&&&&&&&&&&&&&&")
            idxs.extend(global_indices)
            gammas.extend(scores)
            # torch.cuda.empty_cache()
            # greedyLists.append(greedyFinal)
        # self.grads_per_elem = grads
        torch.cuda.empty_cache()
        print("*******final", len(idxs), len(gammas), "*****")
        return idxs, gammas

    def compute_gradients(self, valid=False, perBatch=False, perClass=False):
        """
        Computes the gradient of each element.

        Here, the gradients are computed in a closed form using CrossEntropyLoss with reduction set to 'none'.
        This is done by calculating the gradients in last layer through addition of softmax layer.

        Using different loss functions, the way we calculate the gradients will change.

        For LogisticLoss we measure the Mean Absolute Error(MAE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:

        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left| x_n - y_n \\right|,

        where :math:`N` is the batch size.


        For MSELoss, we measure the Mean Square Error(MSE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:

        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left( x_n - y_n \\right)^2,

        where :math:`N` is the batch size.
        Parameters
        ----------
        valid: bool
            if True, the function also computes the validation gradients
        perBatch: bool
            if True, the function computes the gradients of each mini-batch
        perClass: bool
            if True, the function computes the gradients using perclass dataloaders
        """
        if (perBatch and perClass):
            raise ValueError(
                "batch and perClass are mutually exclusive. Only one of them can be true at a time")

        embDim = self.model.get_embedding_dim()
        if perClass:
            trainloader = self.pctrainloader
            if valid:
                valloader = self.pcvalloader
        else:
            trainloader = self.trainloader
            if valid:
                valloader = self.valloader

        if isinstance(trainloader.dataset[0], dict):
            for batch_idx, ((features, inputs), targets) in enumerate(trainloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if batch_idx == 0:
                    out, l1 = self.model(**batch, last=True, freeze=True)
                    loss = self.loss(out, batch['labels'].view(-1)).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(
                            l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if perBatch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                else:
                    out, l1 = self.model(**batch, last=True, freeze=True)
                    loss = self.loss(out, batch['labels'].view(-1)).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(
                            batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * \
                            l1.repeat(1, self.num_classes)
                    if perBatch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(
                                dim=0).view(1, -1)
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

        else:
            for batch_idx, ((features, inputs), targets) in enumerate(trainloader):
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(
                            l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if perBatch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = self.loss(out, targets).sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(
                            batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * \
                            l1.repeat(1, self.num_classes)

                    if perBatch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(
                                dim=0).view(1, -1)
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

        torch.cuda.empty_cache()

        if self.linear_layer:
            self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        else:
            self.grads_per_elem = l0_grads

        if valid:
            if isinstance(valloader.dataset[0], dict):
                for batch_idx, ((features, inputs), targets) in enumerate(valloader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    if batch_idx == 0:
                        out, l1 = self.model(**batch, last=True, freeze=True)
                        loss = self.loss(out, batch['labels'].view(-1)).sum()
                        l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            l0_expand = torch.repeat_interleave(
                                l0_grads, embDim, dim=1)
                            l1_grads = l0_expand * \
                                l1.repeat(1, self.num_classes)
                        if perBatch:
                            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                l1_grads = l1_grads.mean(dim=0).view(1, -1)
                    else:
                        out, l1 = self.model(**batch, last=True, freeze=True)
                        loss = self.loss(out, batch['labels'].view(-1)).sum()
                        batch_l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            batch_l0_expand = torch.repeat_interleave(
                                batch_l0_grads, embDim, dim=1)
                            batch_l1_grads = batch_l0_expand * \
                                l1.repeat(1, self.num_classes)
                        if perBatch:
                            batch_l0_grads = batch_l0_grads.mean(
                                dim=0).view(1, -1)
                            if self.linear_layer:
                                batch_l1_grads = batch_l1_grads.mean(
                                    dim=0).view(1, -1)
                        l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                        if self.linear_layer:
                            l1_grads = torch.cat(
                                (l1_grads, batch_l1_grads), dim=0)
            else:
                for batch_idx, ((features, inputs), targets) in enumerate(valloader):
                    inputs, targets = inputs.to(self.device), targets.to(
                        self.device, non_blocking=True)
                    if batch_idx == 0:
                        out, l1 = self.model(inputs, last=True, freeze=True)
                        loss = self.loss(out, targets).sum()
                        l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            l0_expand = torch.repeat_interleave(
                                l0_grads, embDim, dim=1)
                            l1_grads = l0_expand * \
                                l1.repeat(1, self.num_classes)
                        if perBatch:
                            l0_grads = l0_grads.mean(dim=0).view(1, -1)
                            if self.linear_layer:
                                l1_grads = l1_grads.mean(dim=0).view(1, -1)
                    else:
                        out, l1 = self.model(inputs, last=True, freeze=True)
                        loss = self.loss(out, targets).sum()
                        batch_l0_grads = torch.autograd.grad(loss, out)[0]
                        if self.linear_layer:
                            batch_l0_expand = torch.repeat_interleave(
                                batch_l0_grads, embDim, dim=1)
                            batch_l1_grads = batch_l0_expand * \
                                l1.repeat(1, self.num_classes)

                        if perBatch:
                            batch_l0_grads = batch_l0_grads.mean(
                                dim=0).view(1, -1)
                            if self.linear_layer:
                                batch_l1_grads = batch_l1_grads.mean(
                                    dim=0).view(1, -1)
                        l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                        if self.linear_layer:
                            l1_grads = torch.cat(
                                (l1_grads, batch_l1_grads), dim=0)

            torch.cuda.empty_cache()
            if self.linear_layer:
                self.val_grads_per_elem = torch.cat(
                    (l0_grads, l1_grads), dim=1)
            else:
                self.val_grads_per_elem = l0_grads
        return self.batch_scores

    def update_model(self, model_params):
        """
        Update the models parameters

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing models parameters
        """
        self.model.load_state_dict(model_params)
