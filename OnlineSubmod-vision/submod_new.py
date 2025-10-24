import torch
from torch.func import functional_call, vmap, grad
import copy
import random
import math
from submod_grads import *
import time
device = "cuda"

def timed_execution(func, prefix=""):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"{prefix} Execution Time: {exec_time:.4f} seconds")
        return result  # Return whatever the original function returns
    return wrapper

def uniform_mixing_slow(l1, l2, lamb):
    n = len(l1)
    return [
            random.choice(l1) if random.random() < lamb else random.choice(l2)
            for _ in range(n)
        ]
    
def uniform_mixing_fast(l1, l2, lamb):
    n = len(l1)
    mask = np.random.rand(n) < lamb

    l1_random_choices = np.random.choice(l1, n)  # Pre-select random elements from l1
    l2_random_choices = np.random.choice(l2, n)  # Pre-select random elements from l2

    result = np.where(mask, l1_random_choices, l2_random_choices)

    return result.tolist()

def uniform_mixing(l1, l2, lamb, args):
    slow_mode = args["slow_mixing"]
    f = uniform_mixing_fast
    if(slow_mode): uniform_mixing_slow
    return f(l1, l2, lamb)

def importance_sampling_batched(submod_idxs, prev_opt_idxs, best_arm, args = None,
                                image_grads=None):
    sampling_mode = args["sampling_mode"]
    lamb = args["lamb"]
    opt_idxs = submod_idxs[best_arm]
    if(prev_opt_idxs is None):
        prev_opt_idxs = opt_idxs
        
    if(sampling_mode == "uniform"):
        new_opt = uniform_mixing(opt_idxs, prev_opt_idxs, lamb, args)
        
    elif(sampling_mode == "uniform_arm"):
        # Select an arm different from best_arm, thus mixing always happens
        exclusion_list = [i for i, arm in enumerate(submod_idxs) if i != best_arm]
        another_arm = random.choice(exclusion_list)
        new_opt= uniform_mixing(opt_idxs, submod_idxs[another_arm], lamb, args)
        
    elif(sampling_mode == "uniform_arm_noexclude"):
        # Mix only when new random arm is different from best arm, 
        # else dont mix and just return the optimal arm
        new_opt = opt_idxs
        another_arm = random.choice(range(len(submod_idxs)))
        if(another_arm != best_arm):
            new_opt= uniform_mixing(opt_idxs, submod_idxs[another_arm], lamb, args)
            
    elif(sampling_mode == "gradient_norm"):
        norms = torch.norm(image_grads, dim=-1)
        prev_norms, curr_norms = norms[torch.tensor(prev_opt_idxs)], norms[torch.tensor(opt_idxs)]
        frac = int(lamb*len(opt_idxs))
        curr_idxs = torch.topk(curr_norms, frac)[1].tolist()
        prev_idxs = torch.topk(prev_norms, len(opt_idxs) - frac)[1].tolist()
        new_opt = curr_idxs+prev_idxs
        
    elif(sampling_mode is None):
        new_opt = opt_idxs
    else:
        raise NotImplementedError
    return new_opt

def get_greedy_list(funcs, submod_budget, multi_thread=False, optimizer="NaiveGreedy"):
    greedyList = [None for i in range(len(funcs))]
    def submod_maximize(f, budget, optimizer):
        return f.maximize(budget = budget, optimizer=optimizer, 
                    stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, 
                    verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)
    for i,f in enumerate(funcs):
        # Maximize the function
        _greedy = submod_maximize(f, submod_budget, optimizer=optimizer)
        greedyList[i] = _greedy
    return greedyList

def get_lamb(lamb, lamb_mode, step):
    if(lamb_mode is None):
        return lamb
    if(lamb_mode == "exp1"):
        return math.exp(-step)
    if(lamb_mode == "exp2"):
        return 1-math.exp(-step) + 1e-1

def eps_greedy_composition_batched(model, testset, loss_fn, step, funcs, submod_budget, 
                                   moment_sum, args, val_sim, optimizer="StochasticGreedy", 
                                   greedyOnly=False, opt_grads=None, valloader=None, val_grads=None, train_grads=None, trainloader=None,**kwargs):
    lamb = args["lamb"]
    pi = args["pi"]
    # thresh = step/((step+lamb)**pi)
    thresh = step/((step+lamb)**pi)
    eps = torch.rand(1).item()
    dbg("eps thresh", eps, thresh, print_debug=args["print_debug"])
    greedyList = get_greedy_list(funcs, submod_budget, optimizer)
    if((eps > thresh) or greedyOnly):
        best_index = best_submod_bandit(model, greedyList, args["eta_n"], moment_sum,
                                          val_sim, opt_grads=opt_grads, trainloader=trainloader,loss_fn=loss_fn,
                                          val_grads=val_grads, train_grads=train_grads, testloader=valloader, args=args)
        return "greedy", greedyList, best_index
    else:
        sample = torch.randint(len(greedyList), ()).item()
        return "uniform", greedyList, sample
    

def best_submod_bandit(model, greedyList,eta_n, moment_sum,
                               val_sim, loss_fn=None, testloader=None, trainloader=None, opt_grads=None, val_grads=None, train_grads=None, args=None):
    best_index = 0
    if(train_grads is None):
        train_grads = calc_grads_features_perbatch(model, loss_fn, trainloader)
    if(val_grads is None):
        val_grads = calc_grads_features_perbatch(model, loss_fn, testloader)
    if val_grads is None or train_grads is None:
        raise
    
    indices_list = [[greedyList[i][j][0] for j in range(len(greedyList[i]))] for i in range(len(greedyList))]
    weights_list = [[greedyList[i][j][1] for j in range(len(greedyList[i]))] for i in range(len(greedyList))]

    metric_list = calc_metric(indices_list , eta_n, train_grads, val_grads, opt_grads, 
                              val_sim, weights_list=weights_list, moment_sum=moment_sum)
 
    max_index = torch.argmax(metric_list)
    best_metric = metric_list[max_index].item()
    best_index = max_index.item()
    dbg("best metric", best_index, best_metric, metric_list.shape, metric_list, print_debug=args["print_debug"] )
    
    return best_index


def calc_metric(indices_list,eta_n, imp_sample_grads, val_grads, opt_grads, reduction="mean", 
                val_sim="mean", weights_list=None, moment_sum=None):    
    indices_list = torch.tensor(indices_list)
    weights_list = torch.tensor(weights_list)
    
    val_grads_mat = val_grads
    if(val_sim == "mean"):
        val_grads_mat = torch.mean(val_grads, dim=0, keepdim=True)
    moment_sum_local =  moment_sum.mean(0) if moment_sum is not None else None
    def func(submod_indices, weights):
        # check again
        # print("weights shape", weights.shape, imp_sample_grads[submod_indices].shape, 
        #       (weights.unsqueeze(1).to("cuda")*imp_sample_grads[submod_indices]).shape )
        # mat1 = weights.unsqueeze(1).to("cuda")*imp_sample_grads[submod_indices]
        term1 = eta_n*imp_sample_grads[submod_indices]@(val_grads_mat.transpose(0,1)) # s_imp,s_val
        # term1, _ = torch.max(term1, dim=1, keepdim=True)
        if(opt_grads is None):
            term2 = 0
        else:
            grad_sum = torch.sum(opt_grads, dim=0, keepdim=True)
            if(moment_sum_local is not None):
                moment_sum_temp = moment_sum_local
                hessian = (moment_sum_temp)
            
            else: hessian = torch.eye(grad_sum.transpose(0,1).shape[0]).to("cuda")

            term2 = eta_n*eta_n*imp_sample_grads[submod_indices]@((hessian@(grad_sum.transpose(0,1)))) # B',1
        metric =  term1 - term2
        metric =  weights.unsqueeze(1).to("cuda")*metric
        return metric
    
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    with torch.autocast(device_type=device):
        metric_list = vmap(func, in_dims=(0,0))(indices_list,weights_list)
    if(reduction == "mean"):
        metric_list = torch.mean(metric_list, dim=1)
    
    return metric_list


# def get_new_idxs_batched(idxs, gammas, batch_size, budget_num_batches, trainloader):
#     print("****Refreshing****")
#     print("Lens", len(idxs))
#     batches_idxs = len(idxs)
#     diff = budget_num_batches - batches_idxs
#     print("diff2", diff, budget_num_batches, batches_idxs, len(set(idxs)))
#     if diff > 0:
#         print("Adding random batches", diff)
#         num_train = len(trainloader.dataset)
#         remainList = set(np.arange(num_train)).difference(set(idxs))
#         new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
#         prev_len = len(idxs)
#         idxs.extend(new_idxs)
#         gammas.extend([1 for _ in range(diff)])
#         print("Length delta", prev_len, len(idxs))
#     return idxs, gammas

