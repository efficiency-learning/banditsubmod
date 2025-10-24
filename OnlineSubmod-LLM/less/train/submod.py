import torch
from torch.func import functional_call, vmap, grad
import copy
import random
import math
import time
import numpy as np
import submodlib
from submodlib_cpp import ConcaveOverModular 
device = "cuda"
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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
                                image_grads=None, step_normed=None):
    sampling_mode = args["sampling_mode"]
    
    lamb = args["lamb_imp"]
    _lam = args["lamb"]
    if(lamb is None):
        lamb = get_lamb(_lam, args["lamb_mode"], step=step_normed)
    opt_idxs = submod_idxs[best_arm]
    if(sampling_mode is None):
        return opt_idxs
    if(prev_opt_idxs is None):
        print("***Setting prev_opt_idxs***")
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
        curr_idxs = torch.topk(curr_norms, frac)[1]
        curr_idxs = torch.tensor(opt_idxs, device=curr_idxs.device)[curr_idxs].tolist()
        prev_idxs = torch.topk(prev_norms, len(opt_idxs) - frac)[1]
        prev_idxs = torch.tensor(prev_opt_idxs, device=prev_idxs.device)[prev_idxs].tolist()
        
        new_opt = curr_idxs+prev_idxs

    else:
        raise NotImplementedError
    return new_opt

def get_greedy_list(funcs, submod_budget, optimizer="NaiveGreedy"):
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
        return math.exp(-0.5*step)
    if(lamb_mode == "exp2"):
        return max(min(1-math.exp(-0.5*step), 1), 1e-1)


def greats_greedy_selection(scores, interaction_matrix, K):
    scores = scores.copy()
    selected_indices = []

    for _ in range(K):
        idx_max = np.argmax(scores)
        selected_indices.append(idx_max)

        scores -= interaction_matrix[idx_max, :]
        scores[idx_max] = -np.inf

    return selected_indices



def get_funcs(n=None, metric="euclidean", sijs=None, qq_sijs=None, qsijs=None, lr=None):
    Q = qsijs.shape[1]
    # sijs += 1e-12
    # qsijs += 1e-12
    # qq_sijs += 1e-12
    funcs = [
        submodlib.FacilityLocationMutualInformationFunction(n, Q, data_sijs=sijs, query_sijs=qsijs),
        submodlib.GraphCutMutualInformationFunction(n, Q, query_sijs=qsijs),
        # submodlib.ConcaveOverModularFunction(n, Q, query_sijs=qsijs, queryDiversityEta=1, mode=ConcaveOverModular.inverse),
        submodlib.LogDeterminantMutualInformationFunction(n, Q, 0.2, data_sijs=sijs, query_sijs=qsijs, query_query_sijs=qq_sijs),
    ]
    
    # funcs = [
    #         submodlib.GraphCutFunction(n, mode="dense", metric=metric, num_neighbors=4, lambdaVal = 0.1,ggsijs=sijs, separate_rep=False),
    #         submodlib.LogDeterminantFunction(n, mode="dense", lambdaVal = 0.1,  metric=metric, num_neighbors=4, sijs=sijs),
    #         submodlib.FacilityLocationFunction(n, mode="dense", metric=metric, num_neighbors=8, sijs=sijs, separate_rep=False),
    #         submodlib.DisparityMinFunction(n = n, mode="dense", num_neighbors=8, sijs=sijs),
    #         submodlib.DisparitySumFunction(n = n, mode="dense", num_neighbors=8, sijs=sijs),
    #     ]
    return funcs

def eps_greedy_composition(scores, interaction_matrix, submod_sijs, query_sijs, query_query_sijs, step,  submod_budget, 
                                   args,  optimizer="NaiveGreedy", 
                                   lr=None,
                                   logs=None, step_normed=None, **kwargs):
    n = len(scores)
    sijs = submod_sijs
    qsijs = query_sijs
    qq_sijs = query_query_sijs
    # print("len scores", n, sijs.shape)
    greedyOnly = args["greedy_only"]
    uniformOnly = args["uniform_only"]
    funcs = get_funcs(n, sijs=sijs, qsijs=qsijs, qq_sijs=qq_sijs, lr=lr)
    # scores = torch.from_numpy(scores).to("cuda")
    # interaction_matrix = torch.from_numpy(interaction_matrix).to("cuda")
    # sijs = torch.from_numpy(sijs).to("cuda")
    
    lamb = args["lamb"]
    lamb = get_lamb(lamb, args["lamb_mode"], step_normed if step_normed is not None else step)
    pi = args["pi"]
    # thresh = step/((step+lamb)**pi)
    
    T = args["total_steps"]
    frac = args["imp_thresh_frac"]
    scaling  = ((2)**(1/pi))/(T*frac)
    step = (step+1)*scaling
    thresh = 1/(step**pi)
    
    eps = torch.rand(1).item()
    print("eps thresh", eps, thresh)
    greedyList = get_greedy_list(funcs, submod_budget, optimizer=optimizer)
    
    extra_arm = args["extra_arm"]
    if(extra_arm):
        n = int(submod_budget*len(scores))
        n = min(len(greedyList[0]), n)
        add_idx = greats_greedy_selection(scores, interaction_matrix, n)
        add_w = [1]*len(add_idx)
        extra = [[idx, w] for idx,w in zip(add_idx, add_w)]
        greedyList.append(extra)
        
    if(greedyOnly and uniformOnly): raise
    if((not uniformOnly and ((eps > thresh)) or greedyOnly)):
        best_index = best_submod_bandit(scores, interaction_matrix, greedyList,  args=args, logs=logs)
        return "greedy", greedyList, best_index
    else:
        sample = torch.randint(len(greedyList), ()).item()
        return "uniform", greedyList, sample

def best_submod_bandit(scores, interaction_matrix, greedyList, args=None, logs=None):
    best_index = 0
    indices_list = [[greedyList[i][j][0] for j in range(len(greedyList[i]))] for i in range(len(greedyList))]
    weights_list = [[greedyList[i][j][1] for j in range(len(greedyList[i]))] for i in range(len(greedyList))]
    metric_list = get_submod_utility(scores, interaction_matrix, indices_list, weights_list)
    
    max_index = np.argmax(metric_list)
    best_metric = metric_list[max_index].item()
    best_index = max_index.item()
    print("best metric", best_index, best_metric, metric_list)

    # if(args["print_debug"]):
    #     term1_term2 = mean_w_met[best_index]
    #     term1_term2 = (torch.tensor([1,-1], device=term1_term2.device)*term1_term2)
    #     print("best term1 term2", term1_term2)
    #     curr_epoch = logs["curr_epoch"]
    #     logs["metric"].append({
    #         "epoch": curr_epoch, 
    #         "metric": term1_term2.tolist(), 
    #         "total": metric_list[best_index].item()
    #     })
    
    
    return best_index


def get_submod_arm_utility(scores, interaction_matrix):
    scores = scores.copy()
    utility = 0
    K = len(scores)
    for i in range(K):
        # idx_max = np.argmax(scores)
        idx_max = i
        utility += scores[idx_max]
        scores -= interaction_matrix[idx_max, :]
        scores[idx_max] = -np.inf

    return utility


def get_submod_utility(scores, interaction_matrix, indices_list,  weights_list=None,):
    indices_list = np.array(indices_list)
    weights_list = np.array(weights_list)
    ARMS = len(indices_list)
    scores_subset = scores[indices_list] # shape: (k, idxs) or (k, idxs, val_bs)
    int_subset = np.stack([
        interaction_matrix[np.ix_(idxs, idxs)]
        for idxs in indices_list
    ])  # shape: (k, idxs, idxs)
    utility = [get_submod_arm_utility(scores_subset[i], int_subset[i]) for i in range(ARMS)]
    
    return utility

# def calc_metric(scores, indices_list,reduction="mean",  weights_list=None, ):    
#     indices_list = torch.tensor(indices_list)
#     weights_list = torch.tensor(weights_list)
    
#     with torch.autocast(device_type=device):
#         metric_list = scores[indices_list]  # shape (arms, k)
#         metric_list = metric_list.sum(dim=1)   # shape (arms,)
#     w_metric_list = weights_list.unsqueeze(-1).to("cuda")*metric_list
#     utility = w_metric_list.sum(-1, keepdim=True)
    
#     if(reduction == "mean"):
#         utility = torch.mean(utility, dim=1)
#         mean_w_met = torch.mean(w_metric_list, dim=1)
    
#     return utility, mean_w_met
