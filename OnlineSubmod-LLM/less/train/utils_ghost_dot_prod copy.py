import torch
import numpy as np
import less
import time
from .submod import eps_greedy_composition, importance_sampling_batched


def find_topk_GClayers(module, topk_layers):
    assert topk_layers > 0
    GC_layers = []
    last_layers = [module.base_model.model.model.layers[-(i+1)] for i in range(topk_layers)]
    for layer in last_layers:
        GC_layers = GC_layers + find_GClayers(layer)
    return GC_layers

def find_bottomk_GClayers(module, bottomk_layers):
    assert bottomk_layers > 0
    GC_layers = []
    last_layers = [module.base_model.model.model.layers[i] for i in range(bottomk_layers)]
    for layer in last_layers:
        GC_layers = GC_layers + find_GClayers(layer)
    return GC_layers

def find_GClayers(module):

    GC_layers = []
    for layer_str in dir(module):
        layer = getattr(module, layer_str)
        if type(layer) in [less.layers.lora_layers.GCLoRALinear, less.layers.linear.GCLinear]:
            # print('Found GC Layer: {}'.format(layer_str))
            GC_layers.append( layer )

    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            GC_layers = GC_layers + find_GClayers(immediate_child_module)
            
    return GC_layers



# For GREATS, while we can save the fwd and bwd pass by using the one from ghost,
# it does not seem to be working with the current implementation of the gradient accumulation.
# So here we just do another fwd and bwd pass over the selected data points for clean implementation.
def compute_GradProd_GC_per_iter(model, device, batch_train, validation_loader, optimizer, trainable_layers, 
                               per_val=False, return_tracin_and_similarity=True, return_val_sim=False, grads_topk=-1):

    # Get first batch from validation loader
    batch_val = next(iter(validation_loader))
        

    # Get the batch size of the validation and training batches
    val_bs = batch_val['input_ids'].shape[0]
    train_bs = batch_train['input_ids'].shape[0]

    optimizer.zero_grad()

    # Get maximum sequence length from both batches
    max_seq_len = max(
        batch_train['input_ids'].shape[1],
        batch_val['input_ids'].shape[1]
    )

    # Pad training batch if needed
    if batch_train['input_ids'].shape[1] < max_seq_len:
        pad_length = max_seq_len - batch_train['input_ids'].shape[1]
        batch_train = {
            'input_ids': torch.nn.functional.pad(batch_train['input_ids'], (0, pad_length), value=0),
            'attention_mask': torch.nn.functional.pad(batch_train['attention_mask'], (0, pad_length), value=0),
            'labels': torch.nn.functional.pad(batch_train['labels'], (0, pad_length), value=-100)  # Use -100 for labels padding
        }

    # Pad validation batch if needed
    if batch_val['input_ids'].shape[1] < max_seq_len:
        pad_length = max_seq_len - batch_val['input_ids'].shape[1]
        batch_val = {
            'input_ids': torch.nn.functional.pad(batch_val['input_ids'], (0, pad_length), value=0),
            'attention_mask': torch.nn.functional.pad(batch_val['attention_mask'], (0, pad_length), value=0),
            'labels': torch.nn.functional.pad(batch_val['labels'], (0, pad_length), value=-100)  # Use -100 for labels padding
        }

    combined_inputs = {
        k: torch.cat([batch_train[k], batch_val[k]], dim=0) 
        for k in batch_train.keys() if k != 'labels'  # Exclude labels
    }
    combined_labels = combined_inputs["input_ids"]
    
    # Free memory from individual batches
    del batch_train, batch_val
    torch.cuda.empty_cache()

    outputs = model(**combined_inputs)
    logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
    
    # Compute per-sample losses manually
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")  # Use 'none' to get per-sample losses
    # Reshape logits and labels for loss computation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = combined_labels[..., 1:].contiguous()
    # Compute loss for each position
    per_position_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    # Reshape to [batch_size, seq_len-1]
    per_position_loss = per_position_loss.view(shift_labels.size())
    valid_token_mask = combined_inputs["attention_mask"][..., 1:]       # (B, L‑1) bool
    token_loss = per_position_loss * valid_token_mask                 # zero out pads
    loss = token_loss.sum(1) / valid_token_mask.sum(1).clamp(min=1)
    
    pre_acts = [layer.pre_activation for layer in trainable_layers]
    
    # Single backward pass using mean loss for gradient computation
    Z_grad = torch.autograd.grad(loss.mean(), pre_acts, retain_graph=False)

    dLdZ_a_train_lst = []
    dLdZ_a_val_lst = []
    
    for layer, zgrad in zip(trainable_layers, Z_grad):

        decompose_results = layer.pe_grad_gradcomp(zgrad, per_sample=True)

        # Pre-allocate lists with known size
        train_results = [None] * len(decompose_results)
        val_results = [None] * len(decompose_results)

        # Single loop with direct indexing
        for i, (dLdZ, a) in enumerate(decompose_results):
            # Use torch.split instead of slicing for better memory efficiency
            dLdZ_train, dLdZ_val = torch.split(dLdZ, [train_bs, dLdZ.size(0) - train_bs])
            a_train, a_val = torch.split(a, [train_bs, a.size(0) - train_bs])
            
            train_results[i] = (dLdZ_train, a_train)
            val_results[i] = (dLdZ_val, a_val)

        dLdZ_a_train_lst.extend(train_results)
        dLdZ_a_val_lst.extend(val_results)

    assert not (per_val and return_val_sim)
    no_mean_flag = per_val or return_val_sim
    # Compute Gradient Dot-Product between training and validation batches
    grad_dotproduct_score = np.zeros((train_bs, val_bs)) if no_mean_flag else np.zeros(train_bs)
    grad_cosine_score = np.zeros((train_bs, val_bs)) if no_mean_flag else np.zeros(train_bs)

    # Compute pairwise similarity between training samples
    if return_tracin_and_similarity:
        similarity_local_score = np.zeros((train_bs, train_bs))
        similarity_local_cos_score = np.zeros((train_bs, train_bs))

    assert len(dLdZ_a_train_lst) == len(dLdZ_a_val_lst)
    layers_done = 0
    assert grads_topk != 0
    for (dLdZ, a), (dLdZ_val, a_val) in zip(dLdZ_a_train_lst, dLdZ_a_val_lst):

        if per_val:
            dot_prod, cos_sim = grad_dotprod(dLdZ, a, dLdZ_val, a_val)
            grad_dotproduct_score += to_np(dot_prod)
            grad_cosine_score += to_np(cos_sim)
        elif return_val_sim:
            dot_prod, cos_sim = grad_dotprod(dLdZ, a, dLdZ_val, a_val)
            grad_dotproduct_score += to_np(dot_prod)
            grad_dotproduct_score_mean = to_np((dot_prod).mean(dim=1))
            grad_cosine_score += to_np(cos_sim)
        else:
            dot_prod, cos_sim = grad_dotprod(dLdZ, a, dLdZ_val, a_val)
            grad_dotproduct_score += to_np((dot_prod).mean(dim=1))
            grad_cosine_score += to_np((cos_sim).mean(dim=1))

        if return_tracin_and_similarity:
            dot_prod, cos_sim = grad_dotprod(dLdZ, a, dLdZ, a)
            similarity_local_score += to_np(dot_prod)
            similarity_local_cos_score += to_np(cos_sim)
            
        layers_done += 1
        # Compute grads for only topk layers
        if(layers_done == grads_topk): break
            
        if return_val_sim:
            dot_prod, cos_sim = grad_dotprod(dLdZ_val, a_val, dLdZ_val, a_val)
            val_local_score = to_np(dot_prod)
            val_local_cos_score = to_np(cos_sim)
            use_cosine = False
            assert grad_dotproduct_score.shape == (train_bs,val_bs)
            assert grad_cosine_score.shape == (train_bs,val_bs)
            
            assert similarity_local_score.shape == (train_bs,train_bs)
            assert similarity_local_cos_score.shape == (train_bs,train_bs)
            
            assert val_local_score.shape == (val_bs, val_bs)
            assert val_local_cos_score.shape == (val_bs, val_bs)
            
            if(use_cosine):
                return grad_dotproduct_score_mean, grad_dotproduct_score, similarity_local_score, val_local_score, grad_cosine_score, similarity_local_cos_score, val_local_cos_score
            return grad_dotproduct_score_mean, grad_dotproduct_score, similarity_local_score, val_local_score, grad_dotproduct_score, similarity_local_score, val_local_score
        del dLdZ, a, dLdZ_val, a_val
        torch.cuda.empty_cache()
        
    return grad_dotproduct_score, similarity_local_score



def compute_GradProd_onlinesubmod(model, device, batch_train, validation_loader, optimizer, trainable_layers, 
                               batch_val=None, grads_topk=-1, use_cosine=True):

    # Get first batch from validation loader
    if(batch_val is None):
        batch_val = next(iter(validation_loader))
        

    # Get the batch size of the validation and training batches
    val_bs = batch_val['input_ids'].shape[0]
    train_bs = batch_train['input_ids'].shape[0]

    optimizer.zero_grad()

    # Get maximum sequence length from both batches
    max_seq_len = max(
        batch_train['input_ids'].shape[1],
        batch_val['input_ids'].shape[1]
    )
    print("batch train", batch_train)

    # Pad training batch if needed
    if batch_train['input_ids'].shape[1] < max_seq_len:
        pad_length = max_seq_len - batch_train['input_ids'].shape[1]
        batch_train = {
            'input_ids': torch.nn.functional.pad(batch_train['input_ids'], (0, pad_length), value=0),
            'attention_mask': torch.nn.functional.pad(batch_train['attention_mask'], (0, pad_length), value=0),
            'labels': torch.nn.functional.pad(batch_train['labels'], (0, pad_length), value=-100)  # Use -100 for labels padding
        }

    # Pad validation batch if needed
    if batch_val['input_ids'].shape[1] < max_seq_len:
        pad_length = max_seq_len - batch_val['input_ids'].shape[1]
        batch_val = {
            'input_ids': torch.nn.functional.pad(batch_val['input_ids'], (0, pad_length), value=0),
            'attention_mask': torch.nn.functional.pad(batch_val['attention_mask'], (0, pad_length), value=0),
            'labels': torch.nn.functional.pad(batch_val['labels'], (0, pad_length), value=-100)  # Use -100 for labels padding
        }

    combined_inputs = {
        k: torch.cat([batch_train[k], batch_val[k]], dim=0) 
        for k in batch_train.keys() if k != 'labels'  # Exclude labels
    }
    combined_labels = combined_inputs["input_ids"]
    
    # Free memory from individual batches
    del batch_train, batch_val
    torch.cuda.empty_cache()

    outputs = model(**combined_inputs)
    logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
    
    # Compute per-sample losses manually
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")  # Use 'none' to get per-sample losses
    # Reshape logits and labels for loss computation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = combined_labels[..., 1:].contiguous()
    # Compute loss for each position
    per_position_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    # Reshape to [batch_size, seq_len-1]
    per_position_loss = per_position_loss.view(shift_labels.size())
    valid_token_mask = combined_inputs["attention_mask"][..., 1:]       # (B, L‑1) bool
    token_loss = per_position_loss * valid_token_mask                 # zero out pads
    loss = token_loss.sum(1) / valid_token_mask.sum(1).clamp(min=1)
    
    
    pre_acts = [layer.pre_activation for layer in trainable_layers]
    
    # Single backward pass using mean loss for gradient computation
    Z_grad = torch.autograd.grad(loss.mean(), pre_acts, retain_graph=False)

    dLdZ_a_train_lst = []
    dLdZ_a_val_lst = []
    
    for layer, zgrad in zip(trainable_layers, Z_grad):

        decompose_results = layer.pe_grad_gradcomp(zgrad, per_sample=True)

        # Pre-allocate lists with known size
        train_results = [None] * len(decompose_results)
        val_results = [None] * len(decompose_results)

        # Single loop with direct indexing
        for i, (dLdZ, a) in enumerate(decompose_results):
            # Use torch.split instead of slicing for better memory efficiency
            dLdZ_train, dLdZ_val = torch.split(dLdZ, [train_bs, dLdZ.size(0) - train_bs])
            a_train, a_val = torch.split(a, [train_bs, a.size(0) - train_bs])
            
            train_results[i] = (dLdZ_train, a_train)
            val_results[i] = (dLdZ_val, a_val)

        dLdZ_a_train_lst.extend(train_results)
        dLdZ_a_val_lst.extend(val_results)

    # Compute Gradient Dot-Product between training and validation batches
    grad_dotproduct_score = np.zeros((train_bs, val_bs))
    grad_cosine_score = np.zeros((train_bs, val_bs))

    similarity_local_score = np.zeros((train_bs, train_bs))
    similarity_local_cos_score = np.zeros((train_bs, train_bs))
    
    val_local_score = np.zeros((val_bs, val_bs))
    val_local_cos_score = np.zeros((val_bs, val_bs))
    

    assert len(dLdZ_a_train_lst) == len(dLdZ_a_val_lst)
    
    layers_done = 0
    assert grads_topk != 0

    for (dLdZ, a), (dLdZ_val, a_val) in zip(dLdZ_a_train_lst, dLdZ_a_val_lst):

        dot_prod, cos_sim = grad_dotprod(dLdZ, a, dLdZ_val, a_val)
        grad_dotproduct_score += to_np(dot_prod)
        grad_cosine_score += to_np(cos_sim)
        del dot_prod, cos_sim

        dot_prod, cos_sim = grad_dotprod(dLdZ, a, dLdZ, a)
        similarity_local_score += to_np(dot_prod)
        similarity_local_cos_score += to_np(cos_sim)
        del dot_prod, cos_sim
        
        dot_prod, cos_sim = grad_dotprod(dLdZ_val, a_val, dLdZ_val, a_val)
        val_local_score += to_np(dot_prod)
        val_local_cos_score += to_np(cos_sim)
        del dot_prod, cos_sim
        
        layers_done +=1
        if(layers_done == grads_topk): break
        
        del dLdZ, a, dLdZ_val, a_val
        torch.cuda.empty_cache()
        
        
    assert grad_dotproduct_score.shape == (train_bs,val_bs)
    assert grad_cosine_score.shape == (train_bs,val_bs)
    
    assert similarity_local_score.shape == (train_bs,train_bs)
    assert similarity_local_cos_score.shape == (train_bs,train_bs)
    
    assert val_local_score.shape == (val_bs, val_bs)
    assert val_local_cos_score.shape == (val_bs, val_bs)
     
    if(use_cosine):
        return grad_dotproduct_score, similarity_local_score, val_local_score, grad_cosine_score, similarity_local_cos_score, val_local_cos_score
    return grad_dotproduct_score, similarity_local_score, val_local_score, grad_dotproduct_score, similarity_local_score, val_local_score
        

def to_np(tensor):
    return (tensor.float()).cpu().detach().numpy()

def update_list(original, input_element):
    # Check if the input is a list
    if isinstance(input_element, list):
        # Concatenate with the original list
        return original + input_element
    else:
        # Append to the original list
        original.append(input_element)
        return original


def grad_dotprod(A1, B1, A2, B2) -> torch.Tensor:
    """Compute gradient sample norm for the weight matrix in a linear layer."""
    if A1.dim() == 2 and B1.dim() == 2:
        return grad_dotprod_non_sequential(A1, B1, A2, B2)
    elif A1.dim() == 3 and B1.dim() == 3:
        return grad_dotprod_sequential(A1, B1, A2, B2)
    else:
        raise ValueError(f"Unexpected input shape: {A1.size()}, grad_output shape: {B1.size()}")


def grad_dotprod_non_sequential(A1, B1, A2, B2):
    dot_prod_1 = torch.matmul(A1, A2.T)  # [batch, nval]
    dot_prod_2 = torch.matmul(B1, B2.T)  # [batch, nval]
    dot_prod = dot_prod_1 * dot_prod_2

    # Norms
    norm_A1 = torch.norm(A1, dim=1, keepdim=True)  # [batch, 1]
    norm_A2 = torch.norm(A2, dim=1, keepdim=True)  # [nval, 1]
    norm_B1 = torch.norm(B1, dim=1, keepdim=True)
    norm_B2 = torch.norm(B2, dim=1, keepdim=True)

    norm_prod_1 = torch.matmul(norm_A1, norm_A2.T)  # [batch, nval]
    norm_prod_2 = torch.matmul(norm_B1, norm_B2.T)  # [batch, nval]

    cosine_prod = dot_prod / (norm_prod_1 * norm_prod_2 + 1e-8)

    return dot_prod, cosine_prod



def grad_dotprod_sequential(A1, B1, A2, B2, chunk_size=1024):
    (b, t, p), (_, _, d) = A1.size(), B1.size()
    nval = A2.size(0)

    A = torch.bmm(B1.permute(0, 2, 1), A1).flatten(start_dim=1)  # [b, p*d]
    B = torch.bmm(B2.permute(0, 2, 1), A2).flatten(start_dim=1)  # [nval, p*d]

    dot_prod = torch.matmul(A, B.T)  # [b, nval]

    # Norms
    A_norm = torch.norm(A, dim=1, keepdim=True)  # [b, 1]
    B_norm = torch.norm(B, dim=1, keepdim=True)  # [nval, 1]
    norm_matrix = torch.matmul(A_norm, B_norm.T)  # [b, nval]

    cosine_prod = dot_prod / (norm_matrix + 1e-8)

    return dot_prod, cosine_prod

                

def _chunked_matmul(A1, A2, chunk_size=128):
    """
    Performs matrix multiplication in chunks for memory efficiency.

    Parameters:
    A1 (torch.Tensor): The first tensor with shape [n1, c1, h1, w1]
    A2 (torch.Tensor): The second tensor with shape [n2, c2, w2, h2]
    chunk_size (int): The size of each chunk to be multiplied

    Returns:
    torch.Tensor: The result of the matrix multiplication with shape [n1, c2, h1, h2]
    """
    # Validate input shapes
    if A1.shape[-1] != A2.shape[-2]:
        raise ValueError(f"Inner dimensions must match for matrix multiplication, got {A1.shape[-1]} and {A2.shape[-2]}")

    # Determine output shape
    n1, c1, h1, w1 = A1.shape
    n2, c2, w2, h2 = A2.shape

    if w1 != w2:
        raise ValueError(f"Inner matrix dimensions must agree, got {w1} and {w2}")

    # Prepare the result tensor on the same device as the inputs
    result = torch.zeros(n1, c2, h1, h2, device=A1.device, dtype=A1.dtype)

    # Perform the multiplication in chunks
    for start in range(0, w1, chunk_size):
        end = min(start + chunk_size, w1)
        A1_chunk = A1[:, :, :, start:end]  # [8, 1, 1024, chunk_size]
        A2_chunk = A2[:, :, start:end, :]  # [1, 8, chunk_size, 1024]

        # Multiply the chunks
        result += torch.matmul(A1_chunk, A2_chunk)

    return result


def greedy_selection(scores, interaction_matrix, K):
    """
    Select K data points based on the highest scores, dynamically updating scores
    by subtracting interactions with previously selected data points.

    Parameters:
    - scores: A numpy array of initial scores for each data point.
    - interaction_matrix: A numpy matrix of pairwise interactions between data points.
    - K: The number of data points to select.

    Returns:
    - selected_indices: Indices of the selected data points.
    """
    # Ensure scores is a mutable numpy array to update it in-place
    scores = scores.copy()
    selected_indices = []

    for _ in range(K):
        # Select the index with the highest score
        idx_max = np.argmax(scores)
        selected_indices.append(idx_max)

        # Update scores by subtracting interactions with the selected data point
        scores -= interaction_matrix[idx_max, :]

        # Set the score of the selected data point to a very large negative value
        # to ensure it's not selected again
        scores[idx_max] = -np.inf

    return selected_indices



def random_selection(scores, interaction_matrix, K):
    # Ensure scores is a mutable numpy array to update it in-place
    selected_indices = np.random.choice(len(scores), size=K, replace=False)
    return selected_indices



def submod_selection(scores, interaction_matrix, sijs, qsijs, qq_sijs, K, args, lr, step):
    mode, greedyList, best_arm = eps_greedy_composition(scores=scores, interaction_matrix=interaction_matrix, submod_sijs=sijs, 
                                                        query_sijs=qsijs,
                                                        query_query_sijs=qq_sijs,
                                                        lr=lr,
                                                        submod_budget=K, args=args, step=step, greedyOnly=False)
    submod_indices = [[arm[i][0] for i in range(len(arm))] for arm in greedyList]
    weights = [[arm[i][1] for i in range(len(arm))] for arm in greedyList]
    opt_indices = submod_indices[best_arm]
    opt_weights = weights[best_arm]
    # submod_weights = [greedyFinal[i][1] for i in range(len(greedyFinal))]
    selected_indices = opt_indices
    # print("selected indices", selected_indices)
    return selected_indices, opt_weights




