import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad
import threading
import copy
import numpy as np
from generate_order import compute_dino_cls_image_embeddings, load_feature_model
from tqdm import tqdm

device = "cuda"
model = None
loss_fn = None
# DEBUG = True

def dbg(*args, print_debug=False, **kwargs):
    # global DEBUG
    if print_debug:
        print(*args, **kwargs)
        
def get_sims(a,b):
    return a@(b.transpose(0,1))

def get_random_images(testset, num_points, batch_size):
    images, labels = get_random_batch(testset, batch_size)
    rand = torch.randint(images.shape[0], (num_points,))
    val_images, val_labels = images[rand], labels[rand]
    return val_images, val_labels


def get_random_batch(testset, batch_size):
    images = testset[0]
    labels = testset[1]
    num_batches = images.shape[0]//batch_size
    rand = torch.randint(num_batches, ())
    a = rand*batch_size
    b = rand*batch_size+batch_size
    val_images, val_labels = images[a:b], labels[a:b]
    return val_images, val_labels

def cat(x, batch_size):
    return  torch.cat([x[key].view(batch_size, -1) for key in x.keys()], dim=1)


def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    out = functional_call(model, (params, buffers), (batch,))
    dbg("out", sample.shape, batch.shape, out.shape, "targets", targets.shape, targets)

    loss = loss_fn(out, targets)
    return loss

def calc_grads_features(model, loss_fn, trainloader, valloader):
    # global model, loss_fn
    # model = pmodel
    # cached_state_dict = copy.deepcopy(model.state_dict())
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    train_grads = calc_grads_features_perbatch(model, loss_fn, trainloader)
    val_grads = calc_grads_features_perbatch(model, loss_fn, valloader)
    # print("labels size", train_grads.shape, val_grads.shape)
    # torch.cuda.empty_cache()
    # model.load_state_dict(cached_state_dict)
    model.train()
    return train_grads, val_grads

def calc_grads_features_samplewise(model, loss_fn, train_images, train_labels, val_samples, val_batch_size):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    train_grads = calc_grads_last_layer(model, loss_fn, train_images, train_labels, take_mean=False)
    val_images, val_labels = get_random_images(testset=val_samples, num_points=1, batch_size=val_batch_size)
    val_grads = calc_grads_last_layer(model, loss_fn, val_images, val_labels, take_mean=False)
    model.train()
    return train_grads, val_grads

IMAGE_FEATURES_DINO = None
VAL_FEATURES_DINO = None
def calc_dino_features(trainloader, valloader, mean, std):
    global IMAGE_FEATURES_DINO, VAL_FEATURES_DINO
    if(IMAGE_FEATURES_DINO is None):
        train_features = calc_dino_features_perbatch(trainloader, mean, std)
        IMAGE_FEATURES_DINO = train_features
    else:
        train_features = IMAGE_FEATURES_DINO
    
    if(VAL_FEATURES_DINO is None):
        val_features = calc_dino_features_perbatch(valloader, mean, std)
        VAL_FEATURES_DINO = val_features
    else:
        val_features = VAL_FEATURES_DINO
        
    return train_features, val_features


def calc_grads_last_layer(model, loss_fn, images, labels, take_mean=True):
    if(loss_fn is None):
        loss_fn = torch.nn.CrossEntropyLoss()
    # with torch.set_grad_enabled(True):
    with torch.autocast(device_type="cuda"):
        '''
            This code is directly from CORDS
        '''
        out, _ = model(images, last=True, freeze=True)
        loss = loss_fn(out, labels)
        
        '''
            This is the what we should actually use to get final layer grads,
            but for uniformity(of comparing with baselines), we comment this out
        '''
        # out2 = model(inputs)
        # loss2 = loss_fn(out2, targets)
        # print("model", model.parameters())
        # grads = torch.autograd.grad(loss2, model.parameters() , retain_graph=True)[-1].unsqueeze(0)
        subset_grads = torch.autograd.grad(loss, out, retain_graph=False, create_graph=False)[0]
        if(take_mean):
            subset_grads = subset_grads.mean(dim=0, keepdim=True)
        
    return subset_grads


def calc_grads_features_perbatch(pmodel, ploss_fn, dataloader):
    perbatch_grads = []
    device = "cuda"
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        grads = calc_grads_last_layer(pmodel, ploss_fn, images, labels)
        
        perbatch_grads.append(grads)
        
    return torch.cat(perbatch_grads, dim=0)


def calc_dino_features_perbatch(dataloader, mean, std):
    perbatch_features = None
    device = "cuda"
    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
        return tensor * std + mean
    def emb(images):
        return compute_dino_cls_image_embeddings(images,device, 
                                                       return_tensor=True)
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        if images.dtype == torch.uint8:
            images = images.float()
        ch3Images = images
        # Convert the input tensor to FloatTensor if it's of type ByteTensor
        if images.shape[1] == 1:
            ch3Images = torch.cat([images,images,images], dim=1)
            # features = torch.cat([features,features,features], dim=1)
            # labels = torch.cat([labels,labels,labels], dim=1)
        ch3Images = denormalize(ch3Images, mean, std)
        
        features = emb(ch3Images)
        gMean = features.mean(0, keepdim=True)
        if perbatch_features is None:
            perbatch_features = gMean
        else:
            perbatch_features = torch.cat((perbatch_features, gMean), dim=0)
    
    return perbatch_features

@DeprecationWarning
def calc_grads_all_params(pmodel, ploss_fn, subset, testset, num_val_points):
    global model, loss_fn
    model = pmodel
    cached_state_dict = copy.deepcopy(model.state_dict())
    clone_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(clone_dict)
    model.eval()
    loss_fn = ploss_fn
    # Per-sample gradients
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    # print("keys", [k for k in params.keys()], [k for k in buffers.keys()])
    subset_grads = ft_compute_sample_grad(params, buffers, subset["images"], subset["labels"].to(device))
    val_images, val_labels = get_random_batch(testset, num_val_points)
    val_grads = ft_compute_sample_grad(params, buffers, val_images, val_labels)
    subset_size = subset["images"].shape[0]
    subset_grads = cat(subset_grads, subset_size) # B,P
    val_grads = cat(val_grads, val_images.shape[0]) # Bv,P
    
    # torch.cuda.empty_cache()
    model.load_state_dict(cached_state_dict)
    model.train()
    return subset_grads, val_grads

