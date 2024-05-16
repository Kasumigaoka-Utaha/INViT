import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_cluster import knn
import argparse
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

### parser creation

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

    

def create_parser(config_dict):
    """
    The create_parser function takes a dictionary of configuration parameters and returns an argparse.ArgumentParser instance
    with the appropriate arguments added to it. The function also returns a DotDict instance with the same keys as the input
    dictionary, but with values set to their default values (as specified in config_dict). This is useful for accessing 
    configuration parameters by name instead of by index.
    
    :param config_dict: Create the parser
    :return: A tuple of the parser and a dotdict instance
    """
    parser = argparse.ArgumentParser(description='Auto-generated parser')
    config_instance = DotDict(**config_dict)

    for key, value in config_dict.items():
        arg_type = type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=value, help=f'{key} ({arg_type.__name__})')

    return parser, config_instance

def count_identical_elements(tensor1, tensor2):
    """
    Count the number of identical elements regardless of their position along dimension 1.

    Parameters:
    - tensor1 (torch.Tensor): First input tensor with shape (bsz, num_nodes).
    - tensor2 (torch.Tensor): Second input tensor with shape (bsz, num_nodes).

    Returns:
    - List[int]: Number of identical elements regardless of position on dimension 1 for each row.
    """
    # Ensure that tensors have the same shape

    identical_counts = []
    length_1 = tensor1.size(1)
    length_2 = tensor2.size(1)
    
    # Iterate over rows
    for i in range(tensor1.size(0)):
        # Find unique elements in the intersection of the two rows
        unique_elements = torch.unique(torch.cat((tensor1[i], tensor2[i])))
        
        # Count the number of elements in the intersection
        identical_count = length_1+length_2-len(unique_elements)
        
        identical_counts.append(identical_count)

    return identical_counts

### augmentation function

def Scale(X):
    """
    The Scale function takes in a batch of points and scales them to be between 0 and 1.
    It does this by translating the points so that the minimum x-value is at 0, 
    and then dividing all x-values by the maximum value. It does this for both dimensions.
    
    :param X: Store the data and the scale_method parameter is used to determine how to scale it
    :param scale_method: Decide whether to scale the data based on the boundary of all points or just
    :return: The scaled x and the ratio
    """
    B = X.size(0)
    SIZE = X.size(1)
    X = X - torch.reshape(torch.min(X,1).values,(B,1,2)).repeat(1,SIZE,1) # translate
    ratio_x = torch.reshape(torch.max(X[:,:,0], 1).values - torch.min(X[:,:,0], 1).values,(-1,1))
    ratio_y = torch.reshape(torch.max(X[:,:,1], 1).values - torch.min(X[:,:,1], 1).values,(-1,1))
    ratio = torch.max(torch.cat((ratio_x,ratio_y),1),1).values
    ratio[ratio==0] = 1
    X = X / (torch.reshape(ratio,(B,1,1)).repeat(1,SIZE,2))
    return X, ratio

def Scale_for_vrp(X,num):
    """
    The Scale function takes in a batch of points and scales them to be between 0 and 1.
    It does this by translating the points so that the minimum x-value is at 0, 
    and then dividing all x-values by the maximum value. It does this for both dimensions.
    
    :param X: Store the data and the scale_method parameter is used to determine how to scale it
    :param scale_method: Decide whether to scale the data based on the boundary of all points or just
    :return: The scaled x and the ratio
    """
    B = X.size(0)
    SIZE = X.size(1)
    graph = X[:,:num,:]
    min_values = torch.reshape(torch.min(graph,1).values,(B,1,2)).repeat(1,SIZE,1)
    X = X - min_values # translate
    ratio_x = torch.reshape(torch.max(graph[:,:,0], 1).values - torch.min(graph[:,:,0], 1).values,(-1,1))
    ratio_y = torch.reshape(torch.max(graph[:,:,1], 1).values - torch.min(graph[:,:,1], 1).values,(-1,1))
    ratio = torch.max(torch.cat((ratio_x,ratio_y),1),1).values
    ratio[ratio==0] = 1
    X = X / (torch.reshape(ratio,(B,1,1)).repeat(1,SIZE,2))
    X[ratio==0,:,:] = X[ratio==0,:,:]+min_values[ratio==0,:,:]
    return X, ratio

def Rotate_aug(X):
    """
    The Rotate_aug function takes in a batch of points and rotates them by a random angle.
    The function also scales the points to be between 0 and 1.
    
    :param X: Pass the input data to the function
    :return: The rotated point cloud and the ratio of the bounding box
    """
    device = X.device
    B = X.size(0)
    SIZE = X.size(1)
    Theta = torch.rand((B,1),device=device)* 2 * np.pi
    Theta = Theta.repeat(1,SIZE)
    tmp1 = torch.reshape(X[:,:,0]*torch.cos(Theta) - X[:,:,1]*torch.sin(Theta),(B,SIZE,1))
    tmp2 = torch.reshape(X[:,:,0]*torch.sin(Theta) + X[:,:,1]*torch.cos(Theta),(B,SIZE,1))
    X_out = torch.cat((tmp1, tmp2), dim=2)
    X_out += 10
    X_out, ratio = Scale(X_out)
    return X_out, ratio

def Reflect_aug(X):
    """
    The Reflect_aug function takes in a batch of points and performs the following operations:
        1. Rotate each point by a random angle between 0 and 2pi radians
        2. Reflect each point across the x-axis (i.e., multiply y coordinate by -2)
        3. Add 10 to all coordinates so that no points are negative anymore (this is for convenience)
        4. Scale all coordinates down to be between 0 and 1
    
    :param X: Pass the data points to the function
    :return: A reflected point cloud and a scale ratio
    """
    device = X.device
    B = X.size(0)
    SIZE = X.size(1)
    Theta = torch.rand((B,1),device=device)* 2 * np.pi
    Theta = Theta.repeat(1,SIZE)
    tmp1 = torch.reshape(X[:,:,0]*torch.cos(2*Theta) + X[:,:,1]*torch.sin(2*Theta),(B,SIZE,1))
    tmp2 = torch.reshape(X[:,:,0]*torch.sin(2*Theta) - X[:,:,1]*torch.cos(2*Theta),(B,SIZE,1))
    X_out = torch.cat((tmp1, tmp2), dim=2)
    X_out += 10
    X_out, ratio = Scale(X_out)
    return X_out, ratio

def mix_aug(X):
    """
    The mix_aug function takes in a batch of images and returns the same batch with half of them rotated and half reflected.
    The function also returns the ratio between the number of pixels that are black after augmentation to before augmentation.
    
    :param X: Pass in the data
    :return: The augmented images and the ratio of the number of augmented images to original ones
    """
    X_out = X.clone()
    X_out[0::2],ratio = Rotate_aug(X[0::2])
    X_out[1::2],ratio = Reflect_aug(X[1::2])
    return X_out,ratio

def run_aug(aug,x,aug_num=None,aug_all=False):
    """
    The run_aug function takes in an augmentation type, a batch of images, and two optional arguments.
    The first optional argument is the number of images to augment per batch. The second is whether or not to 
    augment all the images in the batch (defaults to False). It then returns a copy of x with some augmented 
    images inserted into it.
    
    :param aug: Select the augmentation to apply
    :param x: Pass in the data
    :param aug_num: Control the number of augmented images in each batch
    :param aug_all: Decide whether to apply the augmentation on all images or only a subset of them
    :return: A tensor with the same size as x, but with some of its values replaced by augmented data
    """
    x_clone = x.clone()
    if aug == 'rotate':
        x_out,_ = Rotate_aug(x)
    elif aug == 'reflect':
        x_out,_ = Reflect_aug(x)
    elif aug == 'mix':
        x_out,_ = mix_aug(x)
    elif aug == 'noise':
        x_out = x+torch.rand(x.size(), device=x.device)*1e-5
    else:
        x_out = x
    if not aug_all:
        if aug_num is not None:
            x_out[0::aug_num]=x_clone[0::aug_num]
        else:
            x_out[0]=x_clone[0]
    return x_out


### candidate-related
def calulate_mask_for_candidate(scores,threshold):
    """
    The calulate_mask_for_candidate function takes in a tensor of scores and a threshold.
    It returns a mask that is the same size as the input tensor, where each row has at most one True value.
    The True values are determined by taking the cumulative sum of each row, starting from column 0 to num_scores-2. 
    If any element in this cumulative sum is greater than or equal to threshold, then all elements after it will be masked out.
    
    :param scores: Calculate the mask
    :param threshold: Determine the number of tokens to be masked
    :return: A mask
    """
    bsz = scores.size(0)
    num_scores = scores.size(1)
    temp_scores = torch.zeros(scores.size(),device=scores.device)
    zero_to_bsz = torch.arange(bsz,)
    for i in range(num_scores):
        temp_scores[:,i:]+=scores[:,i:i+1]
    temp_scores = torch.cat((temp_scores,torch.ones((bsz,1),device=scores.device)),dim=1)
    temp_mask = temp_scores<threshold
    idx = torch.sum(temp_mask,dim=1)
    output_mask = ~temp_mask
    output_mask[zero_to_bsz,idx] = False
    output_mask = output_mask[:,:num_scores]
    return output_mask
    

### knn-related
def create_distance_mask_for_knn(last_visited_node,idx,graph,mask=None,ratio=10):
    """
    The create_distance_mask_for_knn function takes in the last visited node, the index of nodes to be considered for next visit,
    the graph and a mask. It returns a new mask which is an addition of the input mask and distance_mask. The distance_mask is 
    created by calculating distances between each node in idx with last visited node. If any of these distances are greater than 
    a threshold (which is calculated as ratio*distance from first element in idx to last visited node), then that particular entry 
    in distance_mask will be 1 else 0.
    
    :param last_visited_node: Calculate the distance between the last visited node and all other nodes
    :param idx: Specify the indices of the nodes that we want to consider for this step
    :param graph: Calculate the distance between nodes
    :param mask: Mask out the nodes that are already visited
    :param ratio: Control the distance threshold
    :return: A mask that is used to filter out nodes that are too far away from the last visited node
    """
    bsz,num_nodes = idx.size(0),idx.size(1)
    b_idx = torch.arange(0,bsz).repeat(num_nodes).sort()[0].to(graph.device)
    idx_for_ref = idx.view((bsz*num_nodes,))
    selected_graph = graph[b_idx,idx_for_ref].view((bsz,num_nodes,-1))
    distance_matrix = torch.sum( (last_visited_node - selected_graph)**2 , dim=2)**0.5
    threshold = (distance_matrix[:,0]*ratio).view((bsz,1)).repeat((1,num_nodes))
    distance_mask = (distance_matrix>threshold)
    if mask is not None:
        out_mask = mask+distance_mask
    else:
        out_mask = distance_mask
    return out_mask

### TSP-related

def compute_tsp_tour_length(x, tour): 
    """
    Compute the length of a batch of tours
    Inputs : x of size (bsz, nb_nodes, 2) batch of tsp tour instances
             tour of size (bsz, nb_nodes) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
    bsz = x.shape[0]
    nb_nodes = tour.shape[1]
    arange_vec = torch.arange(bsz, device=x.device)
    first_cities = x[arange_vec, tour[:,0], :] # size(first_cities)=(bsz,2)
    previous_cities = first_cities
    L = torch.zeros(bsz, device=x.device)
    with torch.no_grad():
        for i in range(1,nb_nodes):
            current_cities = x[arange_vec, tour[:,i], :] 
            L += torch.sum( (current_cities - previous_cities)**2 , dim=1 )**0.5 # dist(current, previous node) 
            previous_cities = current_cities
        L += torch.sum( (current_cities - first_cities)**2 , dim=1 )**0.5 # dist(last, first node)  
    return L


def Normalization_layer(X,num,problem='tsp'):
    """
    The Scale function takes in a batch of points and scales them to be between 0 and 1.
    It does this by translating the points so that the minimum x-value is at 0, 
    and then dividing all x-values by the maximum value. It does this for both dimensions.
    
    :param X: Store the data and the scale_method parameter is used to determine how to scale it
    :param scale_method: Decide whether to scale the data based on the boundary of all points or just
    :return: The scaled x and the ratio
    :doc-author: Trelent
    """
    if problem=='tsp':
        X,_ = Scale_for_vrp(X,num)
        X[X<0]=0
        X[X>1]=1
    else:
        X,_ = Scale_for_vrp(X,num)
    return X

### VRP-related

def is_vrp_finished(demands):
    return torch.sum(demands).item()==0



def compute_vrp_tour_length(x, tour): 
    """
    Compute the length of a batch of tours
    Inputs : x of size (bsz, nb_nodes, 2) batch of tsp tour instances
             tour of size (bsz, nb_nodes) batch of sequences (node indices) of tsp tours
    Output : L of size (bsz,)             batch of lengths of each tsp tour
    """
    bsz = x.shape[0]
    nb_nodes = tour.shape[1]
    arange_vec = torch.arange(bsz, device=x.device)
    depot = x[arange_vec, -1, :] # size(first_cities)=(bsz,2)
    first_cities = x[arange_vec, tour[:,0], :]
    previous_cities = first_cities
    L = torch.zeros(bsz, device=x.device)
    with torch.no_grad():
        for i in range(1,nb_nodes):
            current_cities = x[arange_vec, tour[:,i], :] 
            L += torch.sum( (current_cities - previous_cities)**2 , dim=1 )**0.5 # dist(current, previous node) 
            previous_cities = current_cities
        L += torch.sum( (current_cities - depot)**2 , dim=1 )**0.5 # dist(last, depot)  
        L += torch.sum( (first_cities - depot)**2 , dim=1 )**0.5 # dist(first, depot)  
    return L


def create_ref_matrix(x,bsz):
    """
    The create_ref_matrix function takes in a tensor of indices and the batch size,
    and returns a matrix with each row containing all the indices up to that index.
    For example, if x = [3,2] and bsz = 2 then create_ref_matrix(x) will return:
    [[0 1 2], 
 [0 1]]
    
    :param x: Pass the tensor of lengths to the function
    :param bsz: Determine the number of batches
    :return: A tensor of size (bsz*max_len)
    """
    matrix = torch.tensor([])
    for i in range(bsz):
        matrix = torch.cat((matrix,torch.arange(x[i].item())),dim=0)
    return matrix.long()


def get_knn_candidate(nodes,k,last_visited_node,last_visited_idx,mask=None):
    """
    The get_knn_candidate function takes in a batch of nodes, the number of neighbors to consider,
    the last visited node and its index. It returns the indices of the k nearest neighbors for each 
    node in the batch as well as a mask indicating which indices are valid (i.e., not equal to -100). 
    The function is used by get_knn_candidate_loss.
    
    :param nodes: Store the nodes in the graph
    :param k: Determine the number of nearest neighbors to be returned
    :param last_visited_node: Find the nearest neighbors of each node
    :param last_visited_idx: Store the last visited node index
    :param mask: Mask out the nodes that have been visited
    :return: The indices of the k nearest neighbors for each node in the batch
    """
    bsz = nodes.size(0)
    nb_nodes = nodes.size(1)
    b_one = torch.arange(0,bsz).to(nodes.device)
    b_nodes = torch.arange(0,bsz).repeat(nb_nodes).sort()[0].to(nodes.device)
    all_idx = torch.arange(0,nb_nodes).repeat((bsz,1)).to(nodes.device)
    last_visited_node = last_visited_node.squeeze()
    if mask is not None:
        remain_nodes = nodes[~mask]
        b_nodes = b_nodes.view((bsz,-1))
        remain_bsz = b_nodes[~mask]
        available_vec = torch.sum(~mask,dim=1)
        available_vec[available_vec>k] = k
        all_idx = all_idx[~mask]
    else:
        remain_nodes = nodes.view((bsz,nb_nodes,-1))
        remain_bsz = b_nodes.view((bsz,-1))
        available_vec = torch.ones(bsz)*k
        all_idx = all_idx.view((bsz*nb_nodes,))
    available_vec = available_vec.long()
    num_remain_nodes = remain_nodes.size(0)
    if num_remain_nodes == 0:
        output_idx = last_visited_idx.repeat(1,k).long()
        output_mask = torch.ones((bsz,k)).long().bool().to(nodes.device)
    else:
        knn_idx = knn(remain_nodes, last_visited_node, k, remain_bsz, b_one)
        ref_matrix = create_ref_matrix(available_vec,bsz)
        output_idx = last_visited_idx.repeat(1,k).long()
        node_bsz = knn_idx[0,:]
        idx = knn_idx[1,:]
        true_idx = all_idx[idx]
        output_idx[node_bsz,ref_matrix] = true_idx
        output_mask = torch.ones((bsz,k)).long().bool().to(nodes.device)
        output_mask[node_bsz,ref_matrix] = False
    return output_idx,output_mask

### instance generation
def generate_tsp_instance(args,device,if_test=False):
    """
    The generate_tsp_instance function generates a TSP instance.
    
    :param args: Pass the arguments from the command line to this function
    :param device: Specify the device on which to run the model
    :param if_test: Determine whether the data augmentation is used
    :return: The augmented data and the original data
    """
    if if_test:
        aug_num = args.test_aug_num
    else:
        aug_num = args.aug_num
    x = torch.rand(int(args.bsz/aug_num), args.nb_nodes, args.dim_input_nodes, device=device) # size(x)=(bsz, nb_nodes, dim_input_nodes) 
    x_repeat = x.unsqueeze(1).repeat((1,aug_num,1,1)).view((args.bsz,args.nb_nodes,args.dim_input_nodes))
    x_aug = run_aug(args.aug,x_repeat,aug_num)
    return x_aug, x_repeat

def generate_vrp_instance(args,device,if_test=False):
    """
    The generate_vrp_instance function generates a batch of VRP instances.
    
    :param args: Pass the arguments to the function
    :param device: Specify which device the data is loaded on
    :param if_test: Determine whether to use the test_aug_num or aug_num parameter
    :return: The input_aug and x_repeat
    """
    if if_test:
        aug_num = args.test_aug_num
    else:
        aug_num = args.aug_num
    x = torch.rand(int(args.bsz/aug_num), args.nb_nodes+1, args.dim_input_nodes, device=device) # size(x)=(bsz, nb_nodes, dim_input_nodes) 
    demand = torch.FloatTensor(int(args.bsz/aug_num),args.nb_nodes).uniform_(0, 9).long() + 1 
    demand = demand.to(device)
    x_repeat = x.unsqueeze(1).repeat((1,aug_num,1,1)).view((args.bsz,args.nb_nodes+1,args.dim_input_nodes))
    demand_repeat = demand.unsqueeze(1).repeat((1,aug_num,1)).view((args.bsz,args.nb_nodes))
    x_aug = run_aug(args.aug,x_repeat,aug_num)
    depot_aug = x_aug[:,-1,:] 
    nodes_aug = x_aug[:,0:-1,:] 
    input_aug = {'loc':nodes_aug,'demand':demand_repeat,'depot':depot_aug}
    return input_aug, x_repeat

### plot related 
def dist_stat(tsp_instances: torch.Tensor, tours: torch.Tensor):
    """
    selection statistics over ABSOLUTE distance
    :param tsp_instances: a batch of (bsz, size, 2) tensor
    :param tours: a batch of (bsz, size) tensor
    :return:
    """
    assert tsp_instances.dim() == 3
    assert tours.dim() == 2

    bsz, size, _ = tsp_instances.size()
    assert tours.size(0) == bsz
    assert tours.size(1) == size

    stats = []
    for i in range(bsz):
        tsp_instance = tsp_instances[i]
        tour = tours[i]

        starting_points = tour[:size-1]
        selected_points = tour[1:]

        per_instance_stat = torch.norm(tsp_instance[starting_points] - tsp_instance[selected_points], dim=1, p=2)
        stats.append(per_instance_stat)

    agg_stats = torch.cat(stats, dim=0)
    return agg_stats


def read_from_logs(args):
    log_name = args.data_path+'ckpt/'+args.problem+'/train/logs'+'/'+args.checkpoint_model +'.txt'

    # save the unchanged hyperparameters
    checkpoint_model = args.checkpoint_model
    nb_batch_per_epoch = args.nb_batch_per_epoch
    nb_batch_eval = args.nb_batch_eval
    nb_epochs = args.nb_epochs
    nb_nodes = args.nb_nodes
    bsz = args.bsz


    file = open(log_name,'r',1)
    line = file.readline()
    line = file.readline()
    line = file.readline()
    while line != '':
        split_line = line.split('=')
        key = split_line[0]
        if key in args:
            arg_type = type(args[key])
            if arg_type is bool:
                temp = split_line[1]
                split_str = temp.split('\n')[0]
                args[key] = True if split_str == 'True' else False
            elif arg_type is not str:
                args[key] = arg_type(split_line[1])
            else:
                temp = split_line[1]
                split_str = temp.split('\n')[0]
                args[key] = split_str
        line = file.readline()

    # load the unchanged hyperparameters
    args.checkpoint_model = checkpoint_model
    args.nb_batch_per_epoch = nb_batch_per_epoch
    args.nb_epochs = nb_epochs
    args.nb_nodes = nb_nodes
    args.nb_batch_eval = nb_batch_eval
    args.bsz = bsz


        
    