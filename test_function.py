import torch
import torch.nn as nn
import time
import argparse

import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from pathlib import Path

import warnings
from load_data import load_instances_with_baselines
from utils.utils_for_model import run_aug, compute_vrp_tour_length, compute_tsp_tour_length
from utils.utilities import cvrplib_collections,tsplib_collections,get_dist_matrix,calculate_tour_length_by_dist_matrix,normalize_nodes_to_unit_board,avg_list,load_tsplib_file,load_cvrplib_file,choose_bsz,check_cvrp_solution_validity,parse_tsplib_name,parse_cvrplib_name
warnings.filterwarnings("ignore", category=UserWarning)


def run_tsplib_test_knn(model,action_k,state_k,path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(path)
    aug = 'mix'
    # main loop
    st1 = []
    st2 = []
    st3 = []
    st4 = []
    tsplib_names = list(tsplib_collections.keys())
    tsplib_names.sort(key=lambda x: parse_tsplib_name(x)[1])

    print(f"Start evaluation...")
    for i in range(len(tsplib_names)):
        name = tsplib_names[i]
        opt_len = tsplib_collections[name]
        _, size = parse_tsplib_name(name)

        # prepare env
        instance, _ = load_tsplib_file(root, name)
        dist_matrix = get_dist_matrix(instance).to(device)

        # normalize instance for tsplib
        normalized_instance = normalize_nodes_to_unit_board(instance)
        size = normalized_instance.size(0)
        bsz = choose_bsz(size)
        normalized_instance = torch.tensor(normalized_instance).float().to(device)
        normalized_instance = normalized_instance.unsqueeze(0)
        normalized_instance = normalized_instance.repeat((bsz,1,1))
        X = run_aug(aug,normalized_instance)
        with torch.no_grad():
            tour, _ = model(X, action_k, state_k, choice_deterministic=True)
        length_by_agent = compute_tsp_tour_length(normalized_instance,tour)
        idx = length_by_agent.min(dim=0).indices.item()
        best_tour = tour[idx,:]

        # evaluate tour length
        tour_len = calculate_tour_length_by_dist_matrix(dist_matrix, best_tour).item()
        tour_len = math.ceil(tour_len)
        gap = tour_len / opt_len - 1


        code = [name, size, tour_len, gap]
        if size <= 100:
            st1.append(code)
        elif size <= 1000:
            st2.append(code)
        elif size <= 10000:
            st3.append(code)
        else:
            st4.append(code)
        print(f"Instance {i:4d} {name:10}: model len {tour_len:.3f} to opt {opt_len:.3f} "
              f"-> gap {gap * 100:.3f}%.")

    # conclusion
    print(f"\n\n")
    print(f"TSP 1~100     : {len(st1)} instances, "
          f"gap {avg_list([x[3] for x in st1]) * 100:.3f}%")
    print(f"TSP 101~1000  : {len(st2)} instances, "
          f"gap {avg_list([x[3] for x in st2]) * 100:.3f}%")
    print(f"TSP 1001~10000: {len(st3)} instances, "
          f"gap {avg_list([x[3] for x in st3]) * 100:.3f}%")
    print(f"TSP >10000    : {len(st4)} instances, "
          f"gap {avg_list([x[3] for x in st4]) * 100:.3f}%")

def run_cvrplib_test_knn(model,action_k,state_k,path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(path)
    problem_type='cvrp'
    aug = 'mix'
    # main loop
    st1 = []
    st2 = []
    st3 = []
    st4 = []
    cvrplib_names = list(cvrplib_collections.keys())
    cvrplib_names.sort(key=lambda x: parse_cvrplib_name(x)[1])

    print(f"Start evaluation...")
    for i in range(len(cvrplib_names)):
        name = cvrplib_names[i]
        opt_len = cvrplib_collections[name]
        _, load_size = parse_cvrplib_name(name)

        # prepare env
        depot, nodes, demands, capacity, name = load_cvrplib_file(root, name)
        demands = demands.to(device)
        size = nodes.size(0)
        assert size == load_size
        depot_nodes = torch.cat((nodes, depot.unsqueeze(dim=0)), dim=0)
        dist_matrix = get_dist_matrix(depot_nodes).to(device)

        normalized_depot_nodes = normalize_nodes_to_unit_board(depot_nodes)
        size = nodes.size(0)
        bsz = choose_bsz(size)
        normalized_instance = torch.tensor(normalized_depot_nodes).float().to(device)
        normalized_instance = normalized_instance.unsqueeze(0)
        normalized_instance = normalized_instance.repeat((bsz,1,1))
        X = run_aug(aug,normalized_instance)
        depot_aug = X[:,-1,:] 
        nodes_aug = X[:,0:-1,:] 
        demand_repeat = demands.unsqueeze(dim=0).repeat((bsz,1))
        input_aug = {'loc':nodes_aug,'demand':demand_repeat,'depot':depot_aug}
        # model inference
        with torch.no_grad():
            tour, _ = model(input_aug, action_k, state_k, capacity, problem_type, choice_deterministic=True)
        length_by_agent = compute_vrp_tour_length(normalized_instance,tour)
        idx = length_by_agent.min(dim=0).indices.item()
        best_tour = tour[idx,:]

        if not check_cvrp_solution_validity(best_tour, demands, size, capacity):
            print(f"Instance {i:4d} {name:10}: Failed to be solved!")
            continue

        # evaluate tour length
        tour_len = calculate_tour_length_by_dist_matrix(dist_matrix, best_tour).item()
        tour_len = math.ceil(tour_len)
        gap = tour_len / opt_len - 1

        code = [name, size, tour_len, gap]
        if size <= 100:
            st1.append(code)
        elif size <= 200:
            st2.append(code)
        elif size <= 500:
            st3.append(code)
        else:
            st4.append(code)

        print(f"Instance {i:4d} {name:10}: model len {tour_len:.3f} to opt {opt_len:.3f} "
              f"-> gap {gap * 100:.3f}%.")

    # conclusion
    print(f"\n\n")
    print(f"CVRP 1~100     : {len(st1)} instances, "
          f"gap {avg_list([x[3] for x in st1]) * 100:.3f}%")
    print(f"CVRP 101~200  : {len(st2)} instances, "
          f"gap {avg_list([x[3] for x in st2]) * 100:.3f}%")
    print(f"CVRP 201~500: {len(st3)} instances, "
          f"gap {avg_list([x[3] for x in st3]) * 100:.3f}%")
    print(f"CVRP 501~1000    : {len(st4)} instances, "
          f"gap {avg_list([x[3] for x in st4]) * 100:.3f}%")

def run_tsp_test_knn(local_k,global_k,aug,model,if_use_local_mask,sizes,bszs,data_path,device,file,distributions,num_instance=None,if_aug=True):
    problem_type = 'tsp'
    for distribution in distributions:
        for i in range(len(sizes)):
            tsp_instances, _, opt_lens = load_instances_with_baselines(data_path, problem_type, sizes[i], distribution)
            opt_lens = torch.tensor(opt_lens).to(device)
            model_length = []
            num = tsp_instances.size(0)
            num = num_instance if isinstance(num_instance,int) else num_instance[i]
            for j in range(num):
                instance = tsp_instances[j]
                #instance_for_plot = instance.clone()
                instance = torch.tensor(instance).float().to(device)
                instance = instance.unsqueeze(0)
                instance = instance.repeat((bszs[i],1,1))
                if if_aug:
                    X = run_aug(aug,instance)
                else:
                    X = instance
                with torch.no_grad():
                    tour, _ = model(X, local_k, global_k, choice_deterministic=True, if_use_local_mask=if_use_local_mask)
                length_by_agent = compute_tsp_tour_length(instance,tour)
                value = length_by_agent.min(dim=0).values.item()
                #idx = length_by_agent.min(dim=0).indices.item()
                #best_tour = tour[idx,:]
                #output_path = data_path+'figures/tour/tsp-size-'+str(sizes[i])+'-'+distribution+'-instance-'+str(j)+'.png'
                #plot_tour(instance_for_plot, best_tour, output_path)
                model_length.append(value)
                info = 'For '+distribution+'-'+problem_type+'-{:d} {:d}-th instance, gap is {:.3f}%.'.format(
                    sizes[i], j, 100*(value-opt_lens[j].item())/opt_lens[j].item()) 
                print(info)

            model_length = torch.tensor(model_length).to(device)
            gap = (model_length-opt_lens[0:num])/opt_lens[0:num]
            out_string = 'For '+distribution+'-tsp-{:d}, mean gap is {:.3f}%, min gap is {:.3f}%, max gap is {:.3f}%, std is {:.3f}.'.format(
                sizes[i], 100*gap.mean(dim=0).item(), 100*gap.min(dim=0).values.item(),100*gap.max(dim=0).values.item(),gap.std(dim=0).item()) 
            file.write(out_string+'\n')


def run_vrp_test_knn(local_k,global_k,aug,model,if_use_local_mask,sizes,bszs,data_path,device,file,distributions,num_instance=None,if_aug=True):
    problem_type = 'cvrp'
    for distribution in distributions:
        for i in range(len(sizes)):
            cvrp_instances, _, opt_lens = load_instances_with_baselines(data_path, problem_type, sizes[i], distribution)
            depot, nodes, demands, capacity = cvrp_instances
            instances = torch.cat((nodes,depot.unsqueeze(dim=1)),dim=1).to(device)
            demands = demands.to(device)
            capacity = capacity.to(device)
            opt_lens = opt_lens.to(device)
            model_length = []
            num = num_instance if isinstance(num_instance,int) else num_instance[i]
            for j in range(num):
                instance = instances[j,:,:].to(device)
                instance = instance.unsqueeze(0)
                instance = instance.repeat((bszs[i],1,1))
                cap = capacity[j].item()
                if if_aug:
                    X = run_aug(aug,instance)
                else:
                    X = instance
                depot_aug = X[:,-1,:] 
                nodes_aug = X[:,0:-1,:] 
                demand_repeat = demands[j,:].unsqueeze(dim=0).repeat((bszs[i],1))
                input_aug = {'loc':nodes_aug,'demand':demand_repeat,'depot':depot_aug}
                with torch.no_grad():
                    tour, _ = model(input_aug, local_k, global_k, cap, problem_type, choice_deterministic=True, if_use_local_mask=if_use_local_mask)
                length_by_agent = compute_vrp_tour_length(instance,tour)
                value = length_by_agent.min(dim=0).values.item()
                #idx = length_by_agent.min(dim=0).indices.item()
                #best_tour = tour[idx,:]
                model_length.append(value)
                info = 'For '+distribution+'-'+problem_type+'-{:d} {:d}-th instance, gap is {:.3f}%.'.format(
                    sizes[i], j, 100*(value-opt_lens[j].item())/opt_lens[j].item()) 
                print(info)
            model_length = torch.tensor(model_length).to(device)
            gap = (model_length-opt_lens[0:num])/opt_lens[0:num]
            gap = torch.tensor(gap).to(device)
            out_string = 'For '+distribution+'-'+problem_type+'-{:d}, model provides solution with mean gap is {:.3f}%, min gap is {:.3f}%, max gap is {:.3f}%, std is {:.3f}.'.format(
                sizes[i], 100*gap.mean(dim=0).item(), 100*gap.min(dim=0).values.item(),100*gap.max(dim=0).values.item(),gap.std(dim=0).item())
            print(out_string)  
            file.write(out_string)
            file.write('\n')
