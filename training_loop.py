import torch
import torch.nn as nn
import time
import argparse

import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings
from utils.utils_for_model import compute_tsp_tour_length,compute_vrp_tour_length,generate_tsp_instance, generate_vrp_instance
#from augmentation import Augmentation
warnings.filterwarnings("ignore", category=UserWarning)



def train_model_with_knn(args,model_train,model_baseline,optimizer_model,scheduler_model,device,file,time_stamp):
    start_training_time = time.time()
    if args.problem == 'cvrp' or args.problem == 'sdvrp':
        capacity = args.CAPACITIES[args.nb_nodes]

    action_k = args.action_k
    state_k = args.state_k

    for epoch in range(0,args.nb_epochs):
        print(epoch)

        ###################
        # Train model for one epoch
        ###################
        start = time.time()
        model_train.train()
        #ratio = calculate_ratio(epoch, total_epoch, args.final_ratio)

        #_tqdm = tqdm(range(1,args.nb_batch_per_epoch+1))

        for _ in range(1,args.nb_batch_per_epoch+1):
            # generate a batch of random instances    
            if args.problem == 'tsp':
                x_aug,x_repeat = generate_tsp_instance(args,device)
                # compute tours for baseline
                with torch.no_grad():
                    tour_baseline, _= model_baseline(x_aug, action_k, state_k, choice_deterministic=True, if_use_local_mask =args.if_use_local_mask)
                L_baseline = compute_tsp_tour_length(x_repeat, tour_baseline)
                # compute tours for model
                tour_train_model,sumLogProbOfActions_model = model_train(x_aug, action_k, state_k, choice_deterministic=False, if_use_local_mask =args.if_use_local_mask)
                # get the lengths of the tours
                L_train_model = compute_tsp_tour_length(x_repeat, tour_train_model) # size(L_train)=(bsz)

            elif args.problem == 'cvrp' or args.problem == 'sdvrp':
                input_aug,x_repeat = generate_vrp_instance(args,device)
                with torch.no_grad():
                    tour_baseline, _ = model_baseline(input_aug,action_k,state_k,capacity,problem=args.problem,choice_deterministic=True, if_use_local_mask =args.if_use_local_mask)
                L_baseline = compute_vrp_tour_length(x_repeat, tour_baseline)
                # compute tours for model
                tour_train_model,sumLogProbOfActions_model = model_train(input_aug,action_k,state_k,capacity,problem=args.problem,choice_deterministic=False, if_use_local_mask =args.if_use_local_mask)
                # get the lengths of the tours
                L_train_model = compute_vrp_tour_length(x_repeat, tour_train_model) # size(L_train)=(bsz)
                
            # backprop
            loss_model = torch.mean( (L_train_model - L_baseline)* sumLogProbOfActions_model)
            optimizer_model.zero_grad()
            loss_model.backward()
            optimizer_model.step()
            
        
        time_one_epoch = time.time()-start
        time_tot = time.time()-start_training_time
        scheduler_model.step()    

        ###################
        # Evaluate train model and baseline on 10k random TSP instances
        ###################
        model_train.eval()

        mean_tour_length_comb0 = 0
        mean_tour_length_comb1 = 0


        for _ in range(0,args.nb_batch_eval):
            # generate a batch of random instances
            if args.problem == 'tsp':
                x_aug,x_repeat = generate_tsp_instance(args,device,if_test=True)
                # compute tours for baseline
                with torch.no_grad():
                    tour_comb0,_= model_baseline(x_aug, action_k, state_k, choice_deterministic=True, if_use_local_mask =args.if_use_local_mask)
                    tour_comb1,_ = model_train(x_aug, action_k, state_k, choice_deterministic=True, if_use_local_mask =args.if_use_local_mask)
                    L_comb0 = compute_tsp_tour_length(x_repeat, tour_comb0)
                    L_comb1 = compute_tsp_tour_length(x_repeat, tour_comb1)

            elif args.problem == 'cvrp' or args.problem == 'sdvrp':
                input_aug,x_repeat = generate_vrp_instance(args,device,if_test=True)
                with torch.no_grad():
                    tour_comb0,_ = model_baseline(input_aug,action_k,state_k,capacity,problem=args.problem,choice_deterministic=True, if_use_local_mask =args.if_use_local_mask)
                    tour_comb1,_ = model_train(input_aug,action_k,state_k,capacity,problem=args.problem,choice_deterministic=True, if_use_local_mask =args.if_use_local_mask)
                    L_comb0 = compute_vrp_tour_length(x_repeat, tour_comb0)
                    L_comb1 = compute_vrp_tour_length(x_repeat, tour_comb1)

            # get the lengths of the tours
            
            L_comb0 = L_comb0.view((int(args.bsz/args.test_aug_num),args.test_aug_num))
            L_comb0 = torch.min(L_comb0,dim=1).values
            mean_tour_length_comb0 += L_comb0.mean().item()
            
            L_comb1 = L_comb1.view((int(args.bsz/args.test_aug_num),args.test_aug_num))
            L_comb1 = torch.min(L_comb1,dim=1).values
            mean_tour_length_comb1 += L_comb1.mean().item()
        
        mean_tour_length_comb0 =  mean_tour_length_comb0/ args.nb_batch_eval
        mean_tour_length_comb1 =  mean_tour_length_comb1/ args.nb_batch_eval

        update_model = mean_tour_length_comb1+ args.tol<mean_tour_length_comb0

        if update_model:
            model_baseline.load_state_dict( model_train.state_dict() )

        
        # Print and save in txt file
        mystring_min = 'Epoch: {:d}, epoch time: {:.3f}min, tot time: {:.3f}day, L_base: {:.3f}, L_train: {:.3f}, update_model: {}.'.format(
            epoch, time_one_epoch/60, time_tot/86400, mean_tour_length_comb0,mean_tour_length_comb1,update_model) 
        print(mystring_min) # Comment if plot display
        file.write(mystring_min+'\n')
        
        # Saving checkpoint
        checkpoint_dir_model = os.path.join(args.data_path+'ckpt/'+args.problem+'/train/model/')
        if not os.path.exists(checkpoint_dir_model):
            os.makedirs(checkpoint_dir_model)
        torch.save({
            'epoch': epoch,
            'time': time_one_epoch,
            'tot_time': time_tot,
            'model_baseline': model_baseline.state_dict(),
            'model_train': model_train.state_dict(),
            'optimizer': optimizer_model.state_dict(),
            }, '{}.pkl'.format(checkpoint_dir_model + "checkpoint_" + time_stamp + "-n{}".format(args.nb_nodes) + "-gpu{}".format(args.gpu_id)))


