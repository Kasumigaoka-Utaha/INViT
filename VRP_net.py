import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
import numpy as np
from utils.utils_for_model import is_vrp_finished, get_knn_candidate, create_distance_mask_for_knn
from encoder import state_encoder_vrp, action_encoder_vrp
from decoder import Transformer_decoder_net




class VRP_net(nn.Module): 
    
    
    def __init__(self, dim_input_nodes, dim_emb, dim_ff, num_state_encoder, nb_layers_state_encoder,nb_layers_action_encoder, nb_layers_decoder, nb_heads,
                 batchnorm=True, if_agg_whole_graph = False):
        super(VRP_net, self).__init__()
        
        # basic info
        self.dim_input = dim_input_nodes
        self.dim_emb = dim_emb
        self.if_agg_whole_graph = if_agg_whole_graph
        self.num_state_encoder = num_state_encoder
        
        self.state_encoders = nn.ModuleList(
             [state_encoder_vrp(dim_input_nodes, dim_emb, dim_ff, nb_layers_state_encoder, nb_heads, batchnorm = batchnorm, if_agg_whole_graph =if_agg_whole_graph) 
             for _ in range(num_state_encoder)] )
        
        self.action_encoder = action_encoder_vrp(dim_input_nodes, dim_emb, dim_ff, nb_layers_action_encoder, nb_heads, batchnorm = batchnorm) 
        
        # decoder layer
        self.decoder = Transformer_decoder_net(dim_emb, nb_heads, nb_layers_decoder)
        self.WK_att_decoder = nn.Linear((num_state_encoder+1)*dim_emb, nb_layers_decoder* dim_emb) 
        self.WV_att_decoder = nn.Linear((num_state_encoder+1)*dim_emb, nb_layers_decoder* dim_emb)
        self.query_mlp = nn.Linear(2*(num_state_encoder+1)*dim_emb, dim_emb)

    def load_pretrained_state_encoder(self,model,i):

        if i >= self.num_state_encoder:
            return

        self.state_encoders[i].load_state_dict(model.state_encoders[0].state_dict())
        
        for name, parameter in self.state_encoders[i].named_parameters():
            parameter.requires_grad = False
        
    def forward(self, x, action_k, state_k, capacity, problem = 'cvrp', choice_deterministic=False, if_use_local_mask = False):

        """
        The forward function of the model.
        
        :param self: Represent the object itself
        :param x: Pass the input data to the forward function
        :param action_k: Specify the number of actions to be considered
        :param state_k: Determine the number of cities that are used as input to the state encoder
        :param capacity: Normalize the demand of each node
        :param problem: Specify the problem type
        :param choice_deterministic: Decide whether to use the greedy policy or not
        :param if_use_local_mask: Create a mask for the attention mechanism
        """
        assert isinstance(state_k,list)
        assert isinstance(action_k,int)
        assert self.num_state_encoder == len(state_k)

        # Get info from input data
        nodes = x['loc'] ## (bsz,nb_nodes,dim_input)
        true_demands = x['demand'] ## (bsz,nb_nodes)
        depot = x['depot'].unsqueeze(dim=1) ## (bsz,1,dim_input)
        

        # get basic parameters
        bsz = nodes.shape[0]
        nb_nodes = nodes.shape[1]
        zero_to_bsz = torch.arange(bsz, device=nodes.device) # [0,1,...,bsz-1]
        full_graph = torch.cat((nodes,depot),dim=1)
        true_demands = torch.cat((true_demands,torch.zeros((bsz,1)).long().to(nodes.device)),dim=1).detach()
        full_demands = true_demands/capacity

        ### list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours = []

        ### list that will contain Float tensors of shape (bsz,) that gives the neg log probs of the choices made at time t
        sumLogProbOfActions = []

        ### variables
        last_visited_node = depot
        depot_idx = torch.zeros((bsz,1)).long().to(nodes.device)-1
        last_visited_idx = depot_idx
        true_capacity_vec = capacity*torch.ones((bsz,1)).long().to(nodes.device)
        true_used_capacity_vec = torch.zeros((bsz,1)).long().to(nodes.device)
        b_a = torch.arange(0,bsz).view((-1,1)).repeat((1,action_k)).to(nodes.device)

        while not is_vrp_finished(full_demands):
            ### initial info
            demands = full_demands[:,:nb_nodes]
            remain_capacity_vec = (true_capacity_vec-true_used_capacity_vec)/capacity
            finished_mask = ~(demands>0)

            ### variation for different problems
            if problem == 'cvrp':
                available_action_mask = ~((demands>0)*(demands<remain_capacity_vec))
            elif problem == 'sdvrp':
                available_action_mask = ~(demands>0)
            else:
                break
            
            depot_bsz = (last_visited_idx.squeeze() == -1)*(torch.sum(demands,dim=1)!=0) # the depot can not be visited when last visited node is the depot and demand is not all finished

            #state_idx, state_mask = get_knn_candidate(nodes,k_state,last_visited_node,last_visited_idx,mask=finished_mask)
            #print(state_idx)
            action_idx, action_mask = get_knn_candidate(nodes,action_k,last_visited_node,last_visited_idx,mask=available_action_mask)
            
            if self.num_state_encoder>0:
                k_state = max(state_k)-action_k
                action_bsz = b_a[~action_mask]
                ref_idx = action_idx[~action_mask]
                mask_for_state = finished_mask.clone()
                mask_for_state[action_bsz,ref_idx]=True
                state_idx, state_mask = get_knn_candidate(nodes,k_state,last_visited_node,last_visited_idx,mask=mask_for_state)
                state_idx = torch.cat((action_idx,state_idx),dim=1)
                state_mask = torch.cat((action_mask,state_mask),dim=1)
            
            if if_use_local_mask:
                action_mask = create_distance_mask_for_knn(last_visited_node,action_idx,nodes,action_mask)
            action_idx_for_choice = torch.cat((action_idx,depot_idx),dim=1)

            # action encoder
            emb_action = self.action_encoder(nodes,action_idx,last_visited_node,depot,demands,remain_capacity_vec,encoder_mask=action_mask)
            emb_q = emb_action[:,action_k:(action_k+1),:]
            emb_q = torch.cat((emb_q,emb_action[:,(action_k+1):(action_k+2),:]),dim=2)
            emb_other = torch.cat((emb_action[:,:action_k,:],emb_action[:,(action_k+1):(action_k+2),:]),dim=1)
            
            # state encoder
            for i in range(self.num_state_encoder):
                temp_k = state_k[i]
                temp_idx = state_idx[:,:temp_k].contiguous()
                temp_mask = state_mask[:,:temp_k]
                emb_state = self.state_encoders[i](nodes,temp_idx,last_visited_node,depot,demands,remain_capacity_vec,finished_mask = finished_mask,encoder_mask=temp_mask)
                emb_q = torch.cat((emb_q,emb_state[:,temp_k:(temp_k+1),:]),dim=2)
                emb_q = torch.cat((emb_q,emb_state[:,(temp_k+1):(temp_k+2),:]),dim=2)
                temp_other = torch.cat((emb_state[:,:action_k,:],emb_state[:,(temp_k+1):(temp_k+2),:]),dim=1)
                emb_other = torch.cat((emb_other,temp_other),dim=2)

            ### decoder
            mask_for_decoder = torch.cat((action_mask,torch.zeros((bsz,1)).to(nodes.device)),dim=1).bool()
            mask_for_decoder[depot_bsz,-1] = True
            # concat the info from local encoder and global encoder
            # Q, K and V
            h_q = self.query_mlp(emb_q)
            K_att_decoder = self.WK_att_decoder(emb_other) # size(K_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
            V_att_decoder = self.WV_att_decoder(emb_other) # size(V_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
            # decode
            prob_next_node = self.decoder(h_q, K_att_decoder, V_att_decoder, mask_for_decoder)
            #print(prob_next_node)

            ### Next node choice
            if choice_deterministic: # greedy (exploit)
                idx = torch.argmax(prob_next_node, dim=1) 
            else: # random (explore)
                idx = Categorical(prob_next_node).sample() # size(query)=(bsz,)

            ### next node info
            next_node_idx = action_idx_for_choice[zero_to_bsz,idx]
            last_visited_node = full_graph[zero_to_bsz,next_node_idx].view((bsz,1,2))
            last_visited_idx = next_node_idx.view((bsz,1))

            ### Update demands and used capacity
            if problem == 'cvrp':
                last_visited_demand =  true_demands[zero_to_bsz,next_node_idx]
                new_used_capacity_vec = (last_visited_demand+true_used_capacity_vec.squeeze())*(next_node_idx!=-1)
                true_demands[zero_to_bsz,next_node_idx] = 0
            elif problem == 'sdvrp':
                last_visited_demand =  true_demands[zero_to_bsz,next_node_idx].unsqueeze(dim=1)
                true_remain_capacity_vec = true_capacity_vec-true_used_capacity_vec
                true_filled_demand = torch.min(torch.cat((last_visited_demand,true_remain_capacity_vec),dim=1),dim=1).values
                new_used_capacity_vec = (true_filled_demand+true_used_capacity_vec.squeeze())*(next_node_idx!=-1)
                true_demands[zero_to_bsz,next_node_idx] = true_demands[zero_to_bsz,next_node_idx]-true_filled_demand.long()
            else:
                break
            true_used_capacity_vec = new_used_capacity_vec.unsqueeze(dim=1)
            full_demands = true_demands/capacity

            ### Update the current tour
            # 2-step probability for actions
            ProbOfChoices = prob_next_node[zero_to_bsz, idx]
            # update
            sumLogProbOfActions.append(torch.log(ProbOfChoices))
            tours.append(next_node_idx)

        # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
        sumLogProbOfActions = torch.stack(sumLogProbOfActions,dim=1).sum(dim=1) # size(sumLogProbOfActions)=(bsz,)

        # convert the list of nodes into a tensor of shape (bsz,num_cities)
        tours = torch.stack(tours,dim=1) # size(col_index)=(bsz, nb_nodes)

        return tours, sumLogProbOfActions