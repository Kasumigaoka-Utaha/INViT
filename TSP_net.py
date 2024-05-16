###################
# Libs
###################

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from torch_cluster import knn
import numpy as np
import warnings
from utils.utils_for_model import create_distance_mask_for_knn
from encoder import state_encoder_tsp, action_encoder_tsp
from decoder import Transformer_decoder_net

warnings.filterwarnings("ignore", category=UserWarning)




class TSP_net(nn.Module): 
    """
    The TSP network is composed of two steps :
      Step 1. Encoder step : Take a set of 2D points representing a fully connected graph 
                             and encode the set with self-transformer.
      Step 2. Decoder step : Build the TSP tour recursively/autoregressively, 
                             i.e. one node at a time, with a self-transformer and query-transformer. 
    Inputs : 
      x of size (bsz, nb_nodes, dim_emb) Euclidian coordinates of the nodes/cities
      deterministic is a boolean : If True the salesman will chose the city with highest probability. 
                                   If False the salesman will chose the city with Bernouilli sampling.
    Outputs : 
      tours of size (bsz, nb_nodes) : batch of tours, i.e. sequences of ordered cities 
                                      tours[b,t] contains the idx of the city visited at step t in batch b
      sumLogProbOfActions of size (bsz,) : batch of sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
    """
    
    def __init__(self, dim_input_nodes, dim_emb, dim_ff, num_state_encoder, nb_layers_state_encoder, nb_layers_action_encoder, nb_layers_decoder, nb_heads,   
                 batchnorm = True, if_agg_whole_graph = False):
        super(TSP_net, self).__init__()
        
        # basic info
        self.dim_input = dim_input_nodes
        self.dim_emb = dim_emb
        self.if_agg_whole_graph = if_agg_whole_graph
        self.num_state_encoder = num_state_encoder

        self.state_encoders = nn.ModuleList(
             [state_encoder_tsp(dim_input_nodes, dim_emb, dim_ff, nb_layers_state_encoder, nb_heads, batchnorm = batchnorm, if_agg_whole_graph =if_agg_whole_graph) 
             for _ in range(num_state_encoder)] )
        
        self.action_encoder = action_encoder_tsp(dim_input_nodes, dim_emb, dim_ff, nb_layers_action_encoder, nb_heads, batchnorm = batchnorm) 
        
        # decoder layer
        self.decoder = Transformer_decoder_net(dim_emb, nb_heads, nb_layers_decoder)
        self.WK_att_decoder = nn.Linear((num_state_encoder+1)*dim_emb, nb_layers_decoder* dim_emb) 
        self.WV_att_decoder = nn.Linear((num_state_encoder+1)*dim_emb, nb_layers_decoder* dim_emb)
        self.query_mlp = nn.Linear((2*num_state_encoder+1)*dim_emb, dim_emb)
        

    def load_pretrained_state_encoder(self,model,i):

        if i >= self.num_state_encoder:
            return

        self.state_encoders[i].load_state_dict(model.state_encoders[0].state_dict())
        
        for _, parameter in self.state_encoders[i].named_parameters():
            parameter.requires_grad = False


    def forward(self, x, action_k, state_k, choice_deterministic=False, if_use_local_mask = False):

        assert isinstance(state_k,list)
        assert isinstance(action_k,int)
        assert self.num_state_encoder == len(state_k)
        
        # Get info from input data
        bsz = x.shape[0]
        nb_nodes = x.shape[1]
        zero_to_bsz = torch.arange(bsz, device=x.device) # [0,1,...,bsz-1]

        # concat the nodes and the input placeholder that starts the decoding
        start_idx = torch.randint(nb_nodes,(bsz,)).to(x.device)

        ### list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours = []
        tours.append(start_idx)

        ### list that will contain Float tensors of shape (bsz,) that gives the neg log probs of the choices made at time t
        sumLogProbOfActions = []

        first_visited_node = x[zero_to_bsz,start_idx,:].view((bsz,1,-1))
        last_visited_node = first_visited_node.clone()
        
        num_nodes = nb_nodes
        mask_global = torch.ones((bsz, nb_nodes), device=x.device).bool()
        mask_global[zero_to_bsz,start_idx] = False
        all_idx = torch.arange(0,nb_nodes).repeat((bsz,1)).to(x.device)

        for t in range(nb_nodes-1):
            ### initial info
            unvisited_matrix = torch.reshape(all_idx[mask_global],(bsz,-1))
            num_nodes = unvisited_matrix.size(1)

            b_graph = torch.arange(0,bsz).repeat(num_nodes).sort()[0].to(x.device)
            unvisited_matrix_idx = unvisited_matrix.view((-1,))
            graph = x[b_graph,unvisited_matrix_idx]
            graph = graph.view((bsz,-1,self.dim_input))

            k_action = min(action_k,num_nodes)
            k_state = min(max(state_k),num_nodes) if self.num_state_encoder>0 else k_action
            graph_for_knn = graph.view((-1,self.dim_input))
            last_visited_node_for_knn = last_visited_node.view((-1,self.dim_input))
            knn_output = knn(graph_for_knn, last_visited_node_for_knn, k_state, b_graph, zero_to_bsz)
            knn_idx = knn_output[1,:]%num_nodes
            knn_idx = knn_idx.view((bsz,k_state)).contiguous()

            # action encoder
            action_idx = knn_idx[:,:k_action].contiguous()
            action_mask = None
            if if_use_local_mask:
                action_mask = create_distance_mask_for_knn(last_visited_node,action_idx,graph)
            emb_action = self.action_encoder(graph,action_idx,last_visited_node,mask=action_mask)
            emb_q = emb_action[:,k_action:(k_action+1),:]
            emb_other = emb_action[:,:k_action,:]
            
            # state encoder
            for i in range(self.num_state_encoder):
                temp_k = min(state_k[i],num_nodes)
                temp_idx = knn_idx[:,:temp_k].contiguous()
                emb_state = self.state_encoders[i](graph,temp_idx,last_visited_node,first_visited_node)
                emb_q = torch.cat((emb_q,emb_state[:,temp_k:(temp_k+1),:]),dim=2)
                emb_q = torch.cat((emb_q,emb_state[:,(temp_k+1):(temp_k+2),:]),dim=2)
                emb_other = torch.cat((emb_other,emb_state[:,:k_action,:]),dim=2)                


            action_idx_for_ref = action_idx.view((bsz*k_action,))
            b_action = torch.arange(0,bsz).repeat(k_action).sort()[0].to(x.device)
            next_idx = unvisited_matrix[b_action,action_idx_for_ref].view(bsz,-1) 
            mask_for_decoder = action_mask.bool() if action_mask is not None else None
           
            ### decoder
            # Q, K and V
            h_q = self.query_mlp(emb_q)
            K_att_decoder = self.WK_att_decoder(emb_other) # size(K_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
            V_att_decoder = self.WV_att_decoder(emb_other) # size(V_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
            # decode
            prob_next_node = self.decoder(h_q, K_att_decoder, V_att_decoder, mask_for_decoder)

            ### next node choosing
            # if not, which is the next node to be visited 
            if choice_deterministic: # greedy (exploit)
                idx = torch.argmax(prob_next_node, dim=1) 
            else: # random (explore)
                idx = Categorical(prob_next_node).sample() # size(query)=(bsz,)

            ### next node info
            next_idx = next_idx.view(bsz,-1) 
            last_visited_idx = next_idx[zero_to_bsz, idx] 
            last_visited_node = x[zero_to_bsz,last_visited_idx,:].view((bsz,1,-1))

            ### Update the current tour
            # probability for actions
            ProbOfChoices = prob_next_node[zero_to_bsz, idx]
            # update
            sumLogProbOfActions.append(torch.log(ProbOfChoices))
            tours.append(last_visited_idx)

            # Update mask
            mask_global[zero_to_bsz, last_visited_idx]=False


        # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
        sumLogProbOfActions = torch.stack(sumLogProbOfActions,dim=1).sum(dim=1) # size(sumLogProbOfActions)=(bsz,)

        # convert the list of nodes into a tensor of shape (bsz,num_cities)
        tours = torch.stack(tours,dim=1) # size(col_index)=(bsz, nb_nodes)

        return tours, sumLogProbOfActions