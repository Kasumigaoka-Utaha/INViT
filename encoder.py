import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import MultiHeadAttention
from utils.utils_for_model import Normalization_layer,run_aug

class Transformer_encoder_net(nn.Module):
    """
    Encoder network based on self-attention transformer
    Inputs :  
      h of size      (bsz, nb_nodes+1, dim_emb)    batch of input cities
    Outputs :  
      h of size      (bsz, nb_nodes+1, dim_emb)    batch of encoded cities
      score of size  (bsz, nb_nodes+1, nb_nodes+1) batch of attention scores
    """
    def __init__(self, nb_layers, dim_emb, nb_heads, dim_ff, batchnorm):
        super(Transformer_encoder_net, self).__init__()
        assert dim_emb == nb_heads* (dim_emb//nb_heads) # check if dim_emb is divisible by nb_heads
        self.MHA_layers = nn.ModuleList( [MultiHeadAttention(nb_heads,dim_emb,dim_emb,dim_emb) for _ in range(nb_layers)] )
        self.linear1_layers = nn.ModuleList( [nn.Linear(dim_emb, dim_ff) for _ in range(nb_layers)] )
        self.linear2_layers = nn.ModuleList( [nn.Linear(dim_ff, dim_emb) for _ in range(nb_layers)] )   
        if batchnorm:
            self.norm1_layers = nn.ModuleList( [nn.BatchNorm1d(dim_emb) for _ in range(nb_layers)] )
            self.norm2_layers = nn.ModuleList( [nn.BatchNorm1d(dim_emb) for _ in range(nb_layers)] )
        else:
            self.norm1_layers = nn.ModuleList( [nn.LayerNorm(dim_emb) for _ in range(nb_layers)] )
            self.norm2_layers = nn.ModuleList( [nn.LayerNorm(dim_emb) for _ in range(nb_layers)] )
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.batchnorm = batchnorm
        
    def forward(self, h, mask=None):      
        # PyTorch nn.MultiheadAttention requires input size (seq_len, bsz, dim_emb) 
        h = h.transpose(0,1) # size(h)=(nb_nodes, bsz, dim_emb)  
        # L layers
        for i in range(self.nb_layers):
            h_rc = h # residual connection, size(h_rc)=(nb_nodes, bsz, dim_emb)
            h = h.transpose(0,1)
            h, score = self.MHA_layers[i](h, h, h, mask) # size(h)=(nb_nodes, bsz, dim_emb), size(score)=(bsz, nb_nodes, nb_nodes)
            h = h.transpose(0,1)
            # add residual connection
            h = h_rc + h # size(h)=(nb_nodes, bsz, dim_emb)
            if self.batchnorm:
                # Pytorch nn.BatchNorm1d requires input size (bsz, dim, seq_len)
                h = h.permute(1,2,0).contiguous() # size(h)=(bsz, dim_emb, nb_nodes)
                h = self.norm1_layers[i](h)       # size(h)=(bsz, dim_emb, nb_nodes)
                h = h.permute(2,0,1).contiguous() # size(h)=(nb_nodes, bsz, dim_emb)
            else:
                h = self.norm1_layers[i](h)       # size(h)=(nb_nodes, bsz, dim_emb) 
            # feedforward
            h_rc = h # residual connection
            h = self.linear2_layers[i](torch.relu(self.linear1_layers[i](h)))
            h = h_rc + h # size(h)=(nb_nodes, bsz, dim_emb)
            if self.batchnorm:
                h = h.permute(1,2,0).contiguous() # size(h)=(bsz, dim_emb, nb_nodes)
                h = self.norm2_layers[i](h)       # size(h)=(bsz, dim_emb, nb_nodes)
                h = h.permute(2,0,1).contiguous() # size(h)=(nb_nodes, bsz, dim_emb)
            else:
                h = self.norm2_layers[i](h) # size(h)=(nb_nodes, bsz, dim_emb)
        # Transpose h
        h = h.transpose(0,1) # size(h)=(bsz, nb_nodes, dim_emb)
        return h, score


class state_encoder_tsp(nn.Module):

    def __init__(self, dim_input_nodes, dim_emb, dim_ff, nb_layers_encoder, nb_heads, batchnorm = True, if_agg_whole_graph = False):
        super(state_encoder_tsp, self).__init__()
        self.input_emb = nn.Linear(dim_input_nodes, dim_emb)
        self.input_emb_for_last = nn.Linear(dim_input_nodes, dim_emb)
        self.input_emb_for_first = nn.Linear(dim_input_nodes, dim_emb)
        self.if_agg_whole_graph = if_agg_whole_graph
        if if_agg_whole_graph:
            self.agg_emb = nn.Linear(dim_input_nodes, dim_emb)

        # encoder layer
        self.encoder = Transformer_encoder_net(nb_layers_encoder, dim_emb, nb_heads, dim_ff, batchnorm)

    def forward(self,graph,idx,last_visited_node,first_visited_node,mask=None):
        bsz = idx.size(0)
        k = idx.size(1)
        idx_for_ref = idx.view((bsz*k,))
        b_k = torch.arange(0,bsz).repeat(k).sort()[0].to(graph.device)
        node_group = graph[b_k,idx_for_ref].view((bsz,k,-1))
        node_group = torch.cat((node_group,last_visited_node),dim=1)
        node_group = torch.cat((node_group,first_visited_node),dim=1)
        if self.if_agg_whole_graph:
            if mask is not None:
                mask = torch.cat((mask,torch.zeros((bsz,1,1),device=graph.device)),dim=2)
            agg_node = torch.mean(graph,dim=1).view((bsz,1,-1))
            node_group = torch.cat((node_group,agg_node),dim=1)
        
        #scaled_node_group = node_group
        scaled_node_group = Normalization_layer(node_group,k+1)
        # init embedding
        scaled_last = scaled_node_group[:,k:(k+1),:]
        scaled_first = scaled_node_group[:,(k+1):(k+2),:]
        scaled_remain = scaled_node_group[:,:k,:]
        emb_last = self.input_emb_for_last(scaled_last)
        emb_first = self.input_emb_for_first(scaled_first)
        emb_remain = self.input_emb(scaled_remain)
        emb_input = torch.cat((emb_remain,emb_last),dim=1)
        emb_input = torch.cat((emb_input,emb_first),dim=1)
        if self.if_agg_whole_graph:
            scaled_agg = scaled_node_group[:,(k+2):(k+3),:]
            emb_agg = self.agg_emb(scaled_agg)
            emb_input = torch.cat((emb_input,emb_agg),dim=1)
        
        # encoding
        emb_out,_ = self.encoder(emb_input,mask)
        return emb_out
        
class action_encoder_tsp(nn.Module):

    def __init__(self, dim_input_nodes, dim_emb, dim_ff, nb_layers_encoder, nb_heads, batchnorm = True):
        super(action_encoder_tsp, self).__init__()
        self.input_emb = nn.Linear(dim_input_nodes, dim_emb)
        self.input_emb_for_last = nn.Linear(dim_input_nodes, dim_emb)

        # encoder layer
        self.encoder = Transformer_encoder_net(nb_layers_encoder, dim_emb, nb_heads, dim_ff, batchnorm)

    def forward(self,graph,idx,last_visited_node,mask=None):
        bsz = idx.size(0)
        k = idx.size(1)
        idx_for_ref = idx.view((bsz*k,))
        b_k = torch.arange(0,bsz).repeat(k).sort()[0].to(graph.device)
        node_group = graph[b_k,idx_for_ref].view((bsz,k,-1))
        mask_for_encoder = None
        if mask is not None:
            copied_last_visited_node = last_visited_node.repeat((1,k,1))
            node_group[mask] =  copied_last_visited_node[mask]
            mask_for_encoder = torch.cat((mask,torch.zeros((bsz,1)).to(graph.device)),dim=1).view((bsz,1,-1)).bool()
        node_group = torch.cat((node_group,last_visited_node),dim=1)

        #scaled_node_group = node_group
        scaled_node_group = Normalization_layer(node_group,k+1)
        # init embedding
        scaled_last = scaled_node_group[:,k:(k+1),:]
        scaled_remain = scaled_node_group[:,:k,:]
        emb_last = self.input_emb_for_last(scaled_last)
        emb_remain = self.input_emb(scaled_remain)
        emb_input = torch.cat((emb_remain,emb_last),dim=1)
        
        # encoding
        emb_out,_ = self.encoder(emb_input,mask_for_encoder)
        return emb_out


class state_encoder_vrp(nn.Module):

    def __init__(self, dim_input_nodes, dim_emb, dim_ff, nb_layers_encoder, nb_heads, batchnorm = True, if_agg_whole_graph = False):
        super(state_encoder_vrp, self).__init__()
        self.input_emb = nn.Linear(dim_input_nodes+1, dim_emb)
        self.input_emb_for_last = nn.Linear(dim_input_nodes+1, dim_emb)
        self.input_emb_for_first = nn.Linear(dim_input_nodes, dim_emb)
        self.if_agg_whole_graph = if_agg_whole_graph
        if if_agg_whole_graph:
            self.agg_emb = nn.Linear(dim_input_nodes+1, dim_emb)

        # encoder layer
        self.encoder = Transformer_encoder_net(nb_layers_encoder, dim_emb, nb_heads, dim_ff, batchnorm)

    def forward(self,graph,idx,last_visited_node,first_visited_node,demands, remain_capacity_vec, finished_mask = None, encoder_mask=None, depot_mask=None):
        bsz = idx.size(0)
        k = idx.size(1)
        nb_nodes = graph.size(1)
        idx_for_ref = idx.view((bsz*k,))
        b_k = torch.arange(0,bsz).repeat(k).sort()[0].to(graph.device)
        node_group = graph[b_k,idx_for_ref].view((bsz,k,-1))
        demands_group = demands[b_k,idx_for_ref].view((bsz,k))
        node_group = torch.cat((node_group,last_visited_node),dim=1)
        node_group = torch.cat((node_group,first_visited_node),dim=1)
        mask_for_encoder = torch.cat((encoder_mask,torch.zeros((bsz,2)).to(graph.device)),dim=1)
        
        if self.if_agg_whole_graph:
            unfinished_graph = graph.clone()
            unfinished_demands = demands.clone()
            unfinished_graph[finished_mask] = 0
            unfinished_demands[finished_mask] = 0
            unfinished_size = (nb_nodes-torch.sum(finished_mask,dim=1)).view((bsz,1))
            unfinished_size[unfinished_size==0]=1
            agg_node = torch.sum(unfinished_graph,dim=1)/unfinished_size
            agg_demand = torch.sum(unfinished_demands,dim=1)/unfinished_size.squeeze()
            node_group = torch.cat((node_group,agg_node.view((bsz,1,-1))),dim=1)
            mask_for_encoder = torch.cat((mask_for_encoder,torch.zeros((bsz,1)).to(graph.device)),dim=1)

        #scaled_node_group = node_group
        scaled_node_group = Normalization_layer(node_group,k+1,problem='vrp')
        # init embedding
        scaled_last = scaled_node_group[:,k:(k+1),:]
        scaled_first = scaled_node_group[:,(k+1):(k+2),:]
        scaled_remain = scaled_node_group[:,:k,:]
        scaled_last = torch.cat((scaled_last,remain_capacity_vec.unsqueeze(dim=2)),dim=2)
        scaled_remain = torch.cat((scaled_remain,demands_group.unsqueeze(dim=2)),dim=2)
        emb_last = self.input_emb_for_last(scaled_last)
        emb_first = self.input_emb_for_first(scaled_first)
        emb_remain = self.input_emb(scaled_remain)
        emb_input = torch.cat((emb_remain,emb_last),dim=1)
        emb_input = torch.cat((emb_input,emb_first),dim=1)
        if self.if_agg_whole_graph:
            scaled_agg = scaled_node_group[:,(k+2):(k+3),:]
            scaled_agg = torch.cat((scaled_agg,agg_demand.view((bsz,1,1))),dim=2)
            emb_agg = self.agg_emb(scaled_agg)
            emb_input = torch.cat((emb_input,emb_agg),dim=1)
        
        # encoding
        mask_for_encoder = mask_for_encoder.view((bsz,1,-1)).bool()
        emb_out,_ = self.encoder(emb_input,mask_for_encoder)
        return emb_out
        
class action_encoder_vrp(nn.Module):

    def __init__(self, dim_input_nodes, dim_emb, dim_ff, nb_layers_encoder, nb_heads, batchnorm = True):
        super(action_encoder_vrp, self).__init__()
        self.input_emb = nn.Linear(dim_input_nodes+1, dim_emb)
        self.input_emb_for_last = nn.Linear(dim_input_nodes+1, dim_emb)
        self.input_emb_for_first = nn.Linear(dim_input_nodes, dim_emb)

        # encoder layer
        self.encoder = Transformer_encoder_net(nb_layers_encoder, dim_emb, nb_heads, dim_ff, batchnorm)

    def forward(self,graph,idx,last_visited_node,first_visited_node,demands, remain_capacity_vec, encoder_mask=None, depot_mask=None):

        bsz = idx.size(0)
        k = idx.size(1)
        idx_for_ref = idx.view((bsz*k,))
        b_k = torch.arange(0,bsz).repeat(k).sort()[0].to(graph.device)
        node_group = graph[b_k,idx_for_ref].view((bsz,k,-1))
        demands_group = demands[b_k,idx_for_ref].view((bsz,k))
        node_group = torch.cat((node_group,last_visited_node),dim=1)
        node_group = torch.cat((node_group,first_visited_node),dim=1)
        mask_for_encoder = torch.cat((encoder_mask,torch.zeros((bsz,2)).to(graph.device)),dim=1)

        #scaled_node_group = node_group
        scaled_node_group = Normalization_layer(node_group,k+1,problem='vrp')
        # init embedding
        scaled_last = scaled_node_group[:,k:(k+1),:]
        scaled_first = scaled_node_group[:,(k+1):(k+2),:]
        scaled_remain = scaled_node_group[:,:k,:]
        scaled_last = torch.cat((scaled_last,remain_capacity_vec.unsqueeze(dim=2)),dim=2)
        scaled_remain = torch.cat((scaled_remain,demands_group.unsqueeze(dim=2)),dim=2)
        emb_last = self.input_emb_for_last(scaled_last)
        emb_first = self.input_emb_for_first(scaled_first)
        emb_remain = self.input_emb(scaled_remain)
        emb_input = torch.cat((emb_remain,emb_last),dim=1)
        emb_input = torch.cat((emb_input,emb_first),dim=1)
        
        # encoding
        mask_for_encoder = mask_for_encoder.view((bsz,1,-1)).bool()
        emb_out,_ = self.encoder(emb_input,mask_for_encoder)
        return emb_out