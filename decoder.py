import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import myMHA

class AutoRegressiveDecoderLayer(nn.Module):
    """
    Single decoder layer based on self-attention and query-attention
    Inputs :  
      h_t of size      (bsz, 1, dim_emb)          batch of input queries
      K_att of size    (bsz, nb_nodes+1, dim_emb) batch of query-attention keys
      V_att of size    (bsz, nb_nodes+1, dim_emb) batch of query-attention values
      mask of size     (bsz, nb_nodes+1)          batch of masks of visited cities
    Output :  
      h_t of size (bsz, nb_nodes+1)               batch of transformed queries
    """
    def __init__(self, dim_emb, nb_heads):
        super(AutoRegressiveDecoderLayer, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.W0_att = nn.Linear(dim_emb, dim_emb)
        self.Wq_att = nn.Linear(dim_emb, dim_emb)
        self.W1_MLP = nn.Linear(dim_emb, dim_emb)
        self.W2_MLP = nn.Linear(dim_emb, dim_emb)
        self.BN_att = nn.LayerNorm(dim_emb)
        self.BN_MLP = nn.LayerNorm(dim_emb)
        self.K_sa = None
        self.V_sa = None
        
    def forward(self, h_t, K_att, V_att, mask=None):
        bsz = h_t.size(0)
        h_t = h_t.view(bsz,1,self.dim_emb) # size(h_t)=(bsz, 1, dim_emb)
        # compute attention between self-attention nodes and encoding nodes in the partial tour (translation process)
        q_a = self.Wq_att(h_t) # size(q_a)=(bsz, 1, dim_emb)
        h_t = h_t + self.W0_att( myMHA(q_a, K_att, V_att, self.nb_heads,mask)[0] ) # size(h_t)=(bsz, 1, dim_emb)
        h_t = self.BN_att(h_t.squeeze()) # size(h_t)=(bsz, dim_emb)
        h_t = h_t.view(bsz, 1, self.dim_emb) # size(h_t)=(bsz, 1, dim_emb)
        # MLP
        h_t = h_t + self.W2_MLP(torch.relu(self.W1_MLP(h_t)))
        h_t = self.BN_MLP(h_t.squeeze(1)) # size(h_t)=(bsz, dim_emb)
        return h_t
    
class Transformer_decoder_net(nn.Module): 
    """
    Decoder network based on self-attention and query-attention transformers
    Inputs :  
      h_t of size      (bsz, 1, dim_emb)                            batch of input queries
      K_att of size    (bsz, nb_nodes+1, dim_emb*nb_layers_decoder) batch of query-attention keys for all decoding layers
      V_att of size    (bsz, nb_nodes+1, dim_emb*nb_layers_decoder) batch of query-attention values for all decoding layers
      mask of size     (bsz, nb_nodes+1)                            batch of masks of visited cities
    Output :  
      prob_next_node of size (bsz, nb_nodes+1)                      batch of probabilities of next node
    """
    def __init__(self, dim_emb, nb_heads, nb_layers_decoder):
        super(Transformer_decoder_net, self).__init__()
        self.dim_emb = dim_emb
        self.nb_heads = nb_heads
        self.nb_layers_decoder = nb_layers_decoder
        self.decoder_layers = nn.ModuleList( [AutoRegressiveDecoderLayer(dim_emb, nb_heads) for _ in range(nb_layers_decoder-1)] )
        self.Wq_final = nn.Linear(dim_emb, dim_emb)
            
    def forward(self, h_t, K_att, V_att, mask=None):
        for l in range(self.nb_layers_decoder):
            K_att_l = K_att[:,:,l*self.dim_emb:(l+1)*self.dim_emb].contiguous()  # size(K_att_l)=(bsz, nb_nodes+1, dim_emb)
            V_att_l = V_att[:,:,l*self.dim_emb:(l+1)*self.dim_emb].contiguous()  # size(V_att_l)=(bsz, nb_nodes+1, dim_emb)
            if l<self.nb_layers_decoder-1: # decoder layers with multiple heads (intermediate layers)
                h_t = self.decoder_layers[l](h_t, K_att_l, V_att_l, mask)
            else: # decoder layers with single head (final layer)
                q_final = self.Wq_final(h_t)
                bsz = h_t.size(0)
                q_final = q_final.view(bsz, 1, self.dim_emb)
                attn_weights = myMHA(q_final, K_att_l, V_att_l, 1, mask, 10)[1] 
        prob_next_node = attn_weights.squeeze(1) 
        return prob_next_node 