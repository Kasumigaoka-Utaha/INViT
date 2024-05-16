import torch
import torch.nn as nn
import torch.nn.functional as F


def myMHA(Q, K, V, nb_heads, mask=None, clip_value=None):
    """
    Compute multi-head attention (MHA) given a query Q, key K, value V and attention mask :
      h = Concat_{k=1}^nb_heads softmax(Q_k^T.K_k).V_k 
    Note : We did not use nn.MultiheadAttention to avoid re-computing all linear transformations at each call.
    Inputs : Q of size (bsz, dim_emb, 1)                batch of queries
             K of size (bsz, dim_emb, nb_nodes+1)       batch of keys
             V of size (bsz, dim_emb, nb_nodes+1)       batch of values
             mask of size (bsz, nb_nodes+1)             batch of masks of visited cities
             clip_value is a scalar 
    Outputs : attn_output of size (bsz, 1, dim_emb)     batch of attention vectors
              attn_weights of size (bsz, 1, nb_nodes+1) batch of attention weights
    """
    bsz, nb_nodes, emd_dim = K.size() #  dim_emb must be divisable by nb_heads
    if nb_heads>1:
        # PyTorch view requires contiguous dimensions for correct reshaping
        Q = Q.transpose(1,2).contiguous() # size(Q)=(bsz, dim_emb, 1)
        Q = Q.view(bsz*nb_heads, emd_dim//nb_heads, 1) # size(Q)=(bsz*nb_heads, dim_emb//nb_heads, 1)
        Q = Q.transpose(1,2).contiguous() # size(Q)=(bsz*nb_heads, 1, dim_emb//nb_heads)
        K = K.transpose(1,2).contiguous() # size(K)=(bsz, dim_emb, nb_nodes+1)
        K = K.view(bsz*nb_heads, emd_dim//nb_heads, nb_nodes) # size(K)=(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        K = K.transpose(1,2).contiguous() # size(K)=(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
        V = V.transpose(1,2).contiguous() # size(V)=(bsz, dim_emb, nb_nodes+1)
        V = V.view(bsz*nb_heads, emd_dim//nb_heads, nb_nodes) # size(V)=(bsz*nb_heads, dim_emb//nb_heads, nb_nodes+1)
        V = V.transpose(1,2).contiguous() # size(V)=(bsz*nb_heads, nb_nodes+1, dim_emb//nb_heads)
    attn_weights = torch.bmm(Q, K.transpose(1,2))/ Q.size(-1)**0.5 # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    if clip_value is not None:
        attn_weights = clip_value * torch.tanh(attn_weights)
    if mask is not None:
        if nb_heads>1:
            mask = torch.repeat_interleave(mask, repeats=nb_heads, dim=0) # size(mask)=(bsz*nb_heads, nb_nodes+1)
        #attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-inf')) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
        attn_weights = attn_weights.masked_fill(mask.bool().unsqueeze(dim=1), float('-1e9')) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)


    attn_weights = torch.softmax(attn_weights, dim=-1) # size(attn_weights)=(bsz*nb_heads, 1, nb_nodes+1)
    attn_output = torch.bmm(attn_weights, V) # size(attn_output)=(bsz*nb_heads, 1, dim_emb//nb_heads)
    if nb_heads>1:
        attn_output = attn_output.transpose(1,2).contiguous() # size(attn_output)=(bsz*nb_heads, dim_emb//nb_heads, 1)
        attn_output = attn_output.view(bsz, emd_dim, 1) # size(attn_output)=(bsz, dim_emb, 1)
        attn_output = attn_output.transpose(1,2).contiguous() # size(attn_output)=(bsz, 1, dim_emb)
        attn_weights = attn_weights.view(bsz, nb_heads, 1, nb_nodes) # size(attn_weights)=(bsz, nb_heads, 1, nb_nodes+1)
        attn_weights = attn_weights.mean(dim=1) # mean over the heads, size(attn_weights)=(bsz, 1, nb_nodes+1)
    return attn_output, attn_weights           



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k, bias=False)
        self.w_ks = nn.Linear(d_model, d_k, bias=False)
        self.w_vs = nn.Linear(d_model, d_v, bias=False)
        self.fc = nn.Linear(d_v, d_model, bias=False)

        self.d_k_head = (d_k//n_head)
        self.d_v_head = (d_v//n_head)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k_head, self.d_v_head, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv0
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.


        attn = torch.matmul(q / self.d_k**0.5, k.transpose(2, 3))   

        if mask is not None:
            attn = attn.masked_fill(mask.bool(), -1e9)
        
        attn = F.softmax(attn, dim=-1)
        q = torch.matmul(attn, v)
        #q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q = q + residual 

        q = self.layer_norm(q)

        return q, attn