import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads) -> None:
        super(SelfAttention,self).__init__()
        self.embed_size=embed_size
        self.heads=heads
        self.head_dim=embed_size//heads

        assert(self.head_dim*self.heads==self.embed_size),"head dim is not retuning an integer value"

        self.values=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.keys=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.query=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.fc_out=nn.Linear(self.embed_size,self.embed_size)

    def forward(self,query,keys,values,masks):
        N=query.shape[0];    
        key_len,value_len,query_len=keys.shape[1],values.shape[1],query.shape[1]
        #reshaping all so that the dot product between queries and keys can be calculated
        keys=keys.reshape(N,key_len,self.head,self,self.head_dim)
        values=values.reshape(N,value_len,self.head,self.head_dim)
        query=query.reshape(N,query_len,self.head,self.head_dim)

        #finding the dot product

        product=torch.einsum


