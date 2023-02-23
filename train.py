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
        keys=keys.reshape(N,key_len,self.heads,self,self.head_dim)
        values=values.reshape(N,value_len,self.heads,self.head_dim)
        query=query.reshape(N,query_len,self.heads,self.head_dim)

        #finding the dot product

        product=torch.einsum("nqhd,nkhd->nhqk",[query,keys])
        if mask is not None:
            product=product.masked_fill(mask==0,float("-1e20"))


        attention=torch.softmax(product/(self.embed_size)**(1/2),dim=3)
        attention_output=torch.einsum("nhqd,nlhd->nqhd",[attention,values]).reshape(N,query_len,self.heads*self.head_dim)
        attention_output=self.fc_out(attention_output)
        return attention_output
        