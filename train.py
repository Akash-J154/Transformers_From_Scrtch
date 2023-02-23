import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self,embeded_size,heads) -> None:
        super(SelfAttention,self).__init__()
        self.embeded_size=embeded_size
        self.heads=heads
        self.head_dim=embeded_size//heads

        assert(self.head_dim*self.heads==self.embeded_size),"head dim is not retuning an integer value"

        self.values=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.keys=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.query=nn.Linear(self.head_dim,self.head_dim,bias=False)
        self.fc_out=nn.Linear(self.embeded_size,self.embeded_size)

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


        attention=torch.softmax(product/(self.embeded_size)**(1/2),dim=3)
        attention_output=torch.einsum("nhqd,nlhd->nqhd",[attention,values]).reshape(N,query_len,self.heads*self.head_dim)
        attention_output=self.fc_out(attention_output)
        return attention_output

    
class TransformerBlock(nn.Module):
    def __init__(self,embeded_size,heads,dropout,forward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention=SelfAttention(embeded_size,heads)
        self.dropout=torch.Dropout(dropout)
        self.norm1=nn.LayerNorm(embeded_size)
        self.norm2=nn.LayerNorm(embeded_size)
        self.feedforward=nn.Sequential(
        nn.Layer(embeded_size,forward_expansion*embeded_size),
        nn.Relu(),
        nn.Layer(forward_expansion*embeded_size,embeded_size))
        
    def forward(self,query,keys,values,masks):
        attention=self.attention(query,keys,values,masks)
        x=self.dropout(self.norm1(attention+query))
        forward=self.feedforward(x)
        out=self.dropout(self.norm2(forward+x))
        return out
class Encoder(nn.Module):
    
    def __init__(self,src_vocab_size,embeded_size,num_layers,heads,device,forward_expansion,dropout,max_length):
        super(Encoder,self)__init__()
        self.embeded_size=embeded_size
        self.dropout=nn.Dropout(dropout)
        self.positional_embedding=nn.Embedding(max_length,embeded_size)
        self.word_embedding=nn.Embedding(src_vocab_size,embeded_size)
        self.layers=nn.ModuleList([TransformerBlock(embeded_size,heads,dropout,forward_expansion)])
        def forward(self,x,masks):
            N,seq_length=x.shape
            positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
            out=self.dropout(self.word_embedding+self.positional_embedding(positions))
            for layer in self.layers:
                out=layer(out,out,out,masks)
            return out
class DecoderBlock(nn.Module):
    def __init__(self,embeded_size,heads,forward_expansion,dropout,device):
        super(DecoderBlock,self)__init__()
        self.dropout=nn.Dropout(dropout)
        self.attention=SelfAttention(embeded_size,query,keys,values,masks)
        self.transformerblock=Transformer(embeded_size,heads,dropout,forward_expansion)
        self.norm1=nn.LayerNorm(embeded_size)
    def forward(self,x,query,keys,values,src_masks,trg_masks):
        attention=self.attention(x,x,x,trg_masks)
        normalization=self.dropout(self.norm1(attention+x))
        out=self.transformerblock(query,keys,values,src_masks)
        return out
class Decoder(nn.Module):
    def __init__(self,trg_vocab_size,embeded_size,num_layers,heads,forward_expansion,dropout,device,max_length):
        super(Decoder,self)__init__()
        self.device=device
        self.word_embedding=nn.Embedding(trg_vocab_size,embeded_size)
        self.positional_embedding=nn.Embedding(max_length,embeded_size)
        self.layers=nn.ModuleList([DecoderBlock(embeded_size,heads,forward_expansion,dropout,device)])
        self.dropout=nn.Dropout(dropout)
        self.fc_out=nn.Linear(embeded_size,trg_vocab_size)
    def forward(self,x,enc_out,src_masks,trg_masks):
        N,seq_length=x.shape
        positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        x=self.dropout(self.positional_embedding(positions)+self.word_embedding(x))
        for layer in self.layers:
            x=layer(x,enc_out,src_masks,trg_masks)
        out=self.fc_out(x)
class Transformer(nn.Module):
    def __init__(self,src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx,embeded_size=256,num_layers=6,forward_expansion=4,heads=8,dropout=0,device='cuda',max_length=100):
        super()__init__(Transformer,self)
        self.encoder=Encoder(src_vocab_size,embeded_size,num_layers,heads,device,forward_expansion,dropout,max_length)
        self.decoder=Decoder(trg_vocab_size,embeded_size,num_layers,heads,device,forward_expansion,dropout,max_length)
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        self.device=device
    def make_src_masks(self,src):
        src_masks=(src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_masks.to(device)
    def make_trg_masks(self,trg):
        N,trg_len=trg.shape
        trg_masks=torch.tril(torch.ones(trg_len,trg_len)).expand(N,1,trg_len,trg_len)
        return trg_masks.to(device)
    def forward()
    
        
                
        