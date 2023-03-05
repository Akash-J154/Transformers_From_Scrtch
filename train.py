import torch
import torch.nn as nn
from torch.nn import ReLU
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
        keys=keys.reshape(N,key_len,self.heads,self.head_dim)
        values=values.reshape(N,value_len,self.heads,self.head_dim)
        query=query.reshape(N,query_len,self.heads,self.head_dim)

        values=self.values(values)
        keys=self.keys(keys)
        query=self.query(query)

        #finding the dot product

        product=torch.einsum("nqhd,nkhd->nhqk",[query,keys])
        if masks is not None:
            product=product.masked_fill(masks==0,float("-1e20"))


        attention=torch.softmax(product/(self.embeded_size)**(1/2),dim=3)
        attention_output=torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(N,query_len,self.heads*self.head_dim)
        attention_output=self.fc_out(attention_output)
        return attention_output

    
class TransformerBlock(nn.Module):
    def __init__(self,embeded_size,heads,dropout,forward_expansion):
        super(TransformerBlock,self).__init__()
        self.attention=SelfAttention(embeded_size,heads)
        self.dropout=nn.Dropout(dropout)
        self.norm1=nn.LayerNorm(embeded_size)
        self.norm2=nn.LayerNorm(embeded_size)
        self.feedforward=nn.Sequential(
        nn.Linear(embeded_size,forward_expansion*embeded_size),
        ReLU(),
        nn.Linear(forward_expansion*embeded_size,embeded_size))
        
    def forward(self,query,keys,values,masks):
        attention=self.attention(query,keys,values,masks)
        x=self.dropout(self.norm1(attention+query))
        forward=self.feedforward(x)
        out=self.dropout(self.norm2(forward+x))
        return out
class Encoder(nn.Module):
    
    def __init__(self,src_vocab_size,embeded_size,num_layers,heads,device,forward_expansion,dropout,max_length):
        super(Encoder,self).__init__()
        self.embeded_size=embeded_size
        self.device=device
        self.positional_embedding=nn.Embedding(max_length,embeded_size)
        self.word_embedding=nn.Embedding(src_vocab_size,embeded_size)
        self.layers = []
        for _ in range(num_layers):
            transformer_block = TransformerBlock(embeded_size,heads,dropout,forward_expansion)
            self.layers.append(transformer_block)
            self.add_module(f"transformer_block_{len(self.layers)}", transformer_block)
        self.fc_out = nn.Linear(embeded_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,masks):
        N,seq_length=x.shape
        positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
        out=self.dropout(self.word_embedding(x)+self.positional_embedding(positions))
        for layer in self.layers:
            out=layer(out,out,out,masks)
        return out
class DecoderBlock(nn.Module):
    def __init__(self,embeded_size,heads,forward_expansion,dropout,device):
        super(DecoderBlock,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.attention=SelfAttention(embeded_size,heads)
        self.device=device
        self.forward_expansion=forward_expansion
        self.transformerblock=TransformerBlock(embeded_size,heads,dropout,forward_expansion)
        self.norm1=nn.LayerNorm(embeded_size)
    def forward(self,x,query,keys,values,src_mask,trg_mask):
        attention=self.attention(x,x,x,trg_mask)
        query=self.dropout(self.norm1(attention+x))
        out=self.transformerblock(query,keys,values,src_mask)
        return out
class Decoder(nn.Module):
    def __init__(self,trg_vocab_size,embeded_size,num_layers,heads,device,forward_expansion,dropout,max_length):
        super(Decoder,self).__init__()
        self.device=device
        self.num_layers=num_layers
        self.word_embedding=nn.Embedding(trg_vocab_size,embeded_size)
        self.positional_embedding=nn.Embedding(max_length,embeded_size)
        self.dropout=nn.Dropout(0)
        self.layers = []
        for _ in range(num_layers):
            decoder_block = DecoderBlock(embeded_size, heads, forward_expansion, dropout, device)
            self.layers.append(decoder_block)
            self.add_module(f"decoder_block_{len(self.layers)}", decoder_block)
        self.fc_out = nn.Linear(embeded_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
       
        self.fc_out=nn.Linear(embeded_size,trg_vocab_size)
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.positional_embedding(positions) + self.word_embedding(x))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):  
    def __init__(self,src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx,embeded_size=32,num_layers=2,forward_expansion=2,heads=8,dropout=0,device='cuda',max_length=100):
        super(Transformer,self).__init__()
        self.encoder=Encoder(src_vocab_size,embeded_size,num_layers,heads,device,forward_expansion,dropout,max_length)
        self.decoder=Decoder(trg_vocab_size,embeded_size,num_layers,heads,device,forward_expansion,dropout,max_length)
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        
        self.device=device
    def make_src_masks(self,src):
        src_mask=(src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    def make_trg_masks(self,trg):
        N,trg_len=trg.shape
        trg_mask=torch.tril(torch.ones(trg_len,trg_len)).expand(N,1,trg_len,trg_len)
        return trg_mask.to(self.device)
    def forward(self,src,trg):
        src_masks=self.make_src_masks(src)
        trg_masks=self.make_trg_masks(trg)
        enc_src=self.encoder(src,src_masks)
        out=self.decoder(trg,enc_src,src_masks,trg_masks)
        return out
    
if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src=torch.tensor([[1,5,6,4,3,9,5,2,0],[1,8,7,3,4,5,6,7,2]]).to(device)
    trg=torch.tensor([[1,7,4,3,5,9,2,0],[1,5,6,2,4,7,6,2]]).to(device)

    src_pad_idx=0
    trg_pad_idx=0
    src_vocab_size=10
    trg_vocab_size=10
    # src_mask = nn.Transformer().generate_square_subsequent_mask(x.size(0))
    # trg_mask = nn.Transformer().generate_square_subsequent_mask(trg.size(0))
    model=Transformer(src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx).to(device)
    out=model(src,trg[:,:-1])
    print(out)

                
        