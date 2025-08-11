import numpy as np
from torch import nn,triu,ones,float32,matmul,arange,sin,cos,zeros,mean,std,reshape,transpose
import math

def mask_input(X):
    neg_inf=-1e9
    seq_len=X.shape[-1]
    mask=ones((seq_len,seq_len))
    mask=triu(mask,1)
    mask=mask*neg_inf
    out=X+mask
    return out
#Positional embedding for the model to understand the positions obviously
class PositionalEncoding(nn.Module):
    def __init__(self, vocab, d_model, max_len, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.vocab=vocab
        self.d_model=d_model
        self.max_len=max_len
        self.embedding=nn.Embedding(vocab,d_model)
        pos = arange(max_len).unsqueeze(1)
        i = arange(d_model).unsqueeze(0)
        angle_rates = 1 / pow(10000, (2 * (i//2)) / d_model)

        self.pe = zeros(max_len, d_model)
        self.pe[:, 0::2] = sin(pos * angle_rates[:, 0::2])
        self.pe[:, 1::2] = cos(pos * angle_rates[:, 1::2])

        self.register_buffer("position_embedding", self.pe.unsqueeze(0))

    def forward(self,X):
        seq_len=X.shape[-1]
        X=self.embedding(X)
        return X+self.pe[:seq_len,:]
class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,max_len,num_heads, **kwargs):
        super(MaskedMultiHeadSelfAttention,self).__init__( **kwargs)
        self.d_model=d_model
        self.max_len=max_len
        self.num_heads=num_heads
        self.q=nn.Linear(d_model,d_model)
        self.k=nn.Linear(d_model,d_model)
        self.v=nn.Linear(d_model,d_model)
        self.final=nn.Linear(d_model,d_model)
        
    def split_heads(self,x, num_heads,batch_size):
    # (batch, seq_len, d_model) → (batch, num_heads, seq_len, head_dim)
        
        x = x.reshape(batch_size, self.max_len, num_heads, self.d_model // num_heads)
        return x.permute(0,2,1,3)
    def forward(self,X):
        batch_size=X.shape[0]

        q=self.q(X)
        k=self.k(X)
        v=self.v(X)
        
        q=self.split_heads(q,self.num_heads,batch_size)
        k=self.split_heads(k,self.num_heads,batch_size)
        v=self.split_heads(v,self.num_heads,batch_size)

        #Scaled dot product
        dk=self.d_model//self.num_heads
        at_score = matmul( q, transpose(k,-2,-1)) / math.sqrt(dk)
        mask=mask_input(at_score)
        weights=nn.functional.softmax(mask,-1)

        attention = matmul(weights, v)

        #merging 
        attention=attention.permute(0,2,1,3)#(batch_size,seq_len,heads,d_model)
        concat_attention=attention.reshape(batch_size,-1,self.d_model)
        #to capture the final meaning of the word according to the context
        out=self.final(concat_attention)
        return out
    
class FeedForward(nn.Module):
    def __init__(self,units,d_model, **kwargs):
        super(FeedForward,self).__init__( **kwargs)
        self.inputs=nn.Linear(d_model,units,)
        self.out=nn.Linear(units,d_model)
    def forward(self,X):
        X=self.inputs(X)  
        X=nn.ReLU(X)
        X=self.out(X)
        return X

class LayerNormalization(nn.Module):
    def __init__(self,d_model,epsilon=1e-5, **kwargs):
        super(LayerNormalization,self).__init__( **kwargs)
        self.d_model=d_model
        self.epsilon=epsilon
        self.beta=nn.Parameter(zeros(d_model)) 
        self.gamma=nn.Parameter(ones(d_model))
    def forward(self,X):
        mn = mean(X,-1,True)
        stdiv= std(X,-1,keepdim=True)
        X_norm=(X-mn)/(stdiv+self.epsilon)
        out=X_norm*self.gamma+self.beta
        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self,d_model,max_len,num_heads, **kwargs):
        super(MultiHeadCrossAttention,self).__init__( **kwargs)
        self.d_model=d_model
        self.max_len=max_len
        self.num_heads=num_heads
        self.q=nn.Linear(d_model,d_model)
        self.k=nn.Linear(d_model,d_model)
        self.v=nn.Linear(d_model,d_model)
        self.final=nn.Linear(d_model,d_model)
        
    def split_heads(self,x, num_heads,batch_size):
    # (batch, seq_len, d_model) → (batch, num_heads, seq_len, head_dim)
        
        x = reshape(x, (batch_size, -1, num_heads, self.d_model // num_heads))
        return x.permute(0, 2, 1, 3)
    def call(self,X,y):
        batch_size=X.shape[0]

        q=self.q(y)
        k=self.k(X)
        v=self.v(X)
        
        q=self.split_heads(q,self.num_heads,batch_size)
        k=self.split_heads(k,self.num_heads,batch_size)
        v=self.split_heads(v,self.num_heads,batch_size)

        #Scaled dot product
        dk=self.d_model//self.num_heads
        at_score = matmul(q, transpose(k,-2,-1)) / math.sqrt(dk)        
        weights=nn.functional.softmax(at_score,-1)
        attention=matmul(weights,v)

        #merging 
        attention=attention.permute(0,2,1,3)#(batch_size,seq_len,heads,d_model)
        concat_attention=reshape(attention,(batch_size,-1,self.d_model))
        #to capture the final meaning of the word according to the context
        out=self.final(concat_attention)
        return out
    

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,max_len,ff_hidden,drop_prob,**kwargs):
        super(DecoderLayer,self).__init__(**kwargs)
        self.masked_attention=MaskedMultiHeadSelfAttention(d_model,max_len,num_heads)
        self.cross_attention=MultiHeadCrossAttention(d_model,max_len,num_heads)
        self.feed_forward=FeedForward(ff_hidden,d_model)
        self.layer_Norm1=LayerNormalization(d_model)
        self.layer_Norm2=LayerNormalization(d_model)
        self.layer_Norm3=LayerNormalization(d_model)
        self.dropout1=nn.Dropout(drop_prob)
        self.dropout2=nn.Dropout(drop_prob)
        self.dropout3=nn.Dropout(drop_prob)
    def forward(self,X,y):
        res=y
        out=self.masked_attention(y)
        out=self.dropout1(out)
        out=self.layer_Norm1(out+res)
        res=out
        out=self.cross_attention(X,out)
        out=self.dropout2(out)
        out=self.layer_Norm2(out+res)
        res=out
        out=self.feed_forward(out)
        out=self.dropout3(out)
        out=self.layer_Norm3(out+res)
        return out
