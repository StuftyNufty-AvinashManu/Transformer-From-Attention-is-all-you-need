from torch import nn,ones,float32,matmul,arange,sin,cos,zeros,mean,std,reshape,transpose,tensor
import math
#Positional embedding for the model to understand the positions obviously
class PositionalEncoding(nn.Module):
    def __init__(self, vocab, d_model, max_len, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.vocab=vocab
        self.d_model=d_model
        self.max_len=max_len
        self.embedding=nn.Embedding(vocab,d_model)
        pos=[]
        for i in range(max_len):
            embed=[]
            for j in range(d_model):
                if j % 2 == 0:
                    embed.append(math.sin(i / (10000 ** (j / d_model))))
                else:
                    embed.append(math.cos(i / (10000 ** ((j - 1) / d_model))))
            pos.append(embed)
        pos=tensor(pos, dtype=float32)
        self.register_buffer("position_embedding",pos)

    def forward(self,X):
        seq_len=X.shape[-1]
        X=self.embedding(X)
        return X+self.position_embedding[:seq_len,:].unsqueeze(0) 
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,max_len,num_heads, **kwargs):
        super(MultiHeadSelfAttention,self).__init__( **kwargs)
        self.d_model=d_model
        self.max_len=max_len
        self.num_heads=num_heads
        self.q=nn.LazyLinear(d_model)
        self.k=nn.LazyLinear(d_model)
        self.v=nn.LazyLinear(d_model)
        self.final=nn.LazyLinear(d_model)
        
    def split_heads(self,x, num_heads,batch_size):
    # (batch, seq_len, d_model) → (batch, num_heads, seq_len, head_dim)
        seq_len=x.shape[-2]
        x = reshape(x, (batch_size, -1, num_heads, self.d_model // num_heads))
        return x.permute(0, 2, 1, 3)
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
        at_score = matmul(q, transpose(k,-2,-1)) / math.sqrt(dk)        
        weights=nn.functional.softmax(at_score,-1)
        attention=matmul(weights,v)

        #merging 
        attention=attention.permute(0,2,1,3)#(batch_size,seq_len,heads,d_model)
        concat_attention=reshape(attention,(batch_size,-1,self.d_model))
        #to capture the final meaning of the word according to the context
        out=self.final(concat_attention)
        return out
    
class FeedForward(nn.Module):
    def __init__(self,units,d_model, **kwargs):
        super(FeedForward,self).__init__( **kwargs)
        self.inputs=nn.LazyLinear(units)
        self.out=nn.LazyLinear(d_model)
    def forward(self,X):
        X=self.inputs(X)  
        X=nn.functional.relu(X)
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
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model,ff_hidden,max_len,num_heads,drop_prob, **kwargs):
        super(EncoderLayer,self).__init__( **kwargs)
        self.attention=MultiHeadSelfAttention(d_model,max_len,num_heads)
        self.feed_forward=FeedForward(ff_hidden,d_model)
        self.layer_norm1=LayerNormalization(d_model)
        self.layer_norm2=LayerNormalization(d_model)
        self.dropout1=nn.Dropout(drop_prob)
        self.dropout2=nn.Dropout(drop_prob)
    def forward(self, X):
        res = X
        X = self.attention(X)
        X = self.dropout1(X)
        X = self.layer_norm1(res + X)
        res = X
        X = self.feed_forward(X)
        X = self.dropout2(X)
        X = self.layer_norm2(res + X)
        return X