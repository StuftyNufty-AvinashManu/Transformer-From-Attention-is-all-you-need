from tensorflow.keras.layers import Embedding,Dense,Layer,Dropout
from tensorflow import matmul,nn,reshape,shape,transpose,float32,cast,convert_to_tensor,Variable,reduce_mean,ones
from tensorflow.linalg import band_part
from tensorflow.math import reduce_std,sqrt
import numpy as np
from keras.saving import register_keras_serializable

    

    
def mask_input(X):
    neg_inf=-1e-9
    seq_len=shape(X)[-1]
    mask=band_part(ones((seq_len,seq_len)),-1,0)
    mask=1.0-mask
    mask=mask*neg_inf


    return cast(mask,dtype=float32)

#Positional embedding for the model to understand the positions obviously
@register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, vocab, d_model, max_len, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.vocab=vocab
        self.d_model=d_model
        self.max_len=max_len
        self.embedding=Embedding(input_dim=vocab,output_dim=d_model,input_length=max_len)
        self.position_embedding=[]
        
        for pos in range(max_len):
            embeddings=[]
            for i in range(d_model):
                if i%2 ==0:
                    embeddings.append(np.cos(pos/10000**((2*i)/d_model)))
                else:
                    embeddings.append(np.sin(pos/10000**((2*i)/d_model)))
            self.position_embedding.append(embeddings)
        self.position_embedding=convert_to_tensor([self.position_embedding],dtype='float32')    
        
        
    def call(self,X):
        X=self.embedding(X)
        return X+self.position_embedding  
        
@register_keras_serializable()
class MaskedMultiHeadSelfAttention(Layer):
    def __init__(self,d_model,max_len,num_heads, **kwargs):
        super(MaskedMultiHeadSelfAttention,self).__init__( **kwargs)
        self.d_model=d_model
        self.max_len=max_len
        self.num_heads=num_heads
        self.q=Dense(d_model)
        self.k=Dense(d_model)
        self.v=Dense(d_model)
        self.final=Dense(d_model)
        
    def split_heads(self,x, num_heads,batch_size):
    # (batch, seq_len, d_model) → (batch, num_heads, seq_len, head_dim)
        
        x = reshape(x, (batch_size, -1, num_heads, self.d_model // num_heads))
        return transpose(x, perm=[0, 2, 1, 3])
    def call(self,X):
        batch_size=shape(X)[0]

        q=self.q(X)
        k=self.k(X)
        v=self.v(X)
        
        q=self.split_heads(q,self.num_heads,batch_size)
        k=self.split_heads(k,self.num_heads,batch_size)
        v=self.split_heads(v,self.num_heads,batch_size)

        #Scaled dot product
        dk=cast(self.d_model//self.num_heads,float32)
        at_score = matmul(q, k, transpose_b=True) / sqrt(cast(dk, float32))        
        mask=mask_input(at_score)
        weights=nn.softmax(at_score+mask,axis=-1)

        attention=matmul(weights,v)

        #merging 
        attention=transpose(attention,perm=[0,2,1,3])#(batch_size,seq_len,heads,d_model)
        concat_attention=reshape(attention,(batch_size,-1,self.d_model))
        #to capture the final meaning of the word according to the context
        out=self.final(concat_attention)
        return out
    
@register_keras_serializable()
class FeedForward(Layer):
    def __init__(self,units,d_model, **kwargs):
        super(FeedForward,self).__init__( **kwargs)
        self.inputs=Dense(units,activation='relu')
        self.out=Dense(d_model,activation='linear')
    def call(self,X):
        X=self.inputs(X)  
        X=self.out(X)
        return X

@register_keras_serializable()
class LayerNormalization(Layer):
    def __init__(self,d_model,epsilon=1e-5, **kwargs):
        super(LayerNormalization,self).__init__( **kwargs)
        self.d_model=d_model
        self.epsilon=epsilon
        self.beta=self.add_weight(name="beta",shape=(d_model,),initializer='zeros',trainable=True) 
        self.gamma=self.add_weight(name="gamma",shape=(d_model,),initializer='ones',trainable=True)
    def call(self,X):
        mean = reduce_mean(X,axis=-1,keepdims=True)
        std= reduce_std(X,axis=-1,keepdims=True)
        X_norm=(X-mean)/(std+self.epsilon)
        out=X_norm*self.gamma+self.beta
        return out
@register_keras_serializable()
class MultiHeadCrossAttention(Layer):
    def __init__(self,d_model,max_len,num_heads, **kwargs):
        super(MultiHeadCrossAttention,self).__init__( **kwargs)
        self.d_model=d_model
        self.max_len=max_len
        self.num_heads=num_heads
        self.q=Dense(d_model)
        self.k=Dense(d_model)
        self.v=Dense(d_model)
        self.final=Dense(d_model)
        
    def split_heads(self,x, num_heads,batch_size):
    # (batch, seq_len, d_model) → (batch, num_heads, seq_len, head_dim)
        
        x = reshape(x, (batch_size, -1, num_heads, self.d_model // num_heads))
        return transpose(x, perm=[0, 2, 1, 3])
    def call(self,X,y):
        batch_size=shape(X)[0]

        q=self.q(y)
        k=self.k(X)
        v=self.v(X)
        
        q=self.split_heads(q,self.num_heads,batch_size)
        k=self.split_heads(k,self.num_heads,batch_size)
        v=self.split_heads(v,self.num_heads,batch_size)

        #Scaled dot product
        dk=cast(self.d_model//self.num_heads,float32)
        at_score = matmul(q, k, transpose_b=True) / sqrt(cast(dk, float32))        
        weights=nn.softmax(at_score,axis=-1)
        attention=matmul(weights,v)

        #merging 
        attention=transpose(attention,perm=[0,2,1,3])#(batch_size,seq_len,heads,d_model)
        concat_attention=reshape(attention,(batch_size,-1,self.d_model))
        #to capture the final meaning of the word according to the context
        out=self.final(concat_attention)
        return out
    

@register_keras_serializable
class DecoderLayer(Layer):
    def __init__(self,d_model,num_heads,max_len,ff_hidden,drop_prob,**kwargs):
        super(DecoderLayer,self).__init__(**kwargs)
        self.masked_attention=MaskedMultiHeadSelfAttention(d_model,max_len,num_heads)
        self.cross_attention=MultiHeadCrossAttention(d_model,max_len,num_heads)
        self.feed_forward=FeedForward(ff_hidden,d_model)
        self.layer_Norm1=LayerNormalization(d_model)
        self.layer_Norm2=LayerNormalization(d_model)
        self.layer_Norm3=LayerNormalization(d_model)
        self.dropout1=Dropout(drop_prob)
        self.dropout2=Dropout(drop_prob)
        self.dropout3=Dropout(drop_prob)
    def call(self,X,y):
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
