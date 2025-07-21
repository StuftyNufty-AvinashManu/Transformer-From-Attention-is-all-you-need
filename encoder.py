from tensorflow.keras.layers import Embedding,Dense,Layer
from tensorflow import concat,random,matmul,nn,reshape,shape,transpose,float32,cast,convert_to_tensor
import math
import numpy as np
#Positional embedding for the model to understand the positions obviously
class PositionalEncoding(Layer):
    def __init__(self, vocab,d_model,max_len):
        super().__init__()
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
        
class MultiHeadSelfAttention(Layer):
    def __init__(self,d_model,max_len,num_heads):
        super().__init__()
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
        at_score=matmul(q,k,transpose_b=True)/math.sqrt(dk)
        weights=nn.softmax(at_score,axis=-1)
        attention=matmul(weights,v)

        #merging 
        attention=transpose(attention,perm=[0,2,1,3])#(batch_size,seq_len,heads,d_model)
        concat_attention=reshape(attention,(batch_size,-1,self.d_model))
        #to capture the final meaning of the word according to the context
        output=self.final(concat_attention)
        return output