from tensorflow.keras.layers import Embedding
from tensorflow import convert_to_tensor
import numpy as np

class PositionalEncoding():
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
        
        
    def call(self,X):
        X=self.embedding(X)
        position_embedding=convert_to_tensor([self.position_embedding],dtype='float32')
        return X+position_embedding
        


