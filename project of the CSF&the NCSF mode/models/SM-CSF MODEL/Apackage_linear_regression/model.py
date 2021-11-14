import tensorflow as tf 
import time
import numpy as np
import config as C

class neural_network:
    def __init__(self,config):
        self.n_input=config["n_inp"]
        self.n_output=config["n_out"]
        self.weight_initialization =  tf.contrib.layers.xavier_initializer()
        
        self.construct_input()
        self.build_model()
        self.build_loss()  
    def construct_input(self):
        self.input=tf.placeholder(tf.float64,[None, self.n_input])
        self.output=tf.placeholder(tf.float64,[None, self.n_output])
        
    def build_model(self):
        self.w=tf.get_variable('weight',
                    shape=[self.n_input,self.n_output],
                    initializer=self.weight_initialization,
                    dtype=tf.float64
                    )
        self.b=tf.get_variable("bias",shape=[self.n_output],
                               initializer=self.weight_initialization,#tf.constant([0.01],tf.float64),
                               dtype=tf.float64)
        self.predict=tf.add(tf.matmul(self.input, self.w), self.b)
    def build_loss(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=self.predict))
            
if __name__=="__main__":      
   stru_config=C.stru_config
   y=neural_network(stru_config)
   print(y.predict)
   print(y.loss)
