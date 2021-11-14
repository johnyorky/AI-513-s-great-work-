import tensorflow as tf 
import time
import numpy as np
import config as C

class neural_network:
    def __init__(self,config):
        self.n_input=config['n_inp']
        self.n_output=config["n_out"]
        self.embedding=config["embedding"]
        self.full_connect_size=config["full_connect_size"]
        self._drop_out=config["drop_out"]
        self.which_model=config["which_model"]
        self.config=config

        self._bias_initializer=tf.zeros_initializer()
        self.weight_initialization =  tf.contrib.layers.xavier_initializer()
        self.construct_input()
         
        if self.which_model=="LSTM":
            self.build_bilstm()
        if self.which_model=="CNN":
            self.build_cnn()
        self.build_predict()
        self.build_loss()  
    tf.reset_default_graph() 
    def construct_input(self):
        if self.embedding:
            self.input=tf.placeholder(tf.int32,[None, self.n_input])
            self.num_size=self.config["num_size"]
            self.embedding_size=self.config["embedding_size"]
            with tf.variable_scope("embedding"):            
                self.W_embedding=tf.get_variable(shape=[self.num_size+1,self.embedding_size],\
                        initializer=self.weight_initialization,name="W_embedding")
            self.input_expand=tf.nn.embedding_lookup(self.W_embedding,self.input)
            self.input_shape=[-1,self.n_input,self.embedding_size]
        else:
            self.input=tf.placeholder(tf.float32,[None, self.n_input])
            self.input_expand=tf.expand_dims(self.input,-1)
            self.input_shape=[-1,self.n_input,1]

        self.output=tf.placeholder(tf.float32,[None, self.n_output])

    def build_bilstm(self,hidden_size=32,layer_num=1):
        inputs=tf.transpose(self.input_expand,[1,0,2])#[20,1,16]
        #print(inputs)        
        self.out_put_size_temp=[self.input_shape[1],self.input_shape[0],self.input_shape[2]]
        inputs=tf.reshape(inputs,[-1,self.out_put_size_temp[2]])
        inputs=tf.split(inputs,self.out_put_size_temp[0])
        #print(inputs)
        #exit()
        lstm_fw_cell=tf.contrib.rnn.BasicLSTMCell(hidden_size,forget_bias=1.0,state_is_tuple=True)
        lstm_bw_cell=tf.contrib.rnn.BasicLSTMCell(hidden_size,forget_bias=1.0,state_is_tuple=True)
        lstm_fw_cell=tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell,input_keep_prob=1.0,output_keep_prob=self._drop_out)
        lstm_bw_cell=tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell,input_keep_prob=1.0,output_keep_prob=self._drop_out)   
        cell_fw=tf.contrib.rnn.MultiRNNCell([lstm_fw_cell]*layer_num,state_is_tuple=True)
        cell_bw=tf.contrib.rnn.MultiRNNCell([lstm_bw_cell]*layer_num,state_is_tuple=True)  
        output,_,_=tf.contrib.rnn.static_bidirectional_rnn(cell_fw,cell_bw,inputs,dtype=tf.float32)
        #self.out_put_temp=tf.concat(output,1)
        #print(output)
        self.out_put_temp=output[-1]

#        self.out_put_temp=tf.reshape(self.out_put_temp,[-1,self.out_put_size_temp[0]*hidden_size*2])   
#        self.out_put_size_temp=[-1,self.out_put_size_temp[0]*hidden_size*2]

        print(self.out_put_temp)
        self.out_put_temp=tf.reshape(self.out_put_temp,[-1,hidden_size*2])   
        self.out_put_size_temp=[-1,hidden_size*2]

    def build_cnn(self,filter_sizes=[5,7,9],filter_nums=32):
        pooled_outputs = []
        self.out_put_temp=tf.expand_dims(self.input_expand,-1)
        self.out_put_size_temp=self.input_shape+[1]
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpoll-%s"%filter_size):
                W_filter = tf.get_variable(shape=[filter_size, self.out_put_size_temp[2], 1, filter_nums],
                                           initializer=self.weight_initialization, name="W_filter")
                bias = tf.get_variable(shape=[filter_nums], initializer=self._bias_initializer, name="bias")
                conv = tf.nn.conv2d(self.out_put_temp, W_filter, strides=[1,1,1,1],padding="VALID", name="conv")
                relu = tf.nn.relu(tf.nn.bias_add(conv,bias), name="relu")
                pooled = tf.nn.max_pool(relu, ksize=[1,self.n_input-filter_size+1,1,1 ],\
                         strides=[1,1,1,1],padding="VALID",name="pool")
                pooled_outputs.append(pooled)
        
        num_filters_total = filter_nums * len(filter_sizes)
        
        relu_poll = tf.concat(pooled_outputs,3)
        
        self.out_put_temp = relu_poll
        
        self.out_put_size_temp = [0, relu_poll.get_shape()[1].value,relu_poll.get_shape()[2].value,relu_poll.get_shape()[3].value]

        self.out_put_temp=tf.reshape(self.out_put_temp,[-1,self.out_put_size_temp[1]*self.out_put_size_temp[2]*self.out_put_size_temp[3]])
    def build_predict(self,):
        self.w_1=tf.get_variable('weight_1',
                    shape=[self.out_put_size_temp[-1],self.full_connect_size],
                    initializer=self.weight_initialization,
                    dtype=tf.float32
                    )
        self.b_1=tf.get_variable("bias_1",shape=[self.full_connect_size],
                               initializer=self.weight_initialization,#tf.constant([0.01],tf.float32),
                               dtype=tf.float32)
        self.hidden=tf.add(tf.matmul(self.out_put_temp, self.w_1), self.b_1)
        self.hidden=tf.nn.sigmoid(self.hidden)
        self.w_2=tf.get_variable('weight_2',
                    shape=[self.full_connect_size,self.n_output],
                    initializer=self.weight_initialization,
                    dtype=tf.float32
                    )
        self.b_2=tf.get_variable("bias_2",shape=[self.n_output],
                               initializer=self.weight_initialization,#tf.constant([0.01],tf.float32),
                               dtype=tf.float32)
        self.predict=tf.add(tf.matmul(self.hidden, self.w_2), self.b_2)

    def build_loss(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=self.predict))
            
if __name__=="__main__":      
    stru_config=C.stru_config
    y=neural_network(stru_config)
    print(y.predict)
    print(y.loss)
    print(self.out_put_temp.shape)