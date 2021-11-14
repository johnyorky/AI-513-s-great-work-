import tensorflow as tf 
import logging
import numpy as np 
import model
import matplotlib.pyplot as plt
import sys
from util import give_batch
import config as C
import time    
import os
import pickle
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
tf.reset_default_graph()
class the_net():
    def __init__(self,train_config,stru_config):
        for item,value in train_config.items():
            print(item)
            print(value)
        self.logger = self.log_config()
        self.save_path=train_config['CKPT']       #存储操作
        self.learning_rate=train_config["LEARNING_RATE"]
        self.batch_size=train_config["BATCHSIZE"]
        self.max_iter=train_config["MAX_ITER"]
        self.step_save=train_config["STEP_SAVE"]     #1
        self.step_show=train_config['STEP_SHOW']        #30
        self.global_steps = tf.Variable(0,trainable=False)    #定义global_steps的初始值】
        self.stru_config=stru_config      #应该是赋值把
        self.is_embedding=C.stru_config["embedding"]
        self.D=give_batch(train,C.data_test,C.stru_config["num_size"])

        self.sess=tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=1, 
            intra_op_parallelism_threads=1,
        ))
        
        self.build_net()
        self.build_opt()
        self.saver=tf.train.Saver(max_to_keep=1)
        self.initialize()
    #tf.reset_default_graph()    
    def build_net(self):
        self.y = model.neural_network(self.stru_config)
        self.loss=self.y.loss#tf.reduce_mean(tf.pow(self.loss_function,2))
    def build_opt(self):
        #self.learning_rate = tf.train.exponential_decay(self.learning_rate, \
        #                        self.global_steps, C.decay_step, C.decay_rate, staircase=True)

        self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                .minimize(self.loss,global_step=self.global_steps)
        
    def initialize(self):             
        print("初始化我想多看看")
        ckpt=tf.train.latest_checkpoint(self.save_path)     #自动找到最近保存的变量文件
        if ckpt!=None:                                      #不等于
            self.saver.restore(self.sess,ckpt)              #打开ckpt路径
        else:
            self.sess.run(tf.global_variables_initializer()) #
    def log_config(self):
        logger = logging.getLogger("%s_log.txt"%C.stru_config["which_model"])
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler("log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    def log(self,to_log):
        self.logger.info(to_log)
    def get_trainable_variables(self):       #查看图计算节点的值0
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  
    def train(self):
        st=time.time()
        for step in range(self.max_iter):
            
            x,y=self.D.do_norm(self.batch_size) if self.is_embedding else self.D.do(self.batch_size)     #在前面给定范围，最终输出一个以batch_size给定样本数量的矩阵
            #print(len(y))
            #exit()
            loss,_,gs=self.sess.run([self.loss,self.opt,self.global_steps],\
                    feed_dict={self.y.input:x,self.y.output:y})
            if (step+1)%self.step_show==0:
                to_print="loss %s,in global step %s, \
                             taks %s seconds"%\
                            (loss,gs,time.time()-st)
                predict=self.sess.run(self.y.predict,feed_dict={self.y.input:x,self.y.output:y})
                self.get_acc(y,predict,name="Train==>")
                print(to_print)
                self.logger.info(to_print)
                st=time.time()                 
                self.test() 
            if (step+1)%self.step_save==0:
                self.saver.save(self.sess, self.save_path+"/check.ckpt")
    def get_acc(self,y,predict,name="Test==>"):
        #print(predict)
        #exit()
        x1=self.D.train_x
        size=len(x1)
        total=len(y)
        total_up=0
        pre_up=0
        #data={}
        pre_up_right=0
        for real,pre in zip(y,predict):
            real=list(real).index(max(real))
            pre=list(pre).index(max(pre))
            if real==0:
                total_up +=1
                if real==pre:
                    pre_up_right +=1
            if pre==0:
                pre_up+=1
        to_print=name+"total: %s, total_up: %s, ratio_total: %s,pre_up: %s, pre_up_right: %s,ratio_pre: %s,ratio_reduce: %s"%(total,total_up,total_up/total\
                ,pre_up,pre_up_right,pre_up_right/(pre_up+1),abs((pre_up_right/(pre_up+1))-(total_up/total)))
        #data["test"]={"up":[pre_up_right/(pre_up+1)]}
        #data["test"]["base"]=[total_up/total,abs((pre_up_right/(pre_up+1))-(total_up/total))]
        #with open("test_result/data_result_%s"%size,"wb")as f:
            #pickle.dump(data,f)
        print2="total: %s, total_up: %s, ratio_total: %s,pre_up: %s, pre_up_right: %s,ratio_pre: %s,ratio_reduce: %s"%(total,total_up,total_up/total\
                ,pre_up,pre_up_right,pre_up_right/(pre_up+1),abs((pre_up_right/(pre_up+1))-(total_up/total)))    
        with open("test_result_lstm/result_%s"%size,"w")as f:
            f.write("total: %s, total_up: %s, ratio_total: %s,pre_up: %s, pre_up_right: %s,ratio_pre: %s,ratio_reduce: %s"%(total,total_up,total_up/total,pre_up,pre_up_right,pre_up_right/(pre_up+1),abs((pre_up_right/(pre_up+1))-(total_up/total))))
        print(to_print)
        self.logger.info(to_print)

    def test(self):
        if self.is_embedding:
            x=self.D.test_x_norm
        else:
            x=self.D.test_x
        y=self.D.test_y
        predict=self.sess.run(self.y.predict,feed_dict={self.y.input:x,self.y.output:y})
        self.get_acc(y,predict)
            
def show_parameter_count(variables):
    print("---------------统计模型参数--------------")
    total_parameters = 0
    for variable in variables:
        name = variable.name
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
            print('{}: {} ({} parameters)'.format(name,shape,variable_parametes))
            total_parameters += variable_parametes
    to_print='Total: {} parameters'.format(total_parameters)
    print(to_print)
    return to_print
    
if __name__=="__main__":
    train_path=C.train_path
    test_path=C.data_test
    train_path_list=glob.glob(train_path+"/*")
    for train in train_path_list:
        print(train)
        tf.reset_default_graph()
        main_net=the_net(C.train_config,C.stru_config)
        all_variable=main_net.get_trainable_variables()
        to_print=show_parameter_count(all_variable)
        main_net.log(to_print)
        main_net.train()
