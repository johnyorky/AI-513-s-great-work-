import numpy as np 
import matplotlib.pyplot as plt
import sys
import json
import pickle
import glob
import config as C

class give_batch():
    train_path=C.train_path
    test_path=C.data_test
    train_path_list=glob.glob(train_path+"/*")
    #for train in train_path_list:

    def __init__(self,train,test_path,resolution,label2id={"up":0,"down":1}):
        self.train=train
        self.test_path =test_path
        self.label2id=label2id
        self.resolution=resolution
        self.load_train_test()
    def norm(self,seq):
        max_=max(seq)
        min_=min(seq)
        ret=[int((x-min_)/(max_-min_)*self.resolution) for x in seq]
        return ret
    def load_train_test(self):
        self.train_x,self.train_x_norm,self.train_y=[],[],[]
        self.test_x,self.test_x_norm,self.test_y=[],[],[]
        for line in open(self.train,"r",encoding="utf-8"):
            feature,label=line.strip().split("\t")
            feature=[float(x)for x in feature.split("<=>")]#json.loads(feature)
            feature_norm=self.norm(feature)
            self.train_x.append(feature)
            self.train_x_norm.append(feature_norm)
            self.train_y.append(self.to_one_hot(self.label2id[label]))
        for line in open(self.test_path,"r",encoding="utf-8"):
            feature,label=line.strip().split("\t")
            feature=[float(x)for x in feature.split("<=>")]#json.loads(feature)
            feature_norm=self.norm(feature)
            self.test_x.append(feature)
            self.test_x_norm.append(feature_norm)
            self.test_y.append(self.to_one_hot(self.label2id[label]))
        self.total_sample=len(self.train_x)
        self.index_pool=[int(x)for x in range(self.total_sample)]
    def to_one_hot(self,num):
        blank=[0]*len(self.label2id)
        blank[num]=1
        return blank
    
    def do_norm(self, batchsize):
        this_index=np.random.choice(self.index_pool,[batchsize])
        x = []#np.random.uniform(self.x_range[0], self.x_range[1],batchsize)    
        y=[]
        for index in this_index:
            x.append(self.train_x_norm[index])
            y.append(self.train_y[index])
        return x,y

    def do(self, batchsize):
        this_index=np.random.choice(self.index_pool,[batchsize])
        x = []#np.random.uniform(self.x_range[0], self.x_range[1],batchsize)    
        y=[]
        for index in this_index:
            x.append(self.train_x[index])
            y.append(self.train_y[index])
        return x,y

       
if __name__ == '__main__':
    for train in train_path_list:   
        D=give_batch(train,test_path)
        x,y=D.do(4)
        for xx,yy in zip(x,y):
            print(xx)
            print(yy)
        x,y=D.do_norm(4)
        for xx,yy in zip(x,y):
            print(xx)
            print(yy)
