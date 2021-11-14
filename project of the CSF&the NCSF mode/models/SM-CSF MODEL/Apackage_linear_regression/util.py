import numpy as np 
import matplotlib.pyplot as plt
import sys
import json
import glob
import config as C

class give_batch():
    train_path=C.train_path
    test_path=C.test_path
    #train_path_list=glob.glob(train_path+"/*")
    #test_path_list=glob.glob(test_path+"/*")
    def __init__(self,train_path,test_path,label2id={"up":0,"down":1}):
        self.train_path=C.train_path
        self.test_path=C.test_path
        self.label2id=label2id
        self.load_train_test()
    def load_train_test(self):
        self.train_x,self.train_y=[],[]
        self.test_x,self.test_y=[],[]
        for line in open(self.train_path,"r",encoding="utf-8"):
            feature,label,original_feature=line.strip().split("\t")
            feature=json.loads(feature)
            fea=[float(x[1])for x in feature.items()]
            self.train_x.append(fea)
            self.train_y.append(self.to_one_hot(self.label2id[label]))
        for line in open(self.test_path,"r",encoding="utf-8"):
            feature,label,original_feature=line.strip().split("\t")
            feature=json.loads(feature)
            fea=[float(x[1])for x in feature.items()]
            self.test_x.append(fea)
            self.test_y.append(self.to_one_hot(self.label2id[label]))
        self.total_sample=len(self.train_x)
        self.index_pool=[int(x)for x in range(self.total_sample)]
    def to_one_hot(self,num):
        blank=[0]*len(self.label2id)
        blank[num]=1
        return blank
    
    def do(self, batchsize):
        this_index=np.random.choice(self.index_pool,[batchsize])
        x = []#np.random.uniform(self.x_range[0], self.x_range[1],batchsize)    
        y=[]
        for index in this_index:
            x.append(self.train_x[index])
            y.append(self.train_y[index])
        return x,y
        
        
if __name__ == '__main__':
    #for train,test in zip(train_path_list,test_path_list):
    D=give_batch(train_path,test_path)
    x,y=D.do(4)
    for xx,yy in zip(x,y):
        print(xx)
        print(yy)
