#!/usr/bin/python
#-*-coding:utf-8-*-
import os,sys
import time
import pdb
import pickle
import logging
import re
import glob
import random
import json
import utils as U
import config as C
def handle_one(seq,effective_features,window_sizes=[[10,1]]):
    ret={}
    for window,stride in window_sizes:
        features=U.do_one(seq,window,stride)
        for feature in effective_features:
            ret[feature]=features.count(feature)#ret.get(feature,0)+1
    return ret


def load_effective_features():
    if glob.glob("result/t_statistical_effective_%s"%train1)==[]:
        print("t_statistical_effective is missing!!!")
        exit()
    data=[]
    with open("result/t_statistical_effective_%s"%train1,"rb")as f:
        data=pickle.load(f)
    return data

def main(train,save_name,effective_features,window_sizes):
    with open(save_name,"w",encoding="utf-8")as f:
        for line in open(train,"r",encoding="utf-8"):
            seqs,label=line.strip().split("\t")
            
            seq=[float(x)for x in seqs.split("<=>")]
            encode=handle_one(seq,effective_features,window_sizes)
            f.write("%s\t%s\t%s\n"%(json.dumps(encode),label,seqs))
    


if __name__=="__main__":
    train_path=C.train_path#"/home/yangjiayao/paper_1/datas/combine/train/xy_train_combine_5000"
    train_path_list=glob.glob(train_path+"/*")
    test_path=C.test_path
    for train in train_path_list:
        train1=train.split("/")[-1]
        train2=train1.split("_")[-1]
        window_sizes=C.window_sizes
        effective_features=load_effective_features()
        main(train,"result/xy_train_%s"%train2,effective_features,window_sizes)
        main(test_path,"result/xy_test_%s"%train2,effective_features,window_sizes)

