#-*-coding:utf-8-*-
import glob
import os,sys
import time
import pdb
import pickle
import logging
import re
import random
import json
import argparse
import utils as U
import config as C
def handle_one(seq,window_sizes=[[5,1]]):
    ret={}
    for window,stride in window_sizes:
        features=U.do_one(seq,window,stride)
        for feature in features:
            ret[feature]=ret.get(feature,0)+1
    return ret

def handle_all(seqs,window_sizes=[[5,1]]):
    ret={}
    for seq in seqs:
        tmp=handle_one(seq,window_sizes)
        for feature,cnt in tmp.items():
            ret[feature]=ret.get(feature,0)+cnt
    return ret

def save_dict(dic,name):
    with open(name,"w",encoding="utf-8")as f:
        for fea,cnt in sorted(dic.items(),key=lambda x:x[1],reverse=True):
            f.write("%s\t%s\n"%(fea,cnt))

def load_train(file_path):
    ret={}
    for line in open(file_path,"r",encoding="utf-8"):
        seq,label=line.strip().split("\t")
        seq=[float(x)for x in seq.split("<=>")]
        xx=len(seq)
        if label not in ret:
            ret[label]=[]
        ret[label].append(seq)
    return ret
def save_dict_pickle(dic,name):
    with open(name,"wb")as f:
        pickle.dump(dic,f)


if __name__ == '__main__':
    file_path=C.train_path#"/home/yangjiayao/paper_1/datas/combine/train"
    train_path_list=glob.glob(file_path+"/*")
    for train in train_path_list:
        train1=train.split("/")[-1]
        print(train1)
        threth=C.threth#1.5
        window_sizes=C.window_sizes
        t_label2seqs=load_train(train)
        t_label2features2cnt={}
        labels_dic={}
        features_dic={}
        for label,seqs in t_label2seqs.items():
            features=handle_all(seqs,window_sizes)
            t_label2features2cnt[label]=features
            save_dict(features,"result/train_%s"%train1+label)
            labels_dic[label]=1
            for fea in features.keys():
                features_dic[fea]=1
        t_statistical={}
        t_statistical_effective={}
        for fea in features_dic.keys():
            cnt={}
            cnt_lst=[]
            for label in labels_dic.keys():
                tmp_x=t_label2features2cnt[label].get(fea,0)
                cnt[label]=tmp_x#t_label2features2cnt[label].get(fea,0)
                cnt_lst.append(tmp_x)
            ratio=max(cnt_lst)/(min(cnt_lst)+1)
            cnt["ratio"]=ratio#max(cnt_lst)/(min(cnt_lst)+1)
            if ratio >= threth:
                t_statistical_effective[fea]=cnt
            t_statistical[fea]=cnt
        save_dict_pickle(t_statistical,"result/t_stati_%s"%train1)
        save_dict_pickle(t_statistical_effective,"result/t_stati_effective_%s"%train1)

