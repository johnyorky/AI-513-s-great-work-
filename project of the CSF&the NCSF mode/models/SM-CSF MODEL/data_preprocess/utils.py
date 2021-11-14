import numpy as np
import os
import sys
import pickle
def do_one(data,kernel_size=5,stride=2):
    L=len(data)
    sec=int(kernel_size/2)
    feature_list=[]
    for i in range(sec,L-sec-1,stride):
        this_data=data[i-sec:i+sec+1]
        this_feature=[]
        #print(this_data)
        #exit()
        for i in range(len(this_data)-1):
            if this_data[i+1]-this_data[i]>0:
                this_feature.append("U")
            else:
                this_feature.append("D")
            
        feature_list.append("_".join(this_feature))
    return feature_list

if __name__=="__main__":
    path="/home/yangjiayao/paper_1/datas/combine/test/xy_test_combine"
    feature_dict={}
    for line in open(path,"r",encoding="utf-8"):
        feature,label=line.strip().split("\t")
        datas=[float(x)for x in feature.split("<=>")]
        fea=do_one(datas)
        #print(fea)
        for f in fea:
            feature_dict[f]=feature_dict.get(f,0)+1

        #exit()
    print(feature_dict)
    
