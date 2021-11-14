#fake price
# 未来收益由两部分决定，1）历史序列，2）位置因素，
#过去N天超过P天的收益为正，则，未来收益
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import pickle
import sys
import config as C
import functions as F
def load_data(path):
    
    feature_label=[]
    for line in open(path,"r",encoding="utf-8"):
        fea,la=line.strip().split("\t")
        fea=[float(x)for x in fea.split("<=>")]#json.loads(fea)
        
        
        feature_label.append([fea,t_label2ind[la]])
    return feature_label


if __name__=="__main__":

    file_name=sys.argv[1]
    t_label2ind={"up":0,"down":1,"middle":2}
    feature_label_test=load_data(file_name)
    batch=[x[0]for x in feature_label_test]
    label=[x[1]for x in feature_label_test]
    ratio=C.ratio
    prob=C.prob
    turn=C.turn
    label_pre=[]
    for ba in batch:
        ba_=[0]+ba
        ba_=[ba_[i+1]-ba_[i] for i in range(len(ba))]
        sig=F.signal_moment(ba_,ratio,turn)
        if sig in ["up","turn_up"]:
            label_pre.append(t_label2ind["up"])
        elif sig in ["down","turn_down"]:
            label_pre.append(t_label2ind["down"])
        else:
            label_pre.append(t_label2ind["middle"])

    acc_up=[x==y for x,y in zip(label,label_pre) if y==0]
    acc_down=[x==y for x,y in zip(label,label_pre) if y==1]
    print(file_name)
    print("原始标签中 0的绝对数量",label.count(0))
    print("原始标签中 1的绝对数量",label.count(1))
    print("总标签数量",len(label))
    print("原始标签中 0的占比",label.count(0)/len(label))
    print("原始标签中 1的占比",label.count(1)/len(label))
    print("-------------------------")
    print("预测标签中0的绝对数量",label_pre.count(0))
    print("预测标签中1的绝对数量",label_pre.count(1))
    print("预测标签中2的绝对数量",label_pre.count(2))
    print("预测标签中 0的占比",label_pre.count(0)/len(label_pre))
    print("预测标签中 1的占比",label_pre.count(1)/len(label_pre))
    print("-------------------------")
    print("预测上涨的正确数量",acc_up.count(True))
    print("预测上涨真实上涨的比例",acc_up.count(True)/len(acc_up))
    print("预测下跌的正确数量",acc_down.count(True))
    print("预测下跌真实下跌的比例",acc_down.count(True)/len(acc_down))