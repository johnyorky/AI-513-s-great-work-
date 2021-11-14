#fake price
# 未来收益由两部分决定，1）历史序列，2）位置因素，
#过去N天超过P天的收益为正，则，未来收益
import numpy as np
import random
import matplotlib.pyplot as plt
import json,sys
import config as C
import functions as F
import utils as U
import glob

def give_corr(signal,cut):
    ret=[]
    num_signal=len(signal)
    for i in range(0,cut):
        tmp=0
        cnt=0
        for j in range(num_signal-i):
            tmp +=signal[j]*signal[j+i]
            cnt +=1
        ret.append(tmp/cnt)
    return ret

def load_feature2weight():
    t={}
    if glob.glob("weight_config")==[]:
       print("请先生成weight_config文件")
       exit()
    for line in open("weight_config","r",encoding="utf-8"):
       ff,pp=line.strip().split("\t")
       t[ff]=float(pp)
    return t

def get_score(seq,w):
    features=U.do_one(seq,w,1)
    score=0
    for ff in features:
        score += t_feature2weight[ff]
    return score

def fake_combine(l,interval,turn,std,ratio,prob):
    ret=np.random.normal(0,std,[l])
    ret_=F.clip(ret,0.1)
    price=[]
    score_lst=[]
    for i in range(interval,l):
        jude=ret_[i-interval-1:i-1]
        jude=[sum(jude[:i]) for i in range(1,len(jude))]
        score=0
        for w in C.windows:
            score +=get_score(jude,w)
        score_lst.append(score)
        if score >=C.score_thred:
            if random.random()>prob:
                ret_[i]=abs(ret_[i])
        else:
            if random.random()>prob:
                ret_[i]=-abs(ret_[i])
    price=[sum(ret_[:i]) for i in range(len(ret_))]
    print(sum(score_lst)/len(score_lst))
 
    return price

if __name__=="__main__":

    t_feature2weight=load_feature2weight()
    num=sys.argv[1]
    num=int(num)
    ratio=C.ratio#0.8
    prob=C.prob#1-0.4
    turn=C.turn
    std=C.std
    L=C.L
    inter=C.inter
    with open("combine","w",encoding="utf-8")as f:
        for j in range(num):
            price_combine=fake_combine(L,inter,turn,std,ratio,prob)
            f.write("%s\n"%json.dumps(price_combine))
    with open("random","w",encoding="utf-8")as f:
        for j in range(num):
            price_random=F.fake_random(L,std)
            f.write("%s\n"%json.dumps(price_random))
    cor_random=give_corr(price_random,int(L/4))
    
    #price_combine=F.fake_moment(L,inter,turn,std,ratio,prob)
    cor_combine=give_corr(price_combine,int(L/4))
    plt.plot(price_random)
    plt.plot(price_combine)
    plt.legend(["random","combine"])
    plt.show()
    plt.plot(cor_random)
    plt.plot(cor_combine)
    plt.legend(["random","combine"])
    plt.show()
    