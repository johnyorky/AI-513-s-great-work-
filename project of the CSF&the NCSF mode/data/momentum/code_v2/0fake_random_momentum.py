#fake price
# 未来收益由两部分决定，1）历史序列，2）位置因素，
#过去N天超过P天的收益为正，则，未来收益
import numpy as np
import random
import matplotlib.pyplot as plt
import json,sys
import config as C
import functions as F
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
if __name__=="__main__":
    #price_random=fake_moment(100,10)
    #print(price_random)
    #exit()
    
    num=sys.argv[1]
    num=int(num)
    ratio=C.ratio#0.8
    prob=C.prob#1-0.4
    turn=C.turn
    std=C.std
    L=C.L
    inter=C.inter
    with open("momentum","w",encoding="utf-8")as f:
        for j in range(num):
            price_random=F.fake_moment(L,inter,turn,std,ratio,prob)
            f.write("%s\n"%json.dumps(price_random))
    with open("random","w",encoding="utf-8")as f:
        for j in range(num):
            price_random=F.fake_random(L,std)
            f.write("%s\n"%json.dumps(price_random))
    #print(price_random)
    #exit()
    cor_random=give_corr(price_random,int(L/4))
    
    price_momentum=F.fake_moment(L,inter,turn,std,ratio,prob)
    cor_moment=give_corr(price_momentum,int(L/4))
    plt.plot(price_random)
    plt.plot(price_momentum)
    plt.legend(["random","momentum"])
    plt.show()
    plt.plot(cor_random)
    plt.plot(cor_moment)
    plt.legend(["random","momentum"])
    plt.show()
    