#fake price
# 未来收益由两部分决定，1）历史序列，2）位置因素，
#过去N天超过P天的收益为正，则，未来收益
import numpy as np
import random
import matplotlib.pyplot as plt
import json
def clip(lst,bound):
    ret=[]
    for c in lst:
        if c >bound:
            ret.append(bound)
        elif c<-bound:
            ret.append(-bound)
        else:
            ret.append(c)
    return ret
    
def fake_random(l):
    ret=np.random.normal(0,0.1,[l])
    ret_=clip(ret,0.1)
    price=[sum(ret_[:i]) for i in range(len(ret_))]
    return price

def signal_moment(lst,ratio):
    cnt_up=0
    cnt_down=0
    total = len(lst)
    for l in lst:
        if l>=0:
            cnt_up +=1
        else:
            cnt_down+=1
    if float(cnt_up)/total >ratio:
        return "up"
    elif float(cnt_down)/total >ratio:
        return "down"
    elif float(cnt_down)/total >0.95:
        return "turn"
    elif float(cnt_up)/total >0.95:
        return "turn_big"
    else:
        return "no"
def fake_moment(l,interval):
    ret=np.random.normal(0,0.1,[l])
    ret_=clip(ret,0.1)
    price=[]
    for i in range(interval,l):
        jude=ret_[i-interval:i]
        if signal_moment(jude,0.7)=="up":
            if random.random()>0.4:
                ret_[i]=abs(ret_[i])
        elif signal_moment(jude,0.7)=="down":
            if random.random()>0.4:
                ret_[i]=-abs(ret_[i])
        elif signal_moment(jude,0.7)=="turn":
            if random.random()>0.4:
                ret_[i]=abs(ret_[i])
        elif signal_moment(jude,0.7)=="turn_big":
            if random.random()>0.4:
                a=2*abs(ret_[i]) 
                ret_[i]=-a if a <0.1 else -0.1 
    
    price=[sum(ret_[:i]) for i in range(len(ret_))]
    return price
if __name__=="__main__":
    #with open("random","w",encoding="utf-8")as f:
    #    for j in range(3000):
    #        price_random=fake_random(900)
    #        f.write("%s\n"%json.dumps(price_random))
    #with open("momentum","w",encoding="utf-8")as f:
    #    for j in range(3000):
     #       price_random=fake_moment(900,20)
    #        f.write("%s\n"%json.dumps(price_random))
    price_random=fake_random(900)
    price_momentum=fake_moment(900,20)
    plt.plot(price_random)
    plt.plot(price_momentum)
    plt.legend(["random","momentum"])
    plt.show()
    