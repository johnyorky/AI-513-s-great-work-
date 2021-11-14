import numpy as np
import random
import utils as U
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
    
def fake_random(l,std):
    ret=np.random.normal(0,std,[l])
    ret_=clip(ret,0.1)
    price=[sum(ret_[:i]) for i in range(len(ret_))]
    return price

def signal_moment(lst,ratio,turn):
    cnt_up=0
    cnt_down=0
    total = len(lst)
    for l in lst:
        if l>=0:
            cnt_up +=1
        else:
            cnt_down+=1
    if float(cnt_down)/total >ratio and float(cnt_down)/total <=turn:
        return "down"
    elif float(cnt_up)/total >ratio and float(cnt_up)/total <=turn:
        return "up"
    elif float(cnt_down)/total >turn:
        return "turn_up"
    elif float(cnt_up)/total >turn:
        return "turn_down"
    else:
        return "no"
def fake_moment(l,interval,turn,std,ratio,prob):
    ret=np.random.normal(0,std,[l])
    ret_=clip(ret,0.1)
    price=[]
    for i in range(interval,l):
        jude=ret_[i-interval:i-1]
        sig=signal_moment(jude,ratio,turn)
        if sig=="up" or sig=="turn_up":
            if random.random()>prob:
                ret_[i]=abs(ret_[i])
        elif sig=="down" or sig=="turn_down":
            if random.random()>prob:
                ret_[i]=-abs(ret_[i])
    price=[sum(ret_[:i]) for i in range(len(ret_))]
    return price
def get_score(seq):
    features=U.do_one(seq,5,1)
    score=0
    for ff in features:
        score += t_feature2weight[ff]
    return score
def fake_combine(l,interval,turn,std,ratio,prob):
    ret=np.random.normal(0,std,[l])
    ret_=clip(ret,0.1)
    price=[]
    for i in range(interval,l):
        jude=ret_[i-interval:i-1]
        score=get_score(jude)
        if score >0.2:
            if random.random()>prob:
                ret_[i]=abs(ret_[i])
    price=[sum(ret_[:i]) for i in range(len(ret_))]
    return price