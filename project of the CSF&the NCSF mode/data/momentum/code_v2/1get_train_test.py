#fake price
# 未来收益由两部分决定，1）历史序列，2）位置因素，
#过去N天超过P天的收益为正，则，未来收益
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import config as C

if __name__=="__main__":
    thre=0
    up_cnt=0
    down_cnt=0
    with open("xy_train_random","w",encoding="utf-8")as f:
        for line in open("random","r",encoding="utf-8"):
            data=json.loads(line.strip())
            for i in range(C.inter-1,len(data)-1):
                feature=[x for x in data[i-C.inter+2:i+1]]
                 
                if (data[i+1]-data[i])/data[i]>=thre:
                    label="up"
                    up_cnt +=1
                elif (data[i+1]-data[i])/data[i]<-thre:
                    label="down"
                    down_cnt +=1
                else:
                    label="middle"
                f.write("%s\t%s\n"%("<=>".join([str(x)for x in feature]),label))
    print("random up %s down %s"%(up_cnt,down_cnt))
    up_cnt=0
    down_cnt=0
    with open("xy_train_momentum","w",encoding="utf-8")as f:
        for line in open("momentum","r",encoding="utf-8"):
            data=json.loads(line.strip())
            for i in range(C.inter-1,len(data)-1):
                feature=[x for x in data[i-C.inter+2:i+1]]
                
                
                
                if (data[i+1]-data[i])/abs(data[i])>=thre:
                    label="up"
                    up_cnt +=1
                elif (data[i+1]-data[i])/abs(data[i])<=-thre:
                    label="down"
                    down_cnt +=1
                else:
                    label="middle"
               
                f.write("%s\t%s\n"%("<=>".join([str(x)for x in feature]),label))
    print("momentum up %s down %s"%(up_cnt,down_cnt))
    
