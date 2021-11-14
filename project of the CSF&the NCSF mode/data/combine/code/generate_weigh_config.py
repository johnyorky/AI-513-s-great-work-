import config as C
import numpy as np
windows=C.windows
pool=["U","D"]

def generate_feature(num):
    num=num-1
    N=2**num
    B_lst=[]
    for n in range(N):
        B=bin(n)
        B=B[2:]
        B="".join(["0"]*(num-len(B)))+B
        B="_".join([x for x in B])
        B=B.replace("0","U")
        B=B.replace("1","D")
        B_lst.append(B)
    return B_lst
    
feature_list=[]
for w in windows:
    feature_list +=generate_feature(w)
L=len(feature_list)
pro_list=np.random.normal(-1,5,[L,])
pro_list[pro_list <0]=0
pro_list=pro_list/np.sum(pro_list)
with open("weight_config","w",encoding="utf-8")as f:
    for ff,pp in zip(feature_list,pro_list):
        f.write("%s\t%s\n"%(ff,pp))
 
            